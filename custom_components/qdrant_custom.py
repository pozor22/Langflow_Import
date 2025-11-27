# from langflow.field_typing import Data
from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output, StrInput, HandleInput, DataFrameInput
from langflow.schema.data import Data
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
import os
from datetime import datetime
from typing import Dict, List
import hashlib


class QdrantCustom(Component):
    display_name = "Qdrant Custom"
    description = "Use as a template to create your own component."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Qdrant"
    name = "QdrantCustom"

    inputs = [
        StrInput(name="collection_name", display_name="Collection Name", required=True),
        StrInput(name="url", display_name="URL"),
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        DataFrameInput(
            name="ingest_data",
            display_name="Ingest Data",
            required=True
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def init_qdrant(self) -> QdrantClient:
        qdrant_client = QdrantClient(url=self.url)
        return qdrant_client

    def get_file_modification_time(self, file_path: str) -> float:
        """Получить время модификации файла"""
        try:
            if os.path.exists(file_path):
                return os.path.getmtime(file_path)
            else:
                return 0.0
        except Exception as e:
            return 0.0

    def get_existing_file_times(self, client: QdrantClient, collection_name: str) -> Dict[str, float]:
        """Получает время модификации файлов из существующих точек в Qdrant"""
        existing_files = {}

        try:
            # Используем scroll для получения всех точек
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )

            for point in scroll_result[0]:
                if 'source' in point.payload and 'modification_time' in point.payload:
                    filename = point.payload['source']
                    existing_time = point.payload['modification_time']
                    existing_files[filename] = existing_time

        except Exception as e:
            pass

        return existing_files

    def create_collection_if_not_exists(self, client: QdrantClient, collection_name: str, vector_size: int) -> bool:
        """Создает коллекцию в Qdrant, если она не существует"""
        try:
            if not client.collection_exists(collection_name):
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0)
                )
                return True
            else:
                return False
        except Exception as e:
            return False

    def delete_points_by_source(self, client: QdrantClient, collection_name: str, filename: str) -> bool:
        """Удаляет все точки, связанные с указанным файлом"""
        try:
            client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchValue(value=filename)
                        )
                    ]
                )
            )
            return True
        except Exception as e:
            return False

    def generate_point_id(self, text: str, source: str, chunk_index: int) -> int:
        """Генерирует ID для точки на основе текста, источника и индекса чанка"""
        unique_string = f"{source}_{chunk_index}_{text}"
        # Используем хэш для создания числового ID
        hash_object = hashlib.md5(unique_string.encode())
        # Берем первые 8 байт хэша и преобразуем в int
        return int(hash_object.hexdigest()[:16], 16) % (2 ** 63)

    def build_output(self) -> Data:
        # Инициализируем Qdrant клиент
        qdrant_client = self.init_qdrant()
        df = self.ingest_data

        # Проверяем, существует ли коллекция
        collection_exists = qdrant_client.collection_exists(self.collection_name)

        # Получаем документы из DataFrame
        documents = []
        for index, row in df.iterrows():
            file_path = row['file_path']
            text = row['text']

            # Получаем время модификации файла
            modification_time = self.get_file_modification_time(file_path)

            # Используем имя файла как источник
            source = os.path.basename(file_path)

            documents.append({
                'file_path': file_path,
                'text': text,
                'source': source,
                'modification_time': modification_time,
                'chunk_index': index
            })

        if not documents:
            return Data(text="No documents to process", data={"processed": 0})

        # Если коллекция существует, проверяем какие файлы нужно обновить
        files_to_process = []
        if collection_exists:
            # Получаем информацию о существующих файлах из Qdrant
            existing_files = self.get_existing_file_times(qdrant_client, self.collection_name)

            # Группируем документы по файлам
            files_dict = {}
            for doc in documents:
                filename = doc['source']
                if filename not in files_dict:
                    files_dict[filename] = []
                files_dict[filename].append(doc)

            # Определяем какие файлы новые или измененные
            for filename, file_docs in files_dict.items():
                if filename not in existing_files:
                    files_to_process.extend(file_docs)
                else:
                    # Сравниваем время модификации
                    current_time = file_docs[0]['modification_time']  # Время одинаково для всех чанков
                    stored_time = existing_files[filename]

                    if current_time != stored_time:
                        # Удаляем старые точки этого файла
                        self.delete_points_by_source(qdrant_client, self.collection_name, filename)
                        files_to_process.extend(file_docs)
                    else:
                        pass
        else:
            # Если коллекции нет, обрабатываем все файлы
            files_to_process = documents

        if not files_to_process:
            return Data(text="No new or changed files to process", data={"processed": 0})

        # Создаем эмбеддинг для первого документа, чтобы определить размер вектора
        try:
            sample_embedding = self.embedding.embed_query(files_to_process[0]['text'])
            vector_size = len(sample_embedding)
        except Exception as e:
            return Data(text=f"Error: {str(e)}", data={"error": str(e)})

        # Создаем коллекцию (если не существует)
        self.create_collection_if_not_exists(qdrant_client, self.collection_name, vector_size)

        points = []
        # Генерируем эмбеддинги для документов, требующих обработки
        for i, doc in enumerate(files_to_process):
            try:
                embedding = self.embedding.embed_query(doc['text'])

                points.append(PointStruct(
                    id=self.generate_point_id(doc['text'], doc['source'], doc['chunk_index']),
                    vector=embedding,
                    payload={
                        "text": doc['text'],
                        "source": doc['source'],
                        "file_path": doc['file_path'],
                        "chunk_index": doc['chunk_index'],
                        "modification_time": doc['modification_time'],
                        "modification_time_readable": datetime.fromtimestamp(doc['modification_time']).strftime(
                            '%Y-%m-%d %H:%M:%S') if doc['modification_time'] > 0 else "Unknown"
                    }
                ))
            except Exception as e:
                continue

        # Загружаем точки в Qdrant используя upsert
        if points:
            try:
                qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

                # Включаем индексацию для поиска (если была отключена)
                qdrant_client.update_collection(
                    collection_name=self.collection_name,
                    hnsw_config=models.HnswConfigDiff(m=16)
                )
            except Exception as e:
                return Data(text=f"Error: {str(e)}", data={"error": str(e)})
        else:
            pass

        # Возвращаем результат с информацией о времени модификации
        file_paths = df['file_path'].tolist()
        texts = df['text'].tolist()
        modification_times = [self.get_file_modification_time(file_path) for file_path in file_paths]
        modification_times_readable = [
            datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S') if time > 0 else "Unknown"
            for time in modification_times
        ]

        return Data(
            text=f"Processed {len(points)} chunks to Qdrant",
            data={
                "file_paths": file_paths,
                "texts": texts,
                "modification_times": modification_times,
                "modification_times_readable": modification_times_readable,
                "processed_chunks": len(points),
                "collection_name": self.collection_name
            }
        )

