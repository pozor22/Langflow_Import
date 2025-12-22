# from langflow.field_typing import Data
from langflow.custom.custom_component.component import Component
from langflow.io import Output, StrInput, HandleInput, DataFrameInput, IntInput
from langflow.schema.data import Data
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
import os
from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import pandas as pd


class QdrantCustom(Component):
    display_name = "Qdrant Custom"
    description = "Custom Qdrant component with HNSW configuration"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Qdrant"
    name = "QdrantCustom"

    inputs = [
        StrInput(name="collection_name", display_name="Collection Name", required=True),
        StrInput(name="url", display_name="URL"),
        StrInput(name="api_key", display_name="API Key", info="Qdrant API Key for authentication"),
        HandleInput(name="embedding", display_name="Embedding", input_types=["Embeddings"]),
        DataFrameInput(
            name="ingest_data",
            display_name="Ingest Data",
            required=True
        ),
        IntInput(
            name="hnsw_m",
            display_name="HNSW M (max connections per node)",
            info="Maximum number of connections per node in the HNSW graph. Default is 16.",
            value=16,
            advanced=False
        ),
        IntInput(
            name="hnsw_ef_construct",
            display_name="HNSW EF Construct",
            info="Size of the candidate list during index construction. Default is 100.",
            value=100,
            advanced=False
        ),
        IntInput(
            name="full_scan_threshold",
            display_name="Full Scan Threshold",
            info="Threshold for switching to full scan. Default is 10000.",
            value=10000,
            advanced=False
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def init_qdrant(self) -> QdrantClient:
        if self.api_key:
            qdrant_client = QdrantClient(
                url=self.url,
                api_key=self.api_key
            )
        else:
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
                hnsw_config = models.HnswConfigDiff(
                    m=self.hnsw_m,
                    ef_construct=self.hnsw_ef_construct,
                    full_scan_threshold=self.full_scan_threshold
                )

                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    ),
                    hnsw_config=hnsw_config
                )
                return True
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
        hash_object = hashlib.md5(unique_string.encode())
        return int(hash_object.hexdigest()[:16], 16) % (2 ** 63)

    def generate_content_hash(self, text: str) -> str:
        """Генерирует хэш содержимого для проверки изменений"""
        return hashlib.md5(text.encode()).hexdigest()

    def get_existing_content_hashes(self, client: QdrantClient, collection_name: str) -> Dict[str, str]:
        """Получает хэши содержимого из существующих точек в Qdrant"""
        existing_hashes = {}

        try:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )

            for point in scroll_result[0]:
                if 'source' in point.payload and 'content_hash' in point.payload:
                    filename = point.payload['source']
                    existing_hash = point.payload['content_hash']
                    existing_hashes[filename] = existing_hash

        except Exception as e:
            pass

        return existing_hashes

    def build_output(self) -> Data:
        # Инициализируем Qdrant клиент
        qdrant_client = self.init_qdrant()
        df = self.ingest_data

        # Проверяем, существует ли коллекция
        collection_exists = qdrant_client.collection_exists(self.collection_name)

        # Получаем документы из DataFrame с обработкой разных форматов
        documents = []
        file_paths = []
        texts = []

        # Определяем имена колонок в DataFrame
        available_columns = df.columns.tolist()

        # Определяем, какая колонка содержит текст
        text_column = None
        for possible_name in ['text', 'content', 'page_content', 'document']:
            if possible_name in available_columns:
                text_column = possible_name
                break

        # Определяем, какая колонка содержит пути к файлам
        file_column = None
        for possible_name in ['file_path', 'path', 'source', 'filename', 'file']:
            if possible_name in available_columns:
                file_column = possible_name
                break

        if not text_column:
            return Data(text="DataFrame must contain a column with text",
                        data={"error": "Missing text column",
                              "available_columns": available_columns})

        for index, row in df.iterrows():
            # Получаем текст
            text = row[text_column]
            if pd.isna(text) or not text:
                continue

            # Получаем путь к файлу (если есть)
            if file_column and pd.notna(row.get(file_column)):
                file_path = str(row[file_column])
                source = os.path.basename(file_path)
            else:
                # Генерируем source на основе session_id или индекса
                if 'session_id' in available_columns and pd.notna(row.get('session_id')):
                    source = f"session_{row['session_id']}"
                    file_path = f"session_{row['session_id']}_{index}"
                elif 'sender' in available_columns and pd.notna(row.get('sender')):
                    source = f"sender_{row['sender']}"
                    file_path = f"sender_{row['sender']}_{index}"
                else:
                    source = f"doc_{index}"
                    file_path = f"doc_{index}"

            text = str(text)

            # Получаем время модификации файла
            modification_time = self.get_file_modification_time(file_path)

            # Генерируем хэш содержимого для проверки изменений
            content_hash = self.generate_content_hash(text)

            documents.append({
                'file_path': file_path,
                'text': text,
                'source': source,
                'modification_time': modification_time,
                'chunk_index': index,
                'content_hash': content_hash,
                'is_real_file': os.path.exists(file_path)
            })

            file_paths.append(file_path)
            texts.append(text)

        if not documents:
            return Data(text="No documents to process", data={"processed": 0})

        # Если коллекция существует, проверяем какие файлы нужно обновить
        files_to_process = []

        if collection_exists:
            # Получаем информацию о существующих файлах из Qdrant
            existing_files = self.get_existing_file_times(qdrant_client, self.collection_name)
            existing_hashes = self.get_existing_content_hashes(qdrant_client, self.collection_name)

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
                    # Новый файл
                    files_to_process.extend(file_docs)
                else:
                    # Файл уже существует
                    current_doc = file_docs[0]  # Берем первый чанк для проверки

                    # Проверяем по времени модификации (только для реальных файлов)
                    if current_doc['is_real_file']:
                        current_time = current_doc['modification_time']
                        stored_time = existing_files[filename]

                        if current_time != stored_time:
                            # Удаляем старые точки этого файла
                            self.delete_points_by_source(qdrant_client, self.collection_name, filename)
                            files_to_process.extend(file_docs)
                    else:
                        # Для сгенерированных файлов проверяем по хэшу содержимого
                        current_hash = current_doc['content_hash']
                        stored_hash = existing_hashes.get(filename)

                        if stored_hash != current_hash:
                            # Удаляем старые точки этого файла
                            self.delete_points_by_source(qdrant_client, self.collection_name, filename)
                            files_to_process.extend(file_docs)
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
            return Data(text=f"Error creating embedding: {str(e)}", data={"error": str(e)})

        # Создаем коллекцию (если не существует)
        collection_created = self.create_collection_if_not_exists(qdrant_client, self.collection_name, vector_size)

        points = []
        processed_files_info = []

        # Генерируем эмбеддинги для документов, требующих обработки
        for i, doc in enumerate(files_to_process):
            try:
                embedding = self.embedding.embed_query(doc['text'])

                # Формируем payload с дополнительными полями
                payload = {
                    "text": doc['text'],
                    "source": doc['source'],
                    "file_path": doc['file_path'],
                    "chunk_index": doc['chunk_index'],
                    "modification_time": doc['modification_time'],
                    "modification_time_readable": datetime.fromtimestamp(doc['modification_time']).strftime(
                        '%Y-%m-%d %H:%M:%S') if doc['modification_time'] > 0 else "Unknown",
                    "content_hash": doc['content_hash']
                }

                # Добавляем дополнительные поля, если они есть в исходных данных
                if 'session_id' in available_columns:
                    payload["session_id"] = str(row.get('session_id', ''))
                if 'sender' in available_columns:
                    payload["sender"] = str(row.get('sender', ''))
                if 'sender_name' in available_columns:
                    payload["sender_name"] = str(row.get('sender_name', ''))

                points.append(PointStruct(
                    id=self.generate_point_id(doc['text'], doc['source'], doc['chunk_index']),
                    vector=embedding,
                    payload=payload
                ))
                processed_files_info.append(doc)
            except Exception as e:
                continue

        # Загружаем точки в Qdrant используя upsert
        if points:
            try:
                qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

                # Включаем индексацию для поиска
                qdrant_client.update_collection(
                    collection_name=self.collection_name,
                    hnsw_config=models.HnswConfigDiff(
                        m=self.hnsw_m,
                        ef_construct=self.hnsw_ef_construct,
                        full_scan_threshold=self.full_scan_threshold
                    )
                )
            except Exception as e:
                return Data(text=f"Error upserting points: {str(e)}", data={"error": str(e)})

        # Используем информацию из обработанных файлов для возвращаемых данных
        modification_times = [doc['modification_time'] for doc in processed_files_info]
        modification_times_readable = [
            datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S') if time > 0 else "Unknown"
            for time in modification_times
        ]

        # Получаем пути и тексты только из обработанных документов
        processed_file_paths = [doc['file_path'] for doc in processed_files_info]
        processed_texts = [doc['text'] for doc in processed_files_info]

        hnsw_config = {
            "m": self.hnsw_m,
            "ef_construct": self.hnsw_ef_construct,
            "full_scan_threshold": self.full_scan_threshold
        }

        return Data(
            text=f"Processed {len(points)} chunks to Qdrant collection '{self.collection_name}'",
            data={
                "file_paths": processed_file_paths,
                "texts": processed_texts,
                "modification_times": modification_times,
                "modification_times_readable": modification_times_readable,
                "processed_chunks": len(points),
                "hnsw_config": hnsw_config,
                "collection_name": self.collection_name,
                "total_documents": len(documents),
                "updated_documents": len(files_to_process),
                "real_files_processed": sum(1 for doc in processed_files_info if doc['is_real_file'])
            }
        )