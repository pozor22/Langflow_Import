import os
import ollama
import PyPDF2
import chardet
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue


OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "test_collection"
EMBEDDING_MODEL = "bge-m3:latest"
DOCUMENTS_DIR = "./exemple"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


qdrant_client = QdrantClient(url=QDRANT_URL)
ollama_client = ollama.Client(host=OLLAMA_URL)


def get_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Разбивает текст на фрагменты с нахлыстом (overlap) как в LangFlow"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)

        chunk = text[start:end]
        chunks.append(chunk)

        if end == len(text):
            break

        start += chunk_size - chunk_overlap

        if start >= len(text):
            break

    return chunks


def detect_encoding(file_path):
    """Определяет кодировку файла"""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def extract_text_from_pdf(file_path):
    """Извлекает текст из PDF файла"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Ошибка при чтении PDF файла {file_path}: {str(e)}")
    return text


def get_files_from_directory(directory_path):
    """Читает все файлы из указанной директории и возвращает с временем модификации"""
    documents = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        modification_time = os.path.getmtime(file_path)

        if filename.endswith('.pdf'):
            content = extract_text_from_pdf(file_path)
            if not content.strip():
                print(f"Не удалось извлечь текст из PDF файла {filename}")
                continue

        elif filename.endswith('.txt'):
            try:
                encoding = detect_encoding(file_path) or 'utf-8'
                with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                    content = file.read()
            except Exception as e:
                print(f"Ошибка при чтении файла {filename}: {str(e)}")
                continue
        else:
            print(f"Неподдерживаемый формат файла: {filename}")
            continue

        chunks = get_chunks(content, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            documents.append({
                'id': f"{filename}_chunk_{i}",
                'text': chunk.strip(),
                'source': filename,
                'modification_time': modification_time,
                'chunk_index': i,
                'total_chunks': len(chunks)
            })

    return documents


def get_existing_file_times(collection_name):
    """Получает время модификации файлов из существующих точек в Qdrant"""
    existing_files = {}

    scroll_result = qdrant_client.scroll(
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

    return existing_files


def create_collection_if_not_exists(client, collection_name, vector_size):
    """Создает коллекцию в Qdrant, если она не существует"""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(m=0)
        )
        print(f"Коллекция {collection_name} создана")
        return True
    else:
        print(f"Коллекция {collection_name} уже существует")
        return False


def delete_points_by_source(collection_name, filename):
    """Удаляет все точки, связанные с указанным файлом"""
    try:
        qdrant_client.delete(
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
        print(f"Удалены точки для файла: {filename}")
        return True
    except Exception as e:
        print(f"Ошибка при удалении точек для файла {filename}: {e}")
        return False


def main():
    collection_exists = qdrant_client.collection_exists(COLLECTION_NAME)

    documents = get_files_from_directory(DOCUMENTS_DIR)
    print(f"Найдено {len(documents)} фрагментов в директории {DOCUMENTS_DIR}")

    if not documents:
        print("Нет документов для обработки")
        return

    files_to_process = []
    if collection_exists:
        existing_files = get_existing_file_times(COLLECTION_NAME)
        print(f"В коллекции найдено {len(existing_files)} файлов")

        files_dict = {}
        for doc in documents:
            filename = doc['source']
            if filename not in files_dict:
                files_dict[filename] = []
            files_dict[filename].append(doc)

        for filename, file_docs in files_dict.items():
            if filename not in existing_files:
                print(f"Новый файл: {filename}")
                files_to_process.extend(file_docs)
            else:
                current_time = file_docs[0]['modification_time']
                stored_time = existing_files[filename]

                if current_time != stored_time:
                    print(f"Файл изменен: {filename} (было: {stored_time}, стало: {current_time})")
                    delete_points_by_source(COLLECTION_NAME, filename)
                    files_to_process.extend(file_docs)
                else:
                    print(f"Файл без изменений: {filename}")
    else:
        files_to_process = documents

    if not files_to_process:
        print("Нет новых или измененных файлов для обработки")
        return

    print(f"Будет обработано {len(files_to_process)} фрагментов")

    try:
        sample_embedding = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=files_to_process[0]['text'])
        vector_size = len(sample_embedding['embedding'])
        print(f"Размерность векторов: {vector_size}")
    except Exception as e:
        print(f"Ошибка при создании эмбеддинга: {e}")
        return

    create_collection_if_not_exists(qdrant_client, COLLECTION_NAME, vector_size)

    points = []
    for i, doc in enumerate(files_to_process):
        try:
            response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=doc['text'])
            embedding = response['embedding']

            points.append(PointStruct(
                id=hash(doc['id']) % (2 ** 63),
                vector=embedding,
                payload={
                    "text": doc['text'],
                    "source": doc['source'],
                    "chunk_index": doc['chunk_index'],
                    "total_chunks": doc['total_chunks'],
                    "modification_time": doc['modification_time']
                }
            ))
            if (i + 1) % 10 == 0:
                print(f"Обработан документ {i + 1}/{len(files_to_process)}")
        except Exception as e:
            print(f"Ошибка при обработке документа {doc['id']}: {e}")
            continue

    if points:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print(f"Успешно загружено {len(points)} векторов в коллекцию {COLLECTION_NAME}")

        qdrant_client.update_collection(
            collection_name=COLLECTION_NAME,
            hnsw_config=models.HnswConfigDiff(m=16)
        )
        print("Индексация HNSW активирована")
    else:
        print("Нет точек для загрузки в Qdrant")


if __name__ == "__main__":
    main()