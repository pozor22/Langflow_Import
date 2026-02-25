# from langflow.field_typing import Data
import os
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict
from langflow.custom.custom_component.component import Component
from langflow.io import (
    Output,
    StrInput,
    HandleInput,
    DataFrameInput,
    DropdownInput,
    IntInput,
    MessageTextInput,
    SecretStrInput
)
from langflow.schema.data import Data
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue


class QdrantIngestComponent(Component):
    display_name = "Qdrant Ingest"
    description = "Indexes documents into the Qdrant vector database"
    icon = "Qdrant"
    name = "QdrantIngest"
    dense_vector_name = "text-dense"
    sparse_vector_name = "text-sparse"
    bm25_model_name = "Qdrant/bm25"


    distance_map = {
        "Cosine": models.Distance.COSINE,
        "Euclidean": models.Distance.EUCLID,
        "Dot Product": models.Distance.DOT,
    }
    distance_keys = list(distance_map.keys())

    inputs = [
        StrInput(
            name="collection_name",
            display_name="Collection Name",
            required=True,
            advanced=False,
            info="Название новой/существующей коллекции."
        ),
        StrInput(
            name="url",
            display_name="Qdrant URL",
            required=True,
            advanced=True,
            info="URL Qdrant"
        ),
        SecretStrInput(
            name="api_key",
            display_name="Qdrant API Key",
            required=True,
            advanced=True,
            info="Qdrant API Key"
        ),
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            required=True,
            info="Эмбеддинг модель"
        ),
        DataFrameInput(
            name="ingest_data",
            display_name="Ingest Data",
            required=True,
            info="Таблица с данными для индексации."
        ),
        DropdownInput(
            name="distance_func",
            display_name="Distance Function",
            options=distance_keys,
            value=distance_keys[0],
            required=True,
            advanced=True,
            info="Метрика близости для новой коллекции. Должна совпадать с метрикой для поиска.",
        ),
        MessageTextInput(
            name="hnsw_m",
            display_name="HNSW M",
            value="16",
            advanced=False,
            required=True,
            info="Количество связей на узел в HNSW-графе. Больше = выше точность, но больше память и дольше индексация.",
        ),
        MessageTextInput(
            name="hnsw_ef_construct",
            display_name="HNSW EF Construct",
            value="100",
            advanced=False,
            required=True,
            info="Сколько кандидатов проверяется при построении индекса. Больше = лучше качество поиска, но медленнее индексация.",
        ),
        MessageTextInput(
            name="full_scan_threshold",
            display_name="Full Scan Threshold",
            value="10000",
            advanced=False,
            required=True,
            info="Порог, ниже которого Qdrant может использовать полный перебор вместо HNSW.",
        ),
        MessageTextInput(
            name="bm25_k",
            display_name="BM25 k",
            value="1.2",
            advanced=False,
            required=True,
            info="Параметр BM25 k для полнотекстового сигнала.",
        ),
        MessageTextInput(
            name="bm25_b",
            display_name="BM25 b",
            value="0.75",
            advanced=False,
            required=True,
            info="Параметр BM25 b для нормализации длины текста.",
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    @staticmethod
    def as_int_qdrant_params(value: str, field: str) -> int:
        try:
            int_value = int(value)

            return int_value
        except (TypeError, ValueError) as e:
            raise ValueError(f"{field} must be integer") from e

    @staticmethod
    def as_float_qdrant_params(value: str, field: str) -> float:
        try:
            float_value = float(value)
            return float_value
        except (TypeError, ValueError) as e:
            raise ValueError(f"{field} must be float") from e

    @staticmethod
    def _get_payload_field(payload: dict, key: str):
        if key in payload:
            return payload.get(key)
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            return metadata.get(key)
        return None

    @staticmethod
    def get_existing_file_times(client: QdrantClient, collection_name: str) -> Dict[str, float]:
        existing_files = {}

        try:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )

            for point in scroll_result[0]:
                payload = point.payload or {}
                filename = QdrantIngestComponent._get_payload_field(payload, "source")
                existing_time = QdrantIngestComponent._get_payload_field(payload, "modification_time")
                if filename is not None and existing_time is not None:
                    existing_files[str(filename)] = existing_time

        except Exception as e:
            pass

        return existing_files

    def create_collection_if_not_exists(self, client: QdrantClient, collection_name: str, vector_size: int) -> tuple[bool, str | None]:
        if client.collection_exists(collection_name):
            return False, None

        hnsw_config = models.HnswConfigDiff(
            m=self.as_int_qdrant_params(value=self.hnsw_m, field="HNSW M"),
            ef_construct=self.as_int_qdrant_params(value=self.hnsw_ef_construct, field="HNSW EF Construct"),
            full_scan_threshold=self.as_int_qdrant_params(value=self.full_scan_threshold, field="Full Scan Threshold")
        )

        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.dense_vector_name: models.VectorParams(
                        size=vector_size,
                        distance=self.distance_map.get(self.distance_func, models.Distance.COSINE)
                    )
                },
                sparse_vectors_config={
                    self.sparse_vector_name: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
                hnsw_config=hnsw_config
            )
            return True, None
        except Exception as hybrid_error:
            self.log(f"Hybrid collection creation failed for '{collection_name}': {hybrid_error}")

        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=self.distance_map.get(self.distance_func, models.Distance.COSINE)
                ),
                hnsw_config=hnsw_config
            )
            return True, "Hybrid sparse config is not available on this Qdrant setup. Created dense-only collection."
        except Exception as dense_error:
            return False, f"Failed to create collection: {dense_error}"

    def get_collection_vector_layout(self, client: QdrantClient, collection_name: str) -> dict:
        layout = {
            "dense_named": False,
            "dense_name": None,
            "sparse_named": False,
        }
        try:
            collection_info = client.get_collection(collection_name)
            vectors_config = collection_info.config.params.vectors

            if isinstance(vectors_config, dict):
                layout["dense_named"] = True
                if self.dense_vector_name in vectors_config:
                    layout["dense_name"] = self.dense_vector_name
                else:
                    layout["dense_name"] = next(iter(vectors_config.keys()), None)

            sparse_vectors_config = getattr(collection_info.config.params, "sparse_vectors", None)
            if isinstance(sparse_vectors_config, dict):
                layout["sparse_named"] = self.sparse_vector_name in sparse_vectors_config
        except Exception:
            pass
        return layout

    def get_collection_runtime_info(self, client: QdrantClient, collection_name: str) -> dict:
        runtime_info = {
            "distance_func": None,
            "hnsw_config": {
                "m": None,
                "ef_construct": None,
                "full_scan_threshold": None,
            },
            "sparse_enabled": False,
            "bm25_k": None,
            "bm25_b": None,
        }
        try:
            collection_info = client.get_collection(collection_name)
            vectors_config = collection_info.config.params.vectors

            vector_params = None
            if isinstance(vectors_config, dict):
                if self.dense_vector_name in vectors_config:
                    vector_params = vectors_config[self.dense_vector_name]
                else:
                    vector_params = next(iter(vectors_config.values()), None)
            else:
                vector_params = vectors_config

            actual_distance = getattr(vector_params, "distance", None) if vector_params else None
            for label, distance_value in self.distance_map.items():
                if distance_value == actual_distance:
                    runtime_info["distance_func"] = label
                    break

            hnsw_conf = getattr(collection_info.config, "hnsw_config", None)
            if hnsw_conf is not None:
                runtime_info["hnsw_config"] = {
                    "m": getattr(hnsw_conf, "m", None),
                    "ef_construct": getattr(hnsw_conf, "ef_construct", None),
                    "full_scan_threshold": getattr(hnsw_conf, "full_scan_threshold", None),
                }

            sparse_vectors_config = getattr(collection_info.config.params, "sparse_vectors", None)
            runtime_info["sparse_enabled"] = isinstance(sparse_vectors_config, dict) and bool(sparse_vectors_config)

            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=128,
                with_payload=True,
                with_vectors=False,
            )
            bm25_k_candidates = []
            bm25_b_candidates = []
            for point in scroll_result[0]:
                payload = point.payload or {}
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                k_val = metadata.get("bm25_k")
                b_val = metadata.get("bm25_b")
                if k_val is not None:
                    with_value = str(k_val)
                    bm25_k_candidates.append(with_value)
                if b_val is not None:
                    with_value = str(b_val)
                    bm25_b_candidates.append(with_value)

            if bm25_k_candidates:
                runtime_info["bm25_k"] = bm25_k_candidates[0]
            if bm25_b_candidates:
                runtime_info["bm25_b"] = bm25_b_candidates[0]
        except Exception:
            pass
        return runtime_info

    @staticmethod
    def delete_points_by_source(client: QdrantClient, collection_name: str, filename: str) -> bool:
        try:
            client.delete(
                collection_name=collection_name,
                points_selector=Filter(
                    should=[
                        FieldCondition(
                            key="metadata.source",
                            match=MatchValue(value=filename)
                        ),
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

    @staticmethod
    def generate_point_id(text: str, source: str, chunk_index: int) -> int:
        unique_string = f"{source}_{chunk_index}_{text}"
        hash_object = hashlib.md5(unique_string.encode())
        return int(hash_object.hexdigest()[:16], 16) % (2 ** 63)

    @staticmethod
    def get_existing_content_hashes(client: QdrantClient, collection_name: str) -> Dict[str, set[str]]:
        existing_hashes: Dict[str, set[str]] = {}

        try:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )

            for point in scroll_result[0]:
                payload = point.payload or {}
                filename = QdrantIngestComponent._get_payload_field(payload, "source")
                existing_hash = QdrantIngestComponent._get_payload_field(payload, "content_hash")
                if filename is not None and existing_hash is not None:
                    filename_key = str(filename)
                    if filename_key not in existing_hashes:
                        existing_hashes[filename_key] = set()
                    existing_hashes[filename_key].add(str(existing_hash))

        except Exception as e:
            pass

        return existing_hashes

    def build_output(self) -> Data:
        self.as_int_qdrant_params(value=self.hnsw_m, field="HNSW M")
        self.as_int_qdrant_params(value=self.hnsw_ef_construct, field="HNSW EF Construct")
        self.as_int_qdrant_params(value=self.full_scan_threshold, field="Full Scan Threshold")
        bm25_k = self.as_float_qdrant_params(value=self.bm25_k, field="BM25 k")
        bm25_b = self.as_float_qdrant_params(value=self.bm25_b, field="BM25 b")
        df = self.ingest_data

        qdrant_client = QdrantClient(
            url=self.url,
            api_key=self.api_key
        )

        collection_exists = qdrant_client.collection_exists(self.collection_name)

        documents = []
        file_paths = []
        texts = []

        available_columns = df.columns.tolist()

        text_column = None
        for possible_name in ['text', 'content', 'page_content', 'document']:
            if possible_name in available_columns:
                text_column = possible_name
                break

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
            text = row[text_column]
            if pd.isna(text) or not text:
                continue

            if file_column and pd.notna(row.get(file_column)):
                file_path = str(row[file_column])
                source = os.path.basename(file_path)
            else:
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

            try:
                if os.path.exists(file_path):
                    modification_time = os.path.getmtime(file_path)
                else:
                    modification_time = 0.0
            except Exception as e:
                modification_time = 0.0

            content_hash = hashlib.md5(text.encode()).hexdigest()

            documents.append({
                'file_path': file_path,
                'text': text,
                'source': source,
                'modification_time': modification_time,
                'chunk_index': index,
                'content_hash': content_hash,
                'is_real_file': os.path.exists(file_path),
                'session_id': str(row.get('session_id')) if 'session_id' in available_columns and pd.notna(row.get('session_id')) else None,
                'sender': str(row.get('sender')) if 'sender' in available_columns and pd.notna(row.get('sender')) else None,
                'sender_name': str(row.get('sender_name')) if 'sender_name' in available_columns and pd.notna(row.get('sender_name')) else None,
            })

            file_paths.append(file_path)
            texts.append(text)

        if not documents:
            return Data(text="No documents to process", data={"processed": 0})

        files_to_process = []

        if collection_exists:
            existing_files = self.get_existing_file_times(qdrant_client, self.collection_name)
            existing_hashes = self.get_existing_content_hashes(qdrant_client, self.collection_name)

            files_dict = {}
            for doc in documents:
                filename = doc['source']
                if filename not in files_dict:
                    files_dict[filename] = []
                files_dict[filename].append(doc)

            for filename, file_docs in files_dict.items():
                if filename not in existing_files:
                    files_to_process.extend(file_docs)
                else:
                    current_doc = file_docs[0]
                    current_hashes = {item["content_hash"] for item in file_docs}
                    stored_hashes = existing_hashes.get(filename, set())

                    if current_doc['is_real_file']:
                        current_time = current_doc['modification_time']
                        stored_time = existing_files[filename]
                        try:
                            stored_time = float(stored_time)
                        except (TypeError, ValueError):
                            pass

                        if (current_time != stored_time) or (stored_hashes != current_hashes):
                            self.delete_points_by_source(qdrant_client, self.collection_name, filename)
                            files_to_process.extend(file_docs)
                    else:
                        if stored_hashes != current_hashes:
                            self.delete_points_by_source(qdrant_client, self.collection_name, filename)
                            files_to_process.extend(file_docs)
        else:
            files_to_process = documents

        if not files_to_process:
            collection_runtime = self.get_collection_runtime_info(qdrant_client, self.collection_name)
            return Data(
                text=f"No new or changed files for collection '{self.collection_name}'",
                data={
                    "processed": 0,
                    "collection_name": self.collection_name,
                    "collection_exists": collection_exists,
                    "distance_func": collection_runtime.get("distance_func"),
                    "hnsw_config": collection_runtime.get("hnsw_config"),
                    "bm25_k": collection_runtime.get("bm25_k"),
                    "bm25_b": collection_runtime.get("bm25_b"),
                    "sparse_enabled": collection_runtime.get("sparse_enabled"),
                },
            )

        try:
            sample_embedding = self.embedding.embed_query(files_to_process[0]['text'])
            vector_size = len(sample_embedding)
        except Exception as e:
            return Data(text=f"Error creating embedding: {str(e)}", data={"error": str(e)})

        collection_created, creation_note = self.create_collection_if_not_exists(
            qdrant_client, self.collection_name, vector_size
        )
        if not qdrant_client.collection_exists(self.collection_name):
            return Data(
                text=f"Error creating collection '{self.collection_name}'",
                data={
                    "error": "Collection was not created",
                    "collection_name": self.collection_name,
                    "collection_creation_error": creation_note,
                },
            )
        vector_layout = self.get_collection_vector_layout(qdrant_client, self.collection_name)

        points = []
        processed_files_info = []

        for i, doc in enumerate(files_to_process):
            try:
                embedding = self.embedding.embed_query(doc['text'])

                metadata_payload = {
                    "source": doc['source'],
                    "file_path": doc['file_path'],
                    "chunk_index": doc['chunk_index'],
                    "modification_time": doc['modification_time'],
                    "modification_time_readable": datetime.fromtimestamp(doc['modification_time']).strftime(
                        '%Y-%m-%d %H:%M:%S') if doc['modification_time'] > 0 else "Unknown",
                    "content_hash": doc['content_hash'],
                    "bm25_k": bm25_k,
                    "bm25_b": bm25_b,
                }

                if doc.get("session_id") is not None:
                    metadata_payload["session_id"] = doc["session_id"]
                if doc.get("sender") is not None:
                    metadata_payload["sender"] = doc["sender"]
                if doc.get("sender_name") is not None:
                    metadata_payload["sender_name"] = doc["sender_name"]

                payload = {
                    "text": doc['text'],
                    "metadata": metadata_payload,
                }

                vector_payload = embedding
                if vector_layout["dense_named"] and vector_layout["dense_name"]:
                    vector_payload = {
                        vector_layout["dense_name"]: embedding,
                    }
                    if vector_layout["sparse_named"]:
                        vector_payload[self.sparse_vector_name] = models.Document(
                            text=doc["text"],
                            model=self.bm25_model_name,
                            options={"k": bm25_k, "b": bm25_b},
                        )

                points.append(PointStruct(
                    id=self.generate_point_id(doc['text'], doc['source'], doc['chunk_index']),
                    vector=vector_payload,
                    payload=payload
                ))
                processed_files_info.append(doc)
            except Exception as e:
                continue

        if points:
            try:
                qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            except Exception as e:
                return Data(text=f"Error upserting points: {str(e)}", data={"error": str(e)})

        modification_times = [doc['modification_time'] for doc in processed_files_info]
        modification_times_readable = [
            datetime.fromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S') if time > 0 else "Unknown"
            for time in modification_times
        ]

        processed_file_paths = [doc['file_path'] for doc in processed_files_info]
        processed_texts = [doc['text'] for doc in processed_files_info]

        collection_runtime = self.get_collection_runtime_info(qdrant_client, self.collection_name)
        hnsw_config = collection_runtime.get("hnsw_config", {})
        runtime_distance = collection_runtime.get("distance_func")
        runtime_bm25_k = collection_runtime.get("bm25_k")
        runtime_bm25_b = collection_runtime.get("bm25_b")
        if runtime_bm25_k is None and collection_created:
            runtime_bm25_k = bm25_k
        if runtime_bm25_b is None and collection_created:
            runtime_bm25_b = bm25_b

        if collection_created:
            status_text = f"Created new collection '{self.collection_name}' and processed {len(points)} chunks"
            collection_action = "created"
        else:
            status_text = f"Loaded {len(points)} chunks into existing collection '{self.collection_name}'"
            collection_action = "updated_existing"

        if creation_note:
            status_text = f"{status_text}. Note: {creation_note}"

        return Data(
            text=status_text,
            data={
                "file_paths": processed_file_paths,
                "texts": processed_texts,
                "modification_times": modification_times,
                "modification_times_readable": modification_times_readable,
                "processed_chunks": len(points),
                "hnsw_config": hnsw_config,
                "distance_func": runtime_distance,
                "bm25_k": runtime_bm25_k,
                "bm25_b": runtime_bm25_b,
                "sparse_enabled": collection_runtime.get("sparse_enabled"),
                "collection_name": self.collection_name,
                "collection_action": collection_action,
                "collection_created": collection_created,
                "collection_creation_note": creation_note,
                "total_documents": len(documents),
                "updated_documents": len(files_to_process),
                "real_files_processed": sum(1 for doc in processed_files_info if doc['is_real_file'])
            }
        )