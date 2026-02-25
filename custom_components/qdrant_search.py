from copy import deepcopy
from typing import Any
from langchain_community.vectorstores import Qdrant
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models
from lfx.base.vectorstores.model import LCVectorStoreComponent, check_cached_vector_store
from lfx.schema.data import Data
from lfx.io import (
    DropdownInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    SecretStrInput,
    StrInput,
)


class QdrantSearchComponent(LCVectorStoreComponent):
    display_name = "Qdrant Search"
    description = "Search the Qdrant comparative collection."
    icon = "Qdrant"
    name = "QdrantSearch"
    dense_vector_name = "text-dense"
    sparse_vector_name = "text-sparse"
    bm25_model_name = "Qdrant/bm25"

    search_base_inputs = [
        component_input
        for component_input in deepcopy(LCVectorStoreComponent.inputs)
        if getattr(component_input, "name", None) != "ingest_data"
    ]
    for component_input in search_base_inputs:
        if getattr(component_input, "name", None) == "search_query":
            component_input.info = "Текст, по которому нужно искать в коллекции."

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
            info="Имя коллекции в Qdrant, где будет выполняться поиск.",
        ),
        StrInput(
            name="url",
            display_name="URL",
            required=True,
            info="URL Qdrant",
        ),
        SecretStrInput(
            name="api_key",
            display_name="Qdrant API Key",
            required=True,
            info="Qdrant API Key",
        ),
        DropdownInput(
            name="distance_func",
            display_name="Distance Function",
            options=distance_keys,
            value=distance_keys[0],
            required=True,
            advanced=True,
            info="Метрика близости. Должна соответствовать метрике коллекции.",
        ),
        MessageTextInput(
            name="semantic_weight",
            display_name="Semantic Search Weight",
            value="0.5",
            required=True,
            advanced=False,
            info="Вес семантической части в гибридном поиске.",
        ),
        MessageTextInput(
            name="keyword_weight",
            display_name="Keyword Search Weight",
            value="0.5",
            required=True,
            advanced=False,
            info="Вес полнотекстовой части (ключевые слова) в гибридном поиске.",
        ),
        StrInput(
            name="content_payload_key",
            display_name="Content Payload Key",
            value="text",
            advanced=True,
            info="Поле payload, где хранится текст документа. Обычно 'text' или 'page_content'.",
        ),
        StrInput(
            name="metadata_payload_key",
            display_name="Metadata Payload Key",
            value="metadata",
            advanced=True,
            info="Поле payload, где хранится метаинформация документа.",
        ),
        *search_base_inputs,
        HandleInput(
            name="embedding",
            display_name="Embedding",
            input_types=["Embeddings"],
            info="Эмбеддинг-модель, совместимая с той, которой индексировались данные.",
        ),
        IntInput(
            name="number_of_results",
            display_name="Number of Results",
            value=4,
            advanced=True,
            info="Сколько результатов вернуть из поиска.",
        ),
    ]

    @check_cached_vector_store
    def build_vector_store(self) -> Qdrant:
        qdrant_kwargs = {
            "collection_name": self.collection_name,
            "content_payload_key": self.content_payload_key,
            "metadata_payload_key": self.metadata_payload_key,
        }

        server_kwargs = {
            "url": self.url,
            "api_key": self.api_key,
        }
        server_kwargs = {k: v for k, v in server_kwargs.items() if v is not None}

        if not isinstance(self.embedding, Embeddings):
            msg = "Invalid embedding object"
            raise TypeError(msg)

        client = QdrantClient(**server_kwargs)

        expected_distance = self.distance_map.get(self.distance_func)
        if expected_distance and client.collection_exists(self.collection_name):
            try:
                collection_info = client.get_collection(self.collection_name)
                vectors_config = collection_info.config.params.vectors

                if isinstance(vectors_config, dict):
                    vector_params = next(iter(vectors_config.values()), None)
                    actual_distance = getattr(vector_params, "distance", None) if vector_params else None
                else:
                    actual_distance = getattr(vectors_config, "distance", None)

                if actual_distance and actual_distance != expected_distance:
                    self.log(
                        f"Distance Function mismatch: collection={actual_distance}, selected={expected_distance}."
                    )
            except Exception:
                pass

        return Qdrant(embeddings=self.embedding, client=client, **qdrant_kwargs)

    @staticmethod
    def as_float_search_params(value: str, field: str) -> float:
        try:
            float_value = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{field} must be float") from e

        if float_value < 0:
            raise ValueError(f"{field} must be >= 0")
        return float_value

    def validate_hybrid_weights(self) -> tuple[float, float]:
        semantic_weight = self.as_float_search_params(self.semantic_weight, "Semantic Search Weight")
        keyword_weight = self.as_float_search_params(self.keyword_weight, "Keyword Search Weight")

        if semantic_weight + keyword_weight > 1.0:
            raise ValueError("Semantic Search Weight + Keyword Search Weight must be <= 1.0")
        if semantic_weight == 0 and keyword_weight == 0:
            raise ValueError("At least one of Semantic Search Weight or Keyword Search Weight must be > 0")

        return semantic_weight, keyword_weight

    def get_collection_vector_layout(self, client: QdrantClient) -> dict[str, Any]:
        layout: dict[str, Any] = {
            "dense_name": None,
            "sparse_name": None,
        }
        collection_info = client.get_collection(self.collection_name)
        vectors_config = collection_info.config.params.vectors

        if isinstance(vectors_config, dict):
            if self.dense_vector_name in vectors_config:
                layout["dense_name"] = self.dense_vector_name
            else:
                layout["dense_name"] = next(iter(vectors_config.keys()), None)

        sparse_vectors_config = getattr(collection_info.config.params, "sparse_vectors", None)
        if isinstance(sparse_vectors_config, dict) and sparse_vectors_config:
            if self.sparse_vector_name in sparse_vectors_config:
                layout["sparse_name"] = self.sparse_vector_name
            else:
                layout["sparse_name"] = next(iter(sparse_vectors_config.keys()), None)
        return layout

    @staticmethod
    def extract_points(response: Any) -> list:
        points = getattr(response, "points", None)
        if points is None:
            return []
        return list(points)

    @staticmethod
    def normalize_scores(score_map: dict[str, float]) -> dict[str, float]:
        if not score_map:
            return {}
        values = list(score_map.values())
        minimum = min(values)
        maximum = max(values)
        if maximum - minimum == 0:
            return {point_id: 1.0 for point_id in score_map}
        return {point_id: (score - minimum) / (maximum - minimum) for point_id, score in score_map.items()}

    def search_documents(self) -> list[Data]:
        vector_store = self.build_vector_store()

        if self.search_query and isinstance(self.search_query, str) and self.search_query.strip():
            semantic_weight, keyword_weight = self.validate_hybrid_weights()
            client = vector_store.client
            if not client.collection_exists(self.collection_name):
                raise ValueError(f"Collection '{self.collection_name}' does not exist")

            layout = self.get_collection_vector_layout(client)
            dense_name = layout.get("dense_name")
            sparse_name = layout.get("sparse_name")

            fetch_limit = max(self.number_of_results * 3, self.number_of_results, 10)
            dense_scores: dict[str, float] = {}
            sparse_scores: dict[str, float] = {}
            points_by_id: dict[str, Any] = {}

            if semantic_weight > 0:
                dense_embedding = self.embedding.embed_query(self.search_query)
                dense_query_kwargs: dict[str, Any] = {
                    "collection_name": self.collection_name,
                    "query": dense_embedding,
                    "limit": fetch_limit,
                    "with_payload": True,
                    "with_vectors": False,
                }
                if dense_name:
                    dense_query_kwargs["using"] = dense_name

                dense_response = client.query_points(**dense_query_kwargs)
                for point in self.extract_points(dense_response):
                    point_id = str(point.id)
                    dense_scores[point_id] = float(point.score)
                    points_by_id[point_id] = point

            if keyword_weight > 0:
                if sparse_name is None:
                    if semantic_weight == 0:
                        raise ValueError("Keyword search weight > 0, but sparse vector is not configured in collection")
                    self.log("Sparse vector is not configured in collection. Keyword Search Weight will be ignored.")
                    keyword_weight = 0.0
                else:
                    sparse_query_doc = models.Document(
                        text=self.search_query,
                        model=self.bm25_model_name,
                    )
                    sparse_response = client.query_points(
                        collection_name=self.collection_name,
                        query=sparse_query_doc,
                        using=sparse_name,
                        limit=fetch_limit,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for point in self.extract_points(sparse_response):
                        point_id = str(point.id)
                        sparse_scores[point_id] = float(point.score)
                        if point_id not in points_by_id:
                            points_by_id[point_id] = point

            dense_normalized = self.normalize_scores(dense_scores)
            sparse_normalized = self.normalize_scores(sparse_scores)

            ranked_items: list[tuple[float, Data]] = []
            for point_id, point in points_by_id.items():
                hybrid_score = (
                    semantic_weight * dense_normalized.get(point_id, 0.0)
                    + keyword_weight * sparse_normalized.get(point_id, 0.0)
                )

                payload = point.payload or {}
                text_value = payload.get(self.content_payload_key)
                if text_value is None:
                    text_value = payload.get("text")
                if text_value is None:
                    text_value = payload.get("page_content")
                if text_value is None:
                    text_value = ""
                if not isinstance(text_value, str):
                    text_value = str(text_value)

                metadata_value = payload.get(self.metadata_payload_key)
                if not isinstance(metadata_value, dict):
                    metadata_value = {}

                ranked_items.append(
                    (
                        hybrid_score,
                        Data(
                            text=text_value,
                            data={
                                "id": point.id,
                                "score": hybrid_score,
                                "semantic_score_raw": dense_scores.get(point_id),
                                "keyword_score_raw": sparse_scores.get(point_id),
                                "semantic_score_normalized": dense_normalized.get(point_id, 0.0),
                                "keyword_score_normalized": sparse_normalized.get(point_id, 0.0),
                                "metadata": metadata_value,
                                "payload": payload,
                            },
                        ),
                    )
                )

            ranked_items.sort(key=lambda item: item[0], reverse=True)
            data = [item[1] for item in ranked_items[: self.number_of_results]]
            self.status = data
            return data

        return []