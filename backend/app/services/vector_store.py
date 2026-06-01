from pathlib import Path
from uuid import UUID, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models


POINT_NAMESPACE = UUID("a8e6e80e-cd36-4b43-8452-fdddaac3c4c5")


class VectorStore:
    def __init__(self, path: Path, collection_name: str):
        path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int) -> None:
        if self.collection_exists():
            if self._vector_size() == vector_size:
                return
            self.client.delete_collection(collection_name=self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def delete_document(self, document_id: str) -> None:
        if not self.collection_exists():
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def upsert_chunks(self, chunks: list[dict]) -> None:
        points = []
        for chunk in chunks:
            point_id = str(uuid5(POINT_NAMESPACE, chunk["chunk_id"]))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk["vector"],
                    payload=chunk["payload"],
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: list[float],
        limit: int,
        document_ids: list[str] | None = None,
    ):
        query_filter = None
        if document_ids:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=document_ids),
                    )
                ]
            )
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(item.name == self.collection_name for item in collections)

    def _vector_size(self) -> int | None:
        info = self.client.get_collection(collection_name=self.collection_name)
        vectors_config = info.config.params.vectors
        size = getattr(vectors_config, "size", None)
        if size is not None:
            return int(size)
        if isinstance(vectors_config, dict) and vectors_config:
            first = next(iter(vectors_config.values()))
            first_size = getattr(first, "size", None)
            if first_size is not None:
                return int(first_size)
            if isinstance(first, dict) and first.get("size") is not None:
                return int(first["size"])
        return None
