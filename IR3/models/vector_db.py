from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

qdrant_client = QdrantClient(":memory:")

def setup_db(collection_name, dim=768):
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

def upload_to_qdrant(collection_name, vectors, payloads):
    points = [
        PointStruct(id=i, vector=vectors[i].tolist(), payload=payloads[i])
        for i in range(len(vectors))
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)