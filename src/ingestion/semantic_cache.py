import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config.settings import settings

logger = logging.getLogger(__name__)

class SemanticCache:
    def __init__(self, client: QdrantClient = None, threshold: float = 0.95):
        # Use the shared client instance if provided, avoiding file lock collisions
        self.client = client if client else QdrantClient(path=str(settings.QDRANT_PATH))
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL_NAME,
            cache_folder="./model_cache"
        )
        self.cache_collection = "semantic_query_cache"
        self.threshold = threshold
        self._ensure_cache_collection()

    def _ensure_cache_collection(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.cache_collection for c in collections)
        if not exists:
            logger.info(f"Initializing fresh Semantic Cache Collection: '{self.cache_collection}'")
            self.client.create_collection(
                collection_name=self.cache_collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def check_cache(self, query: str) -> str:
        query_vector = self.embed_model.get_text_embedding(query)
        results = self.client.search(
            collection_name=self.cache_collection,
            query_vector=query_vector,
            limit=1
        )
        if results and results[0].score >= self.threshold:
            logger.warning(f"SEMANTIC CACHE HIT! Match Confidence: {results[0].score:.4f}. Short-circuiting RAG pipeline.")
            return results[0].payload.get("cached_response")
        logger.info("Semantic cache miss. Route passed to ingestion pipeline.")
        return None

    def update_cache(self, query: str, response: str):
        logger.info("Caching newly generated response vector to persistent memory...")
        query_vector = self.embed_model.get_text_embedding(query)
        import uuid
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.cache_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=query_vector,
                    payload={"raw_query": query, "cached_response": response}
                )
            ]
        )