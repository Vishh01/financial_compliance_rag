import logging
import uuid
import asyncio
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config.settings import settings
from src.ingestion.parser import LayoutAwareParser

logger = logging.getLogger(__name__)

class VectorGenerationEngine:
    def __init__(self):
        logger.info("Initializing Local Vector Generation Engine...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL_NAME,
            cache_folder="./model_cache"
        )
        # Using AsyncQdrantClient to prevent thread blocking during local disk I/O operations
        self.client = AsyncQdrantClient(path=str(settings.QDRANT_PATH))
        self.collection_name = settings.COLLECTION_NAME

    async def initialize_collection(self):
        """Wipes existing database state and provisions standard collection schema parameters."""
        if await self.client.collection_exists(collection_name=self.collection_name):
            logger.info("Clearing old cluster storage instance to ensure uniform schema layout...")
            await self.client.delete_collection(collection_name=self.collection_name)
            
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    async def execute_pipeline(self):
        # Establish structural context collections from target data repository
        parser = LayoutAwareParser()
        documents = parser.load_local_documents()
        
        if not documents:
            logger.warning("Pipeline aborted: No source elements available for structural index mapping.")
            return

        await self.initialize_collection()

        # Document block splitting configuration parameters
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        
        logger.info(f"Generated {len(nodes)} semantic chunks. Compiling high-dimensional embeddings...")
        
        points = []
        for i, node in enumerate(nodes):
            text_content = node.get_content()
            
            # Map native storage properties across metadata layout variants
            file_name = node.metadata.get("file_name") or node.extra_info.get("file_name", "Unknown File")
            page_label = node.metadata.get("page_label") or node.extra_info.get("page_label", "N/A")
            corporate_entity = node.metadata.get("corporate_entity", "Unknown Corporation")
            
            # Injecting coordinates explicitly into textual array to anchor vector space alignment
            formatted_text_block = (
                f"Source: {file_name} | Page: {page_label} | Entity: {corporate_entity}\n"
                f"Context Content:\n{text_content}"
            )
            
            # Compute vector footprint locally on local hardware infrastructure
            vector = self.embed_model.get_text_embedding(formatted_text_block)
            
            payload = {
                "text": text_content,
                "file_name": file_name,
                "page_label": page_label,
                "corporate_entity": corporate_entity,
                "document_type": node.metadata.get("document_type", "SEC Form 10-K Annual Report"),
                "chunk_index": i  # Sequence tracker tracking sequence positioning across the corpus split
            }
            
            # Document authorization clearance level classification filter logic
            filename_lower = file_name.lower()
            if any(k in filename_lower for k in ["internal", "secret", "confidential"]):
                security_tier = "internal"
            else:
                security_tier = "public"
                
            payload["metadata"] = {"classification": security_tier}
            
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
            
            if (i + 1) % 50 == 0 or (i + 1) == len(nodes):
                logger.info(f"Processed vector coordinates: {i + 1}/{len(nodes)}")

        logger.info(f"Committing points directly to local storage at {settings.QDRANT_PATH}...")
        
        # Async batch uploading to prevent thread pool starving
        await self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info("Vector database index successfully built with explicit metadata tracking fields!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    engine = VectorGenerationEngine()
    # Direct asynchronous execution wrapper for diagnostic routines
    asyncio.run(engine.execute_pipeline())