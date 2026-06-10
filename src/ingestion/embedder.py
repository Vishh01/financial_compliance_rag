import logging
import uuid
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
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
        self.client = QdrantClient(path=str(settings.QDRANT_PATH))
        
        if self.client.collection_exists(collection_name=settings.COLLECTION_NAME):
            logger.info("Clearing old cluster storage instance to ensure uniform schema layout...")
            self.client.delete_collection(collection_name=settings.COLLECTION_NAME)
            
        self.client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    def execute_pipeline(self):
        parser = LayoutAwareParser()
        documents = parser.load_local_documents()
        
        if not documents:
            logger.warning("Pipeline aborted: No documents found to index.")
            return

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        
        logger.info(f"Generated {len(nodes)} semantic chunks. Compiling high-dimensional embeddings...")
        
        points = []
        for i, node in enumerate(nodes):
            text_content = node.get_content()
            
            # --- ✅ FIX: SAFE METADATA EXTRACTOR LAYER ---
            # LlamaIndex stores parent/source document metadata keys under node.metadata
            # or extra_info. We map them directly to guarantee they hit the Qdrant payload.
            file_name = node.metadata.get("file_name") or node.extra_info.get("file_name", "Unknown File")
            page_label = node.metadata.get("page_label") or node.extra_info.get("page_label", "N/A")
            corporate_entity = node.metadata.get("corporate_entity", "Unknown Corporation")
            
            # Formulate a context-enriched string block for the vector space to optimize retrieval
            formatted_text_block = (
                f"Source: {file_name} | Page: {page_label} | Entity: {corporate_entity}\n"
                f"Context Content:\n{text_content}"
            )
            
            # Generate embedding on the enriched text block
            vector = self.embed_model.get_text_embedding(formatted_text_block)
            
            # Assemble payload ensuring top-level keys match retriever expectations exactly
            payload = {
                "text": text_content,
                "file_name": file_name,
                "page_label": page_label,
                "corporate_entity": corporate_entity,
                "document_type": node.metadata.get("document_type", "SEC Form 10-K Annual Report")
            }
            
            # --- GOVERNANCE INJECTION MIDDLEWARE ---
            filename_lower = file_name.lower()
            if any(k in filename_lower for k in ["internal", "secret", "confidential"]):
                security_tier = "internal"
            else:
                security_tier = "public"
                
            payload["metadata"] = {"classification": security_tier}
            # ---------------------------------------
            
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
            if (i + 1) % 50 == 0 or (i + 1) == len(nodes):
                logger.info(f"Processed vector coordinates: {i + 1}/{len(nodes)}")

        logger.info(f"Committing points directly to local storage at {settings.QDRANT_PATH}...")
        self.client.upsert(collection_name=settings.COLLECTION_NAME, points=points)
        logger.info("Vector database index successfully built with explicit metadata tracking fields!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    engine = VectorGenerationEngine()
    engine.execute_pipeline()