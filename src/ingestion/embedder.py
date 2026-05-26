import logging
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config.settings import settings
from src.ingestion.parser import LayoutAwareParser

logger = logging.getLogger(__name__)

class VectorGenerationEngine:
    """
    Orchestrates document chunking, localized high-dimensional vector extraction,
    and native transactional commits into the embedded Qdrant Rust engine storage.
    """
    def __init__(self):
        logger.info("Initializing Local Vector Generation Engine...")
        
        # Initialize the localized 6GB RAM optimized embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL_NAME,
            cache_folder="./model_cache"
        )
        
        # Initialize the native Embedded Qdrant client (No Docker Required)
        self.qdrant_client = QdrantClient(path=settings.QDRANT_PATH)
        
        # Bind Qdrant store layer to LlamaIndex abstractions
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=settings.COLLECTION_NAME
        )

    def execute_pipeline(self):
        """
        Executes end-to-end chunking, embedding generation, and vector index persistence.
        """
        # 1. Harvest raw documents using our successful parser module
        parser = LayoutAwareParser()
        documents = parser.load_local_documents()
        
        if not documents:
            logger.warning("Pipeline aborted: Zero document objects available for indexing.")
            return None

        logger.info("Executing Sentence-Level chunking sequence (Chunk Size: 512, Overlap: 50)...")
        
        # 2. Configure standard production overlapping text chunk splitter
        pipeline_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        
        # 3. Create a clean storage tracking context
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        logger.info("Generating high-dimensional embeddings and writing vectors to disk...")
        
        # 4. Generate vectors and commit to embedded storage
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[pipeline_splitter],
            embed_model=self.embed_model,
            show_progress=True
        )
        
        logger.info(f"Vector database index successfully built and committed locally to: {settings.QDRANT_PATH}")
        return index

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("\n--- Starting Vector Database Build Diagnostic Run ---")
    engine = VectorGenerationEngine()
    engine.execute_pipeline()
    print("--- Diagnostic Vector Run Complete ---\n")