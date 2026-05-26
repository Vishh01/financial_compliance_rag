import logging
import uuid
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config.settings import settings
from src.ingestion.parser import LayoutAwareParser

logger = logging.getLogger(__name__)

class VectorGenerationEngine:
    """
    Orchestrates document chunking, localized embedding generation, and direct
    transactional commits into the embedded Qdrant engine without framework wrappers.
    """
    def __init__(self):
        logger.info("Initializing Local Vector Generation Engine...")
        
        # Initialize the localized embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL_NAME,
            cache_folder="./model_cache"
        )
        
        # Initialize the raw local database client directly
        self.client = QdrantClient(path=str(settings.QDRANT_PATH))
        
        # Ensure collection exists with correct vector dimensions (BGE-small is 384)
        if not self.client.collection_exists(collection_name=settings.COLLECTION_NAME):
            logger.info(f"Creating fresh collection cluster: {settings.COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def execute_pipeline(self):
        """
        Executes end-to-end parsing, semantic chunking, and batch vector database upserts.
        """
        parser = LayoutAwareParser()
        documents = parser.load_local_documents()
        
        if not documents:
            logger.warning("Pipeline aborted: No documents found to index.")
            return

        logger.info("Executing Sentence-Level chunking sequence (Size: 512, Overlap: 50)...")
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        
        logger.info(f"Generated {len(nodes)} semantic chunks. Compiling high-dimensional embeddings...")
        
        points = []
        for i, node in enumerate(nodes):
            # Extract raw text and compute vector coordinates
            text_content = node.get_content()
            vector = self.embed_model.get_text_embedding(text_content)
            
            # Preserve all metadata (file names, page numbers, layouts)
            payload = node.metadata.copy()
            payload["text"] = text_content
            
            # Construct a Qdrant native Point structure
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
            )
            
            if (i + 1) % 50 == 0 or (i + 1) == len(nodes):
                logger.info(f"Processed vector coordinates: {i + 1}/{len(nodes)}")

        # Upsert the vectors directly into the disk engine in chunks
        logger.info(f"Committing points directly to local storage at {settings.QDRANT_PATH}...")
        self.client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points
        )
        
        logger.info("Vector database index successfully built and committed locally!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("\n--- Starting Vector Database Build Diagnostic Run ---")
    engine = VectorGenerationEngine()
    engine.execute_pipeline()
    print("--- Diagnostic Vector Run Complete ---\n")