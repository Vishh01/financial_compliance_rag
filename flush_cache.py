import logging
from qdrant_client import QdrantClient
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_semantic_cache():
    logger.info("Connecting to Qdrant storage path...")
    client = QdrantClient(path=str(settings.QDRANT_PATH))
    cache_collection = "semantic_query_cache"
    
    try:
        collections = client.get_collections().collections
        exists = any(c.name == cache_collection for c in collections)
        
        if exists:
            logger.warning(f"Purging existing cache collection: '{cache_collection}'...")
            client.delete_collection(collection_name=cache_collection)
            logger.info("Cache successfully cleared!")
        else:
            logger.info("No cache collection found to delete.")
            
    except Exception as e:
        logger.error(f"Error while flushing cache: {str(e)}")

if __name__ == "__main__":
    clear_semantic_cache()