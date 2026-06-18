import logging
import shutil
from pathlib import Path
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def nuclear_disk_flush():
    logger.info("Initializing Atomic Disk-Level Cache Vaporization...")
    
    # Resolve the exact physical path where Qdrant writes collection segments
    qdrant_root = Path(str(settings.QDRANT_PATH))
    target_cache_folder = qdrant_root / "collections" / "semantic_query_cache"
    
    print(f"\nTargeting Path: {target_cache_folder}")
    
    if target_cache_folder.exists() and target_cache_folder.is_dir():
        try:
            logger.warning("Active Cache Storage block identified on disk. Executing hard wipe...")
            # rmtree bypasses client state memory mappings and strips the files off the file allocation table
            shutil.rmtree(target_cache_folder)
            logger.info("🟢 SUCCESS: Cache segments physically purged right off the disk structure.")
        except Exception as e:
            logger.error(f"🛑 CRITICAL LOCK: OS refused file un-link payload: {str(e)}")
            logger.info("REMEDY: Kill your Uvicorn process (Ctrl+C) to free RAM handles, then re-run this script.")
    else:
        logger.info("🔵 Storage layer clear: The physical 'semantic_query_cache' directory does not exist.")

if __name__ == "__main__":
    nuclear_disk_flush()
    





"""import logging
from qdrant_client import QdrantClient
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clear_semantic_cache():
    logger.info("Initializing vector connection to Qdrant storage engine...")
    
    try:
        # Reverted to direct instantiation to remain fully compliant with your local package version
        client = QdrantClient(path=str(settings.QDRANT_PATH))
        cache_collection = "semantic_query_cache"
        
        collections = client.get_collections().collections
        exists = any(c.name == cache_collection for c in collections)
        
        if exists:
            logger.warning(f"Purging active target cache collection: '{cache_collection}'...")
            client.delete_collection(collection_name=cache_collection)
            logger.info("Semantic vector memory records successfully dropped from cluster storage.")
        else:
            logger.info("Cache index verify: clear. No previous cache structures found to delete.")
                
    except Exception as e:
        logger.error(f"Critical execution error while flushing local query cache: {str(e)}")

if __name__ == "__main__":
    clear_semantic_cache()"""