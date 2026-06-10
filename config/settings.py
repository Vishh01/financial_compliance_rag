''' We are running Qdrant in Embedded Mode (saving data straight to a local folder rather than using Docker).
creating the configuration file that tells our system exactly how to handle this.'''

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API App Settings
    APP_NAME: str = "Financial Compliance RAG Engine"
    API_PORT: int = 8000
    
     # Local AI Stack Components (Optimized for 6GB RAM)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:1.5b"  # For heavy synthesis answering
    EMBED_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    ROUTER_MODEL: str = "qwen2.5:0.5b"  # For lightning-fast query splitting
    
    # Local Storage & Vector Database (Embedded Mode - No Docker Needed!)
    QDRANT_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qdrant_storage")
    COLLECTION_NAME: str = "financial_compliance_documents"
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    class Config:
        env_file = ".env"

settings = Settings()