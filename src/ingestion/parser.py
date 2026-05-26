import os
import logging
from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from config.settings import settings

# Set up clean system logging for production observability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logger = logging.getLogger(__name__)

class LayoutAwareParser:
    """
    Handles secure, local ingestion of high-complexity corporate financial disclosures,
    preserving structural metadata layers for advanced downstream multi-hop retrieval.
    """
    def __init__(self, data_directory: str = settings.DATA_DIR):
        self.data_directory = data_directory
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            logger.info(f"Created landing data directory at: {self.data_directory}")

    def load_local_documents(self) -> List[Document]:
        """
        Scans the local data directory and ingests PDF files, embedding strict metadata 
        stamps to isolate corporate entities and prevent cross-contamination.
        """
        logger.info(f"Initializing document scanning sequence inside: {self.data_directory}")
        
        # Ensure target data assets are present
        supported_files = [f for f in os.listdir(self.data_directory) if f.endswith('.pdf')]
        if not supported_files:
            logger.warning(f"No source material PDFs found in {self.data_directory}. Please add 10-K filings.")
            return []

        # Enforce metadata extraction rules to inject document origin tracks
        def file_metadata_helper(file_path: str) -> dict:
            file_name = os.path.basename(file_path)
            # Extrapolate entity identity from standard file naming conventions
            entity = "Unknown Corporation"
            if "apple" in file_name.lower():
                entity = "Apple Inc."
            elif "nvidia" in file_name.lower():
                entity = "NVIDIA Corporation"
                
            return {
                "file_name": file_name,
                "corporate_entity": entity,
                "document_type": "SEC Form 10-K Annual Report",
                "security_classification": "Proprietary / Local Compliance Restricted"
            }

        try:
            reader = SimpleDirectoryReader(
                input_dir=self.data_directory,
                required_exts=[".pdf"],
                file_metadata=file_metadata_helper
            )
            documents = reader.load_data()
            logger.info(f"Ingestion successful. Parsed {len(documents)} page-level text frames.")
            return documents
        except Exception as e:
            logger.error(f"Critical failure during ingestion sequence execution: {str(e)}")
            raise e

if __name__ == "__main__":
    # Local module diagnostic verification test
    print("\n--- Diagnostic Pipeline Verification Run ---")
    parser = LayoutAwareParser()
    parsed_docs = parser.load_local_documents()
    if parsed_docs:
        print(f"Successfully processed sample data. Target node count: {len(parsed_docs)}")
        print(f"Sample Metadata Mapping: {parsed_docs[0].metadata}\n")