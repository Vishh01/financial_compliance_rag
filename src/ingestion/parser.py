import logging
import re
from pathlib import Path
from llama_index.core import SimpleDirectoryReader

logger = logging.getLogger(__name__)

class LayoutAwareParser:
    def __init__(self, input_dir: str = "./data"):
        self.input_dir = Path(input_dir)
        logger.info(f"Targeting data repository at isolated path: {self.input_dir}")

    def _normalize_text_spacing(self, text: str) -> str:
        """
        Regex text healing layer. Automatically fixes character-smashing and running tokens
        caused by high-density tabular PDF extraction conversions.
        """
        if not text:
            return ""

        # 1. Insert spaces between numeric values and alphabetic metrics (e.g., "167,045million" -> "167,045 million")
        text = re.sub(r'(\d+)\s*([a-zA-Z])', r'\1 \2', text)
        
        # 2. Fix inverted smashing where alphabetic strings end immediately on numbers
        text = re.sub(r'([a-zA-Z])\s*(\d+)', r'\1 \2', text)

        # 3. Force clean spacing on punctuation boundaries that lack right-hand whitespace (e.g., "Americas,94,294" or "million.Thecompany")
        # Ignores decimals/fractions (like 12.5) and formatted values like $10,000 by checking trailing character layouts
        text = re.sub(r'(?<=[a-zA-Z])([,.:;])(?=[a-zA-Z0-9])', r'\1 ', text)

        # 4. Correct smashed words running together at lower-to-uppercase transitions (CamelCase split)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # 5. Collapse duplicate inline whitespace down to a uniform, clean sequence
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()

    def load_local_documents(self) -> list:
        """Reads local directory targets and runs the string healing processor over extracted text nodes."""
        if not self.input_dir.exists():
            logger.warning(f"Repository path target {self.input_dir} is missing. Provisioning empty container layout.")
            self.input_dir.mkdir(parents=True, exist_ok=True)
            return []

        logger.info(f"Scanning target node arrays inside database directory: {self.input_dir}")
        
        # Check for target files before starting the heavy file extraction pass
        supported_extensions = {".pdf", ".txt", ".docx", ".csv"}
        found_files = [f for f in self.input_dir.iterdir() if f.suffix.lower() in supported_extensions]
        
        if not found_files:
            logger.warning("No valid source documentation identified inside the data folder directory.")
            return []

        try:
            reader = SimpleDirectoryReader(input_dir=str(self.input_dir))
            documents = reader.load_data()
            
            logger.info(f"Successfully unpacked {len(documents)} raw document frames. Initiating parsing cleanup...")
            
            # Iterate through the documents and heal the text strings inline
            for doc in documents:
                raw_text = doc.text
                healed_text = self._normalize_text_spacing(raw_text)
                doc.text = healed_text
                
            logger.info("Text spacing normalization pipeline complete across all ingested frames.")
            return documents

        except Exception as e:
            logger.error(f"Critical execution error during file ingestion pass: {str(e)}")
            return []