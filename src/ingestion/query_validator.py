import logging
import requests
import json
from config.settings import settings

logger = logging.getLogger(__name__)

class QueryDeconstructor:
    def __init__(self):
        # Target the ultra-lightweight micro-model
        self.model_name = settings.ROUTER_MODEL  # llama3.2:1b
        self.ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    def validate_and_deconstruct(self, user_query: str) -> list:
        logger.info(f"Micro-Model ({self.model_name}) parsing query complexity...")

        # A highly optimized, direct prompt designed for sub-billion models
        prompt = (
            f"Split this compound question into a clean list of individual search topics. "
            f"Format your output strictly as a simple JSON array of strings. No markdown, no text explanations.\n"
            f"Query: {user_query}\n"
            f"Output:"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 64  # Cut off text generation early to save time
            }
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=10)
            raw_output = response.json().get("response", "").strip()
            
            if "```" in raw_output:
                raw_output = raw_output.split("```")[1].replace("json", "").strip()

            sub_queries = json.loads(raw_output)
            if isinstance(sub_queries, list):
                return sub_queries
            return [user_query]
        except Exception:
            # High-speed fallback: if JSON parsing fails, use the raw query instantly
            return [user_query]