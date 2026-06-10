import logging
import httpx
import json
import re
from config.settings import settings

logger = logging.getLogger(__name__)

class QueryDeconstructor:
    def __init__(self):
        self.model_name = settings.ROUTER_MODEL
        self.ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    async def validate_and_deconstruct(self, user_query: str) -> list:
        """
        Executes high-speed security checks and non-blocking asynchronous 
        deconstruction passes on ingress user traffic.
        """
        logger.info("Executing high-speed security and deterministic deconstruction passes...")

        # Rule 0: 🛡️ Deterministic Jailbreak & Prompt Injection Defense Core
        jailbreak_pattern = re.compile(
            r"(forget\s+(about\s+)?(your\s+)?previous|ignore\s+(the\s+)?(previous\s+)?instruction|"
            r"system\s+override|developer\s+mode|act\s+as\s+a|you\s+are\s+now\s+unbound|"
            r"internal\s+insights|confidential|secret|private\s+data|ignore\s+constraints)", 
            re.IGNORECASE
        )
        
        if jailbreak_pattern.search(user_query):
            logger.warning(f"SECURITY ALERT: Jailbreak attempt blocked on ingress query: '{user_query}'")
            return ["SECURITY_VIOLATION_TRIGGERED"]

        # Rule 1: High-speed structural split for explicitly conjoined compound items (0ms overhead)
        if " and " in user_query.lower():
            parts = re.split(r',\s*and\s+|\s+and\s+', user_query, flags=re.IGNORECASE)
            sub_queries = [p.strip() for p in parts if p.strip()]
            logger.info(f"Deterministic parser split targets in 0ms: {sub_queries}")
            return sub_queries

        # Rule 2: Non-blocking Fallback to Async Micro-Model
        logger.info(f"No explicit separators found. Invoking async micro-model ({self.model_name})...")
        prompt = (
            f"Split this compound question into a valid JSON array of individual strings. "
            f"Query: {user_query}\nOutput:"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": {"type": "array", "items": {"type": "string"}},
            "options": {"temperature": 0.0}
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.ollama_url, json=payload, timeout=5.0)
                response.raise_for_status()
                return json.loads(response.json().get("response", "").strip())
            except Exception as e:
                logger.error(f"Validator fallback routing failed: {str(e)}. Defaulting to raw pass-through.")
                return [user_query]