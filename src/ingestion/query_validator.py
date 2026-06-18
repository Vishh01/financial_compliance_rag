import logging
import httpx
import json
import re
from config.settings import settings

logger = logging.getLogger(__name__)

class QueryDeconstructor:
    def __init__(self):
        self.model_name = settings.ROUTER_MODEL  # Assumed to be qwen2.5:0.5b
        # Use /api/chat for robust, structured system/user parameter handshakes
        self.ollama_url = f"{settings.OLLAMA_BASE_URL}/api/chat"

    async def validate_and_deconstruct(self, user_query: str) -> list:
        """
        Executes high-speed security checks, non-blocking asynchronous spelling 
        correction, query expansion, and sub-query deconstruction.
        """
        logger.info("Executing high-speed security and semantic expansion passes...")

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

        # Rule 1: Invoke LLM Engine for Spelling Correction, Expansion, and Deconstruction
        logger.info(f"Invoking async micro-model ({self.model_name}) for query optimization...")
        
        system_instruction = (
            "You are an expert financial search query optimizer.\n"
            "Your task is to take an incoming user query and return a valid JSON array of strings containing "
            "the optimized search phrases. You must execute these transformations:\n"
            "1. SPELLING CORRECTION: Detect and fix any misspelled technical terms or company corporate names "
            "(e.g., change 'nvida' or 'nvda' to 'NVIDIA', 'aple' or 'apl' to 'Apple').\n"
            "2. QUERY DECONSTRUCTION: If the query is complex or contains multiple targets connected by coordinates or conjunctions, "
            "break them into completely separate independent search queries.\n"
            "3. QUERY EXPANSION: Keep queries concise but expand shorthand context into meaningful search variations.\n"
            "Output strictly a JSON array of strings, nothing else."
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Optimize and split this query: {user_query}"}
            ],
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["queries"]
            },
            "options": {"temperature": 0.0}
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.ollama_url, json=payload, timeout=7.0)
                response.raise_for_status()
                
                # Unpack response from Ollama's standard Chat JSON structure
                response_data = response.json()
                message_content = response_data.get("message", {}).get("content", "").strip()
                
                # Parse the structured schema output object
                parsed_json = json.loads(message_content)
                optimized_queries = parsed_json.get("queries", [user_query])
                
                logger.info(f"Successfully optimized queries to: {optimized_queries}")
                return optimized_queries

            except Exception as e:
                logger.error(f"Validator optimization framework fallback: {str(e)}. Defaulting to raw pass-through.")
                
                # Manual deterministic fallback layer if the network or JSON parsing drops
                if " and " in user_query.lower():
                    parts = re.split(r',\s*and\s+|\s+and\s+', user_query, flags=re.IGNORECASE)
                    return [p.strip() for p in parts if p.strip()]
                
                return [user_query]