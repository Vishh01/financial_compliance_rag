import logging
import requests
import json
from config.settings import settings

logger = logging.getLogger(__name__)

class QueryDeconstructor:
    """
    Pre-Query Validation Layer. Evaluates input complexity and 
    deconstructs compound, multi-tenant questions into targeted sub-queries.
    """
    def __init__(self):
        self.model_name = settings.LLM_MODEL
        self.ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    def validate_and_deconstruct(self, user_query: str) -> list:
        """
        Uses local LLM instruction to analyze a query and break it down 
        into an explicit Python list of focused search strings.
        """
        logger.info("Evaluating query complexity for structural deconstruction...")

        # Craft a highly strict system prompt forcing a raw JSON array output
        system_instructions = (
            "You are an advanced query parser for a database retrieval system.\n"
            "Your task is to look at a user's question and determine if it asks for multiple distinct pieces of information.\n"
            "If it is a compound question, break it down into separate, simple search strings targeting specific topics.\n"
            "If it is already a single simple question, return it as a single-element array.\n\n"
            "CRITICAL OUTPUT RULE:\n"
            "Return ONLY a raw, valid JSON list of strings. Do not include markdown formatting like ```json, no conversational filler, and no explanations.\n\n"
            "Examples:\n"
            'User: "Compare Apple\'s litigation risks and Nvidia\'s chip supply chains."\n'
            'Output: ["Apple litigation regulatory risks", "Nvidia chip manufacturing supply chain constraints"]\n'
            'User: "What are the audit guidelines for risk compliance?"\n'
            'Output: ["audit guidelines for risk compliance"]'
        )

        payload = {
            "model": self.model_name,
            "prompt": f"{system_instructions}\n\nUser Query: \"{user_query}\"\nOutput:",
            "stream": False,
            "options": {
                "temperature": 0.0  # Force maximum determinism
            }
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            raw_output = response.json().get("response", "").strip()
            
            # Defensive clean up in case the local model adds markdown fences anyway
            if raw_output.startswith("```"):
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()

            # Parse the clean JSON array back to a Python List
            sub_queries = json.loads(raw_output)
            
            if isinstance(sub_queries, list):
                logger.info(f"Successfully validated. Split query into {len(sub_queries)} atomic targets.")
                return sub_queries
            else:
                return [user_query]

        except Exception as e:
            logger.error(f"Failed to cleanly parse query deconstruction. Falling back to raw query. Error: {str(e)}")
            return [user_query]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("\n--- Testing Pre-Query Structural Deconstructor ---")
    
    parser = QueryDeconstructor()
    
    # Complex multi-document scenario query
    complex_test = "Show me Apple's legal compliance policies and check if Nvidia has manufacturing or risk delays."
    
    sub_tasks = parser.validate_and_deconstruct(complex_test)
    
    print(f"\nOriginal Input: '{complex_test}'")
    print("Deconstructed Search Elements:")
    for idx, sub_q in enumerate(sub_tasks):
        print(f" -> Sub-Query #{idx+1}: {sub_q}")
    print("\n------------------------------------------------")