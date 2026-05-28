import logging
import requests
import json
import sys

from src.ingestion.retriever import ComplianceRetriever
# Import the new deconstructor
from src.ingestion.query_validator import QueryDeconstructor
from config.settings import settings

logger = logging.getLogger(__name__)

class ComplianceRAGPipeline:
    def __init__(self):
        self.retriever = ComplianceRetriever()
        self.deconstructor = QueryDeconstructor() # Initialize the gatekeeper
        self.model_name = settings.LLM_MODEL
        self.ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    def query(self, user_question: str, user_role: str = "public"):
        print(f"\n--- Processing Ingress Query [Role: {user_role.upper()}] ---")
        
        # 1. RUN PRE-QUERY VALIDATION AND DECONSTRUCTION
        # Breaks down compound questions into an iterable list
        sub_queries = self.deconstructor.validate_and_deconstruct(user_question)
        
        aggregated_context_chunks = []
        seen_payloads = set() # Prevent duplicate text blocks if queries overlap

        # 2. RUN SEARCH LOOP ACROSS ALL ATOMIC SUB-QUERIES
        for sub_q in sub_queries:
            logger.info(f"Executing secure routing pass for sub-target: '{sub_q}'")
            retrieval_response = self.retriever.retrieve_context(
                query=sub_q, 
                user_role=user_role, 
                limit=2 # Grab top 2 highly specific chunks per sub-topic
            )
            
            if retrieval_response["status"] == "blocked":
                print(f"\n[SYSTEM GUARDRAIL BLOCK]: {retrieval_response['message']}\n")
                return

            # Append unique chunks to our context window pool
            for chunk in retrieval_response["context"]:
                # Unique identifier based on document source and text content snippet
                chunk_id = f"{chunk['source']}_{chunk['page']}_{hash(chunk['text'])}"
                if chunk_id not in seen_payloads:
                    seen_payloads.add(chunk_id)
                    aggregated_context_chunks.append(chunk)

        # 3. Format aggregated unique contexts for presentation to the LLM
        formatted_context = ""
        for chunk in aggregated_context_chunks:
            formatted_context += f"--- Source Document: {chunk['source']} (Page {chunk['page']}) ---\n"
            formatted_context += f"Context Block:\n{chunk['text']}\n\n"

        # 4. Prompt Synthesis & Execution
        system_prompt = (
            "You are an elite corporate financial compliance auditor. Your task is to answer the user's question "
            "using ONLY the provided text blocks below. Adhere strictly to these execution constraints:\n"
            "1. Ground your answer entirely in the provided context. Do not extrapolate or assume facts.\n"
            "2. Cite the exact Source Document and Page numbers inline when presenting facts.\n"
            "3. Maintain an objective, professional, analytical corporate tone."
        )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"=== TARGET REFERENCE CONTEXT ===\n"
            f"{formatted_context}\n"
            f"=================================\n\n"
            f"USER QUESTION: {user_question}\n"
            f"SMART COMPLIANCE RESPONSE:"
        )

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": True
        }

        print(f"\n[LLM GENERATION STARTING VIA OLLAMA ({self.model_name})]:")
        try:
            response = requests.post(self.ollama_url, json=payload, stream=True)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk_json = json.loads(line.decode("utf-8"))
                    sys.stdout.write(chunk_json.get("response", ""))
                    sys.stdout.flush()
            print("\n\n--- End of Generation Pass ---\n")
        except requests.exceptions.ConnectionError:
            print(f"\nERROR: Unable to connect to local Ollama instance at {self.ollama_url}.\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    rag_system = ComplianceRAGPipeline()
    
    # A heavy compound question requiring data from multiple distinct files
    target_query = "What are the regulatory litigations for Apple and what are the chip supply chain concerns for Nvidia?"
    rag_system.query(user_question=target_query, user_role="compliance_auditor")