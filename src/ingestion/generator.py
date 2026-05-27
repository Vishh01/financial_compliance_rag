import logging
import requests
import json
import sys
from src.ingestion.retriever import ComplianceRetriever

# Import your exact settings configuration
from config.settings import settings

logger = logging.getLogger(__name__)

class ComplianceRAGPipeline:
    """
    Complete Local RAG Pipeline. Coordinates secure retrieval from Qdrant 
    and context-grounded text generation using the configuration's local Qwen model.
    """
    def __init__(self):
        self.retriever = ComplianceRetriever()
        # Dynamically read values from your settings file
        self.model_name = settings.LLM_MODEL  # Will be 'qwen2.5:1.5b'
        self.ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    def query(self, user_question: str, user_role: str = "public"):
        print(f"\n--- Processing Ingress Query [Role: {user_role.upper()}] ---")
        
        # 1. Fetch filtered context from our retrieval engine
        retrieval_response = self.retriever.retrieve_context(
            query=user_question, 
            user_role=user_role, 
            limit=3
        )
        
        if retrieval_response["status"] == "blocked":
            print(f"\n[SYSTEM GUARDRAIL BLOCK]: {retrieval_response['message']}\n")
            return

        # 2. Extract and format the raw text chunks for the LLM
        context_chunks = retrieval_response["context"]
        formatted_context = ""
        for idx, chunk in enumerate(context_chunks):
            formatted_context += f"--- Source Document: {chunk['source']} (Page {chunk['page']}) ---\n"
            formatted_context += f"Context Block:\n{chunk['text']}\n\n"

        # 3. Construct a strict System Prompt to eliminate hallucinations
        system_prompt = (
            "You are an elite corporate financial compliance auditor. Your task is to answer the user's question "
            "using ONLY the provided text blocks below. Adhere strictly to these execution constraints:\n"
            "1. Ground your answer entirely in the provided context. Do not extrapolate, invent, or assume facts.\n"
            "2. If the context does not contain the explicit information requested, state clearly: 'Based on active context pools, "
            "no explicit data is available to resolve this query.'\n"
            "3. Cite the exact Source Document and Page numbers inline when presenting facts.\n"
            "4. Maintain an objective, professional, analytical corporate tone."
        )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"=== TARGET REFERENCE CONTEXT ===\n"
            f"{formatted_context}\n"
            f"=================================\n\n"
            f"USER QUESTION: {user_question}\n"
            f"SMART COMPLIANCE RESPONSE:"
        )

        # 4. Prepare payload for the local Ollama API execution
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": True
        }

        # 5. Stream the final response directly from your local hardware graphics/CPU
        print(f"\n[LLM GENERATION STARTING VIA OLLAMA ({self.model_name})]:")
        try:
            response = requests.post(self.ollama_url, json=payload, stream=True)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    chunk_json = json.loads(line.decode("utf-8"))
                    token = chunk_json.get("response", "")
                    sys.stdout.write(token)
                    sys.stdout.flush()
            print("\n\n--- End of Generation Pass ---\n")
            
        except requests.exceptions.ConnectionError:
            print(f"\nERROR: Unable to connect to local Ollama instance at {self.ollama_url}. Ensure 'ollama run qwen2.5:1.5b' has been executed in a terminal.\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    
    # Initialize the master pipeline
    rag_system = ComplianceRAGPipeline()
    
    # Execute a secure inquiry tracking an auditor clearance
    target_query = "What specific risk factors are associated with laws, regulations, and audits?"
    rag_system.query(user_question=target_query, user_role="compliance_auditor")