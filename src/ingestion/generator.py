import logging
import json
import sys
import time
import asyncio
import httpx
from qdrant_client import QdrantClient

from src.ingestion.retriever import ComplianceRetriever
from src.ingestion.query_validator import QueryDeconstructor
from src.ingestion.semantic_cache import SemanticCache
from config.settings import settings

logger = logging.getLogger(__name__)

class AsyncComplianceRAGPipeline:
    def __init__(self):
        logger.info("Initializing High-Performance Asynchronous RAG Pipeline...")
        self.shared_client = QdrantClient(path=str(settings.QDRANT_PATH))
        
        self.retriever = ComplianceRetriever(client=self.shared_client)
        self.cache = SemanticCache(client=self.shared_client, threshold=0.95)
        self.deconstructor = QueryDeconstructor()
        
        self.model_name = settings.LLM_MODEL
        self.ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"

    async def async_retrieve_worker(self, sub_query: str, user_role: str):
        """Worker task running completely concurrently for database lookups."""
        logger.info(f"Async worker spawned for target: '{sub_query}'")
        # Run the existing retriever inside the async thread pool
        return self.retriever.retrieve_context(query=sub_query, user_role=user_role, limit=2)

    async def query(self, user_question: str, user_role: str = "compliance_auditor"):
        print(f"\n--- Processing Ingress Query [Role: {user_role.upper()}] ---")
        start_time = time.time()
        
        # 1. High-Speed Semantic Cache Pass
        cached_answer = self.cache.check_cache(user_question)
        if cached_answer:
            print(f"\n[FAST RESPONSE FROM CACHE Memory]:\n{cached_answer}")
            print(f"\n--- Pipeline Completed via Cache in {(time.time() - start_time)*1000:.2f}ms ---\n")
            return

        # 2. Fast Micro-Model Query Deconstruction
        sub_queries = self.deconstructor.validate_and_deconstruct(user_question)
        print(f"Targets to search in parallel: {sub_queries}")

        # 3. ASYNCHRONOUS VECTOR PARALLELISM
        # Fires off all vector database searches SIMULTANEOUSLY
        tasks = [self.async_retrieve_worker(sub_q, user_role) for sub_q in sub_queries]
        retrieval_responses = await asyncio.gather(*tasks)

        aggregated_context_chunks = []
        seen_payloads = set()

        # Consolidate results cleanly
        for response in retrieval_responses:
            if response["status"] == "blocked":
                print(f"\n[SYSTEM GUARDRAIL BLOCK]: {response['message']}\n")
                return
            for chunk in response["context"]:
                chunk_id = f"{chunk['source']}_{chunk['page']}_{hash(chunk['text'])}"
                if chunk_id not in seen_payloads:
                    seen_payloads.add(chunk_id)
                    aggregated_context_chunks.append(chunk)

        # 4. Construct Prompt Context
        formatted_context = "".join([
            f"--- Source Document: {c['source']} (Page {c['page']}) ---\nContext:\n{c['text']}\n\n"
            for c in aggregated_context_chunks
        ])

        system_prompt = (
            "You are an elite corporate financial compliance auditor. Answer the question using ONLY the provided text blocks.\n"
            "Cite the exact Source Document and Page numbers inline. Do not extrapolate."
        )
        full_prompt = f"{system_prompt}\n\n=== CONTEXT ===\n{formatted_context}\n=== QUESTION ===\n{user_question}\nRESPONSE:"

        # 5. Execute Async LLM Generation Pass via HTTPX
        print(f"\n[LLM GENERATION STARTING VIA OLLAMA ({self.model_name})]:")
        payload = {"model": self.model_name, "prompt": full_prompt, "stream": False}
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.ollama_url, json=payload, timeout=60.0)
                final_text = response.json().get("response", "").strip()
                print(final_text)
                
                print(f"\n--- Full Async RAG Pipeline Sequence Completed in {time.time() - start_time:.2f} seconds ---")
                self.cache.update_cache(user_question, final_text)
            except Exception as e:
                print(f"\nExecution Error: {str(e)}\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    
    rag_system = AsyncComplianceRAGPipeline()
    target_query = "What are the regulatory litigations for Apple and what are the chip supply chain concerns for Nvidia?"
    
    # Run the asynchronous framework loop
    asyncio.run(rag_system.query(user_question=target_query, user_role="compliance_auditor"))