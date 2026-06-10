import logging
import json
import time
import asyncio
import httpx
from qdrant_client import QdrantClient

from src.ingestion.retriever import ComplianceRetriever
from src.ingestion.query_validator import QueryDeconstructor
from src.ingestion.semantic_cache import SemanticCache
from config.settings import settings
from src.exceptions.governance import FinancialGovernanceViolation

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
        logger.info(f"Async worker spawned for target: '{sub_query}'")
        return self.retriever.retrieve_context(query=sub_query, user_role=user_role, limit=2)

    async def query(self, user_question: str, user_role: str = "compliance_auditor") -> str:
        print(f"\n--- Processing Ingress Query [Role: {user_role.upper()}] ---")
        start_time = time.time()
        
        # 1. Secure Caching Pass
        cached_answer = self.cache.check_cache(user_question, user_role)
        if cached_answer:
            return cached_answer

        # 2. Asynchronous Query Validation / Deconstruction Pass
        sub_queries = await self.deconstructor.validate_and_deconstruct(user_question)
        
        if "SECURITY_VIOLATION_TRIGGERED" in sub_queries:
            raise FinancialGovernanceViolation("Adversarial prompt signature or data egress trick detected.")

        # 3. Concurrent Retrieval Lookups
        tasks = [self.async_retrieve_worker(sub_q, user_role) for sub_q in sub_queries]
        retrieval_responses = await asyncio.gather(*tasks)

        aggregated_context_chunks = []
        seen_payloads = set()

        for response in retrieval_responses:
            if response is None:
                raise FinancialGovernanceViolation("Retriever returned empty or unvalidated context matrices.")
            if isinstance(response, dict) and response.get("status") == "blocked":
                raise FinancialGovernanceViolation(response.get("message"))
            if not isinstance(response, dict) or "context" not in response:
                raise FinancialGovernanceViolation("Downstream payload failed internal schema layout constraints.")

            for chunk in response["context"]:
                # Safe fallback parsing for dynamic unique identification tracking
                source_file = chunk.get("file_name") or chunk.get("source") or "unknown_file"
                page_ref = chunk.get("page_label") or chunk.get("page") or "N/A"
                
                chunk_id = f"{source_file}_{page_ref}_{hash(chunk['text'])}"
                if chunk_id not in seen_payloads:
                    seen_payloads.add(chunk_id)
                    aggregated_context_chunks.append(chunk)

        if not aggregated_context_chunks:
            return f"No matching context found for the provided query filters under role profile: {user_role.upper()}."

        # 4. Formatted Assembly with Enterprise Safe Fallback Mapping Layer
        formatted_context_blocks = []
        for c in aggregated_context_chunks:
            # --- ✅ CRITICAL RECONCILIATION LAYER ---
            # Resolves the dictionary key misalignment between embedder metadata variations
            source_doc = c.get("file_name") or c.get("source") or "Unknown_Compliance_Document.pdf"
            page_number = c.get("page_label") or c.get("page") or "N/A"
            text_data = c.get("text", "")

            block = (
                f"--- START OF RECORDFILE: {source_doc} (PAGE REFERENCE: {page_number}) ---\n"
                f"CONTENT CHUNK:\n{text_data}\n"
                f"--- END OF RECORDFILE ---\n\n"
            )
            formatted_context_blocks.append(block)
            
        formatted_context = "".join(formatted_context_blocks)

        # 5. Production Constraints Prompt Injection (Hardened Version)
        system_prompt = (
            "You are an elite, zero-tolerance corporate financial compliance auditor operating within strict legal guidelines.\n"
            "Your explicit assignment is to synthesize a compliance response derived strictly from the provided context blocks below.\n\n"
            "CRITICAL OPERATIONAL CONSTRAINTS:\n"
            "1. ANCHORED CITATION VERIFICATION: Every single factual assertion, revenue metric, or liability claim you generate must be immediately "
            "followed by an explicit inline citation matching this exact layout: [FILE: source_name, PAGE: page_num]. Never synthesize broad claims without this anchor.\n"
            "   Example: NVIDIA powered two of the world's top supercomputers in 2018 [FILE: nvidia_2018.pdf, PAGE: 3].\n"
            "2. ABSOLUTE CONTEXT GROUNDING: If a topic, balance sheet item, or company requested in the prompt is missing or incomplete within the context blocks, "
            "state explicitly: 'Information unavailable: Context lacks verified ledger documentation for [Company/Topic]'. Do not utilize parametric memory.\n"
            "3. NO EXTRAPOLATION: You must maintain a zero-temperature analytical profile. Report raw facts exactly as stated without speculative assumptions.\n\n"
            "RESPONSE FORMAT MANDATE:\n"
            "Review the text chunk markers carefully. You must structure your output paragraphs so that every single sentence concludes with its respective [FILE: ..., PAGE: ...] tag."
        )

        # We inject a clear response schema block at the very end to guide the LLM's initial tokens
        full_prompt = (
            f"SYSTEM RULES:\n{system_prompt}\n\n"
            f"=== VERIFIED DOCUMENTATION CONTEXT ===\n{formatted_context}\n"
            f"=== INGRESS QUERY ===\n{user_question}\n\n"
            f"AUDITOR SYNTHESIS RESPONSE (Remember to append [FILE: <source>, PAGE: <page>] to every assertion):\n"
        )
        
        # 6. Non-Blocking Output Cycle Call
        payload = {
            "model": self.model_name, "prompt": full_prompt, "stream": False,
            "keep_alive": 0, "options": { "temperature": 0.0 }
        }
        
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                response = await client.post(self.ollama_url, json=payload)
                response.raise_for_status()
                final_text = response.json().get("response", "").strip()
                self.cache.update_cache(user_question, final_text, user_role)
                return final_text
            except Exception as e:
                return f"Internal generation failure occurred: {str(e)}"