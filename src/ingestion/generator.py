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

# Decoupled exception reference completely protecting runtime from circular loops
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
        
        # 1. High-Speed Semantic Cache Pass
        cached_answer = self.cache.check_cache(user_question)
        if cached_answer:
            print(f"\n[FAST RESPONSE FROM CACHE Memory]:\n{cached_answer}")
            print(f"\n--- Pipeline Completed via Cache in {(time.time() - start_time)*1000:.2f}ms ---\n")
            return cached_answer

        # 2. Query Deconstruction Pass
        sub_queries = self.deconstructor.validate_and_deconstruct(user_question)
        
        # 🛑 SECURITY DEFENSE BREAKPOINT 
        if "SECURITY_VIOLATION_TRIGGERED" in sub_queries:
            print("\n[SECURITY AUDIT BLOCK]: Execution terminated due to adversarial prompt signature.")
            raise FinancialGovernanceViolation("Adversarial prompt signature or data egress trick detected.")

        print(f"Targets to search in parallel: {sub_queries}")

        # 3. Asynchronous Vector Parallel Retrieval via Gather Tree
        tasks = [self.async_retrieve_worker(sub_q, user_role) for sub_q in sub_queries]
        retrieval_responses = await asyncio.gather(*tasks)

        aggregated_context_chunks = []
        seen_payloads = set()

        for response in retrieval_responses:
            if response is None:
                print("\n[SYSTEM GUARDRAIL BLOCK]: Retriever returned an implicit empty None payload.\n")
                raise FinancialGovernanceViolation("Retriever returned empty or unvalidated context matrices.")
                
            if isinstance(response, dict) and response.get("status") == "blocked":
                block_message = response.get("message", "Request does not contain valid compliance coordinates.")
                print(f"\n[SYSTEM GUARDRAIL BLOCK]: {block_message}\n")
                raise FinancialGovernanceViolation(block_message)
                
            if not isinstance(response, dict) or "context" not in response:
                print("\n[SYSTEM GUARDRAIL BLOCK]: Malformed response format encountered in pipeline structure.\n")
                raise FinancialGovernanceViolation("Downstream payload failed internal schema layout constraints.")

            for chunk in response["context"]:
                # Generates deterministic unique identification signature per payload block
                chunk_id = f"{chunk['source']}_{chunk['page']}_{hash(chunk['text'])}"
                if chunk_id not in seen_payloads:
                    seen_payloads.add(chunk_id)
                    aggregated_context_chunks.append(chunk)

        if not aggregated_context_chunks:
            return f"No matching context found for the provided query filters under role profile: {user_role.upper()}."

        # 4. Formatted Context Aggregation Assembly Mapping
        formatted_context = "".join([
            f"--- START OF RECORDFILE: {c['source']} (PAGE REFERENCE: {c['page']}) ---\n"
            f"CONTENT CHUNK:\n{c['text']}\n"
            f"--- END OF RECORDFILE ---\n\n"
            for c in aggregated_context_chunks
        ])

        # 5. Advanced Zero-Tolerance Compliance Prompt Engineering Constraints
        system_prompt = (
            "ROLE AND ETHICAL CLEARANCE:\n"
            "You are an elite, zero-tolerance corporate financial compliance auditor operating within strict legal guidelines.\n"
            "Your explicit assignment is to synthesize a compliance response derived strictly from the provided context blocks below.\n\n"
            "CRITICAL OPERATIONAL CONSTRAINTS:\n"
            "1. ANCHORED CITATION VERIFICATION: Every single factual assertion, revenue metric, or liability claim you generate must be immediately "
            "followed by an explicit inline citation matching this exact layout: [FILE: source_name, PAGE: page_num]. Never synthesize broad claims without this anchor.\n"
            "2. ABSOLUTE CONTEXT GROUNDING: If a topic, balance sheet item, or company requested in the prompt is missing or incomplete within the context blocks, "
            "state explicitly: 'Information unavailable: Context lacks verified ledger documentation for [Company/Topic]'. Do not utilize parametric memory.\n"
            "3. ISOLATED ENTITY BOUNDARIES: Process distinct company audits completely separately. If the request spans multiple entities (e.g., Apple and Nvidia), "
            "you must isolate their evaluations into separate thematic headers. Mixing data lines or comparative financial metrics across companies is strictly forbidden.\n"
            "4. NO EXTRAPOLATION: You must maintain a zero-temperature analytical profile. Report raw facts and verified disclosures exactly as stated without speculative forward-looking assumptions."
        )
        full_prompt = f"{system_prompt}\n\n=== VERIFIED DOCUMENTATION CONTEXT ===\n{formatted_context}\n=== INGRESS QUERY ===\n{user_question}\n\nAUDITOR SYNTHESIS RESPONSE:"

        print(f"\n[LLM GENERATION STARTING VIA OLLAMA ({self.model_name})]:")
        payload = {
            "model": self.model_name, 
            "prompt": full_prompt, 
            "stream": False,
            "keep_alive": 0, 
            "options": { "temperature": 0.0 }
        }
        
        # 6. Execute Async Generation Loop
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                response = await client.post(self.ollama_url, json=payload)
                response.raise_for_status()
                final_text = response.json().get("response", "").strip()
                print(final_text)
                print(f"\n--- Full Async RAG Pipeline Sequence Completed in {time.time() - start_time:.2f} seconds ---")
                self.cache.update_cache(user_question, final_text)
                return final_text
            except Exception as e:
                import traceback
                print(f"\n[Execution Error Detail]: {str(e)}")
                print("--- Full Error Traceback ---")
                traceback.print_exc()
                print("----------------------------\n")
                return f"Internal generation failure occurred: {str(e)}"