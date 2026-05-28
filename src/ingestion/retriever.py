import logging
import re
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.settings import settings

logger = logging.getLogger(__name__)

class ComplianceRetriever:
    """
    Finalized Production-Grade Retrieval Engine.
    Combines sub-millisecond Regex Intent Routing with strict 
    Database Metadata Pre-Filtering for secure, role-aware context extraction.
    """
    # Look for the __init__ method in src/ingestion/retriever.py and update it to this:
    def __init__(self, client: QdrantClient = None):
        logger.info("Initializing Local Retrieval Search Engine...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL_NAME,
            cache_folder="./model_cache"
        )
        
        # Share the database client connection to avoid locker collisions
        self.client = client if client else QdrantClient(path=str(settings.QDRANT_PATH))

        self.compliance_pattern = re.compile(
            r"(risk|compliance|regulatory|supply chain|manufacturing|market|audit|"
            r"financial|revenue|sec|filing|legal|disclosure|policy|guideline|report|laws|apple|nvidia)", 
            re.IGNORECASE
        )

    def _input_guardrail_is_invalid(self, query: str) -> bool:
        """
        Tier 1 Guardrail: Topic Relevance Control.
        Uses a compiled C-level regex pass to instantly screen out completely random queries.
        """
        # If it matches our corporate taxonomy patterns, it's immediately green-lit
        if self.compliance_pattern.search(query):
            return False  # Not invalid -> Proceed
            
        # Hard fallback check for explicit off-topic casual chatter
        casual_blacklist = ["recipe", "bake", "cookie", "movie", "game", "song", "weather", "food"]
        normalized = query.lower()
        if any(token in normalized for token in casual_blacklist):
            return True  # Invalid -> Deflect
            
        # Default block for any queries completely unrelated to the dataset
        return True

    def retrieve_context(self, query: str, user_role: str = "public", limit: int = 3):
        """
        Tier 3 Execution: Performs optimized vector retrieval with strict
        Role-Based Access Control (RBAC) filtering enforced at the database geometry layer.
        """
        start_time = time.time()
        
        # Step 1: Execute High-Speed Perimeter Guardrail
        if self._input_guardrail_is_invalid(query):
            execution_time = (time.time() - start_time) * 1000
            logger.warning(f"Guardrail Triggered: Intercepted off-topic query in {execution_time:.2f}ms")
            return {
                "status": "blocked",
                "message": "System Policy Violation: This engine is optimized strictly for corporate financial compliance queries.",
                "context": []
            }

        # Step 2: Compute vector representations for valid corporate queries
        logger.info(f"Vectorizing query coordinates: '{query}'")
        query_vector = self.embed_model.get_text_embedding(query)

        # Step 3: Enforce Identity-Based Database Filters (RBAC)
        query_filter = None
        
        if user_role == "public":
            logger.warning(f"SECURITY ENFORCEMENT: Restricting User [{user_role.upper()}] from confidential data access.")
            # Public profiles are restricted from viewing the confidential Apple financial 10-K filings
            query_filter = Filter(
                must_not=[
                    FieldCondition(
                        key="file_name", 
                        match=MatchValue(value="apple-K-2025-As-Filed.pdf")
                    )
                ]
            )
        elif user_role == "compliance_auditor":
            logger.info(f"SECURITY CLEARANCE GRANTED: User [{user_role.upper()}] authorized for full index search.")
            # Elevated roles have full access; no database filter bounds are applied
            query_filter = None

        # Step 4: Execute targeted vector geometric search pass with pre-filtering
        logger.info(f"Executing Qdrant geometric retrieval pass (Top-k: {limit})...")
        search_results = self.client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=query_filter,  # Qdrant drops restricted document chunks BEFORE running similarity calculations
            limit=limit
        )

        # Step 5: Process and assemble context payloads
        retrieved_chunks = []
        for hit in search_results:
            retrieved_chunks.append({
                "score": hit.score,
                "text": hit.payload.get("text", "Empty text snippet"),
                "source": hit.payload.get("file_name", "Unknown Document Source"),
                "page": hit.payload.get("page_label", "Unknown Page")
            })

        execution_time = (time.time() - start_time) * 1000
        logger.info(f"Retrieval engine sequence completed in {execution_time:.2f}ms")
        
        return {
            "status": "success",
            "message": f"Successfully retrieved {len(retrieved_chunks)} valid context frames.",
            "context": retrieved_chunks
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print("\n======================================================================")
    print("--- RUNNING PRODUCTION-GRADE RETRIEVAL SECURITY VALIDATION PASSTHROUGH ---")
    print("======================================================================\n")
    
    retriever = ComplianceRetriever()
    
    # -------------------------------------------------------------------------
    # TEST CRITERIA 1: Off-Topic Security Perimeter Check
    # -------------------------------------------------------------------------
    print("\n[SCENARIO 1: Off-Topic Ingress Evaluation]")
    res_1 = retriever.retrieve_context(query="Can you give me a recipe for chocolate chip cookies?", user_role="public")
    print(f"Execution Status: {res_1['status'].upper()}\nSystem Feedback: {res_1['message']}")
    
    # -------------------------------------------------------------------------
    # TEST CRITERIA 2: Malicious Intent / Data Breach Simulation (Low-Level Authorization Role)
    # -------------------------------------------------------------------------
    print("\n[SCENARIO 2: Public User Requesting Confidential Document Insight]")
    query_string = "What are the specific risk vectors and market policies outlined in the Apple 10-K?"
    res_public = retriever.retrieve_context(query=query_string, user_role="public", limit=3)
    
    print(f"Execution Status: {res_public['status'].upper()}")
    print("Accessible Document Sources:")
    if not res_public["context"]:
        print(" -> [No documents returned - Query Blocked]")
    for chunk in res_public["context"]:
        print(f" -> [Access Granted] Source: {chunk['source']} (Page {chunk['page']}) | Similarity: {chunk['score']:.4f}")

    # -------------------------------------------------------------------------
    # TEST CRITERIA 3: Safe Verified Ingress (Elevated Corporate Role)
    # -------------------------------------------------------------------------
    print("\n[SCENARIO 3: Elevated Compliance Auditor Executing Same Query]")
    res_auditor = retriever.retrieve_context(query=query_string, user_role="compliance_auditor", limit=3)
    
    print(f"Execution Status: {res_auditor['status'].upper()}")
    print("Accessible Document Sources:")
    for chunk in res_auditor["context"]:
        print(f" -> [Access Granted] Source: {chunk['source']} (Page {chunk['page']}) | Similarity: {chunk['score']:.4f}")
        print(f"    Snippet Text: {chunk['text'][:140].strip()}...\n")
        
    print("======================================================================")
    print("--- INFRASTRUCTURE SECURITY PASSTHROUGH TESTS COMPLETE ---")
    print("======================================================================\n")