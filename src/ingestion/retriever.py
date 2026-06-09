import logging
import re
from qdrant_client import QdrantClient
from qdrant_client.http import models  # Added for native Qdrant filter schemas
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder
from config.settings import settings

logger = logging.getLogger(__name__)

class ComplianceRetriever:
    def __init__(self, client: QdrantClient = None):
        logger.info("Initializing Local Retrieval Search Engine with Cross-Encoder Reranker...")
        
        # 1. Bi-Encoder for initial broad-net retrieval
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL_NAME,
            cache_folder="./model_cache"
        )
        
        # 2. Local Cross-Encoder Reranker Model
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", automodel_args={"cache_dir": "./model_cache"})
        self.client = client if client else QdrantClient(path=str(settings.QDRANT_PATH))
        self.collection_name = "financial_compliance_documents"

    def retrieve_context(self, query: str, user_role: str, limit: int = 2) -> dict:
        # --- PHASE 1: KEYWORD COMPLIANCE FIREWALL ---
        compliance_pattern = re.compile(
            r"(risk|compliance|regulatory|supply chain|manufacturing|market|audit|"
            r"financial|revenue|sec|filing|legal|disclosure|policy|guideline|report|laws|apple|nvidia)", 
            re.IGNORECASE
        )
        
        if not compliance_pattern.search(query):
            return {
                "status": "blocked",
                "message": "Query dropped by sovereign guardrail: Request does not contain valid compliance coordinates.",
                "context": []
            }

        # --- PHASE 2: ATTRIBUTE-BASED ACCESS CONTROL (ABAC) MAPPING ---
        if user_role == "compliance_auditor":
            # Auditors can query both public and restricted internal files
            allowed_classifications = ["public", "internal"]
        elif user_role == "guest_researcher":
            # Researchers are strictly restricted to public datasets
            allowed_classifications = ["public"]
        else:
            # Defensive rejection of invalid roles
            return {
                "status": "blocked",
                "message": f"Unauthorized Role Allocation: Profile '{user_role}' has no valid access coordinates.",
                "context": []
            }

        # Build dynamic metadata constraint filter matching your vector storage schema
        rbac_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.classification",
                    match=models.MatchAny(any=allowed_classifications)
                    )
                ]
            )

        # --- PHASE 3: BI-ENCODER EMBEDDING GENERATION ---
        query_vector = self.embed_model.get_text_embedding(query)

        # --- PHASE 4: BROAD NET VECTOR RETRIEVAL (WITH SECURE METADATA FILTERS) ---
        raw_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=rbac_filter,  # Strict database-level security barrier injected here
            limit=6  
        )

        if not raw_hits:
            return {
                "status": "blocked", 
                "message": f"Access Denied: No matching compliance context found within clearance profile '{user_role.upper()}'.",
                "context": []
            }

        # --- PHASE 5: CROSS-ENCODER RERANKING PASS ---
        rerank_pairs = [[query, hit.payload.get("text", "")] for hit in raw_hits]
        
        logger.info(f"Rerank Pass: Evaluating {len(raw_hits)} secure candidate chunks via BAAI/bge-reranker-base...")
        rerank_scores = self.reranker.predict(rerank_pairs)

        # --- PHASE 6: RE-SCORING & SYNTHESIS ---
        scored_hits = []
        for idx, score in enumerate(rerank_scores):
            scored_hits.append({
                "score": float(score),
                "text": raw_hits[idx].payload.get("text", ""),
                "source": raw_hits[idx].payload.get("source_file", "Unknown"),
                "page": raw_hits[idx].payload.get("page_number", "N/A")
            })
        
        # Sort entirely by the deep contextual cross-attention relevance score
        scored_hits = sorted(scored_hits, key=lambda x: x["score"], reverse=True)

        # Capture the absolute top-k requested chunks
        final_context = scored_hits[:limit]
        logger.info(f"Reranker complete. Selected top {len(final_context)} chunks with highest relevance profiles.")

        return {
            "status": "success",
            "context": final_context
        }