import logging
import re
from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder
from config.settings import settings

logger = logging.getLogger(__name__)

class ComplianceRetriever:
    def __init__(self, client: QdrantClient = None):
        logger.info("Initializing Local Retrieval Search Engine with Cross-Encoder Reranker...")
        self.embed_model = HuggingFaceEmbedding(
            model_name=settings.EMBED_MODEL_NAME,
            cache_folder="./model_cache"
        )
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

        # --- PHASE 2: ATTRIBUTE-BASED ACCESS CONTROL (ABAC) ---
        if user_role == "compliance_auditor":
            allowed_classifications = ["public", "internal"]
        elif user_role == "guest_researcher":
            allowed_classifications = ["public"]
        else:
            return {
                "status": "blocked",
                "message": f"Unauthorized Role Allocation: Profile '{user_role}' has no valid access coordinates.",
                "context": []
            }

        # --- ✅ SYSTEM CORRECTION: ROBUST ABAC PATH MATCHING ---
        # Checks both flat and nested properties to avoid mapping drops across varying Qdrant versions
        rbac_filter = models.Filter(
            should=[
                models.FieldCondition(key="metadata.classification", match=models.MatchAny(any=allowed_classifications)),
                models.FieldCondition(key="classification", match=models.MatchAny(any=allowed_classifications))
            ]
        )

        # --- PHASE 3: BI-ENCODER AND BROAD NET RETRIEVAL ---
        query_vector = self.embed_model.get_text_embedding(query)
        raw_hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=rbac_filter,
            limit=6  
        )

        if not raw_hits:
            return {
                "status": "blocked", 
                "message": f"Access Denied: No matching compliance context found within profile '{user_role.upper()}'.",
                "context": []
            }

        # --- PHASE 4: CROSS-ENCODER RERANKING ---
        rerank_pairs = [[query, hit.payload.get("text", "")] for hit in raw_hits]
        logger.info(f"Rerank Pass: Evaluating {len(raw_hits)} chunks via Cross-Attention...")
        rerank_scores = self.reranker.predict(rerank_pairs)

        # --- PHASE 5: MAPPING SCHEMA LAYOUT SYNCHRONIZATION ---
        scored_hits = []
        for idx, score in enumerate(rerank_scores):
            payload = raw_hits[idx].payload
            
            # Extract names utilizing absolute safe fallbacks
            file_name = payload.get("file_name") or payload.get("source") or "Unknown_Compliance_Document.pdf"
            page_label = payload.get("page_label") or payload.get("page") or "N/A"
            text_content = payload.get("text", "")

            scored_hits.append({
                "score": float(score),
                "text": text_content,
                "file_name": file_name,     # Explicit key preservation
                "page_label": page_label,   # Explicit key preservation
                "source": file_name,        # Backward compatibility layer
                "page": page_label          # Backward compatibility layer
            })
        
        scored_hits = sorted(scored_hits, key=lambda x: x["score"], reverse=True)
        final_context = scored_hits[:limit]
        logger.info(f"Selected top {len(final_context)} chunks with highest relevance profiles.")

        return {"status": "success", "context": final_context}