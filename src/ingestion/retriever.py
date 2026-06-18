import logging
import gc
import re
import numpy as np
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
        # --- PHASE 1: SEMANTIC COMPLIANCE FIREWALL ---
        logger.info(f"Evaluating security clearance for ingress query: '{query}'")
        
        compliance_anchors = [
            "NVIDIA corporate financial compliance and regulatory risk analysis",
            "Apple SEC disclosure data ledger filings and legal audits",
            "Supply chain manufacturing vulnerabilities and revenue reports"
        ]
        
        target_vector = self.embed_model.get_text_embedding(query)
        anchor_vectors = [self.embed_model.get_text_embedding(anchor) for anchor in compliance_anchors]
        
        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
        max_similarity = max([cosine_similarity(target_vector, av) for av in anchor_vectors])
        logger.info(f"Ingress semantic safety validation score: {max_similarity:.4f}")
        
        if max_similarity < 0.38:
            return {
                "status": "blocked",
                "message": f"Query dropped by semantic guardrail: Request relevance profile ({max_similarity:.3f}) falls below corporate validation metrics.",
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

        # --- PHASE 4: CROSS-ENCODER RERANKING WITH CPU CLEANUP ---
        rerank_pairs = [[query, hit.payload.get("text", "")] for hit in raw_hits]
        logger.info(f"Rerank Pass: Evaluating {len(raw_hits)} chunks via Cross-Attention...")
        
        try:
            rerank_scores = self.reranker.predict(rerank_pairs)
        finally:
            # Force explicit release of CPU matrix calculation maps immediately
            gc.collect()

        # --- PHASE 5: MAPPING SCHEMA LAYOUT SYNCHRONIZATION WITH WINDOW EXPANSION ---
        scored_hits = []
        for idx, score in enumerate(rerank_scores):
            hit = raw_hits[idx]
            payload = hit.payload if hasattr(hit, "payload") and hit.payload is not None else (hit.get("payload", {}) if isinstance(hit, dict) else {})

            file_name = payload.get("file_name") or payload.get("source")
            page_label = payload.get("page_label") or payload.get("page")
            text_content = payload.get("text", "")
            chunk_idx = payload.get("chunk_index") 

            if not file_name and isinstance(payload, dict):
                meta = payload.get("metadata", {})
                if isinstance(meta, dict):
                    file_name = meta.get("file_name") or meta.get("source")
                    page_label = meta.get("page_label") or meta.get("page")

            final_file = str(file_name).strip() if file_name else "Unknown_Compliance_Document.pdf"
            final_page = str(page_label).strip() if page_label else "N/A"

            logger.info(f"[DEBUG RETRIEVER] Chunk {idx+1} -> File: {final_file}, Page: {final_page}")

            scored_hits.append({
                "score": float(score),
                "text": str(text_content).strip(),
                "file_name": final_file,     
                "page_label": final_page,   
                "source": final_file,        
                "page": final_page,
                "chunk_index": chunk_idx
            })
        
        scored_hits = sorted(scored_hits, key=lambda x: x["score"], reverse=True)
        top_hits = scored_hits[:limit]

        # --- PHASE 6: NEIGHBORHOOD CHUNK STITCHING WITH STRUCTURED ANCHOR INJECTION ---
        final_context = []
        for hit in top_hits:
            current_text = hit["text"]
            c_idx = hit["chunk_index"]
            final_file = hit["file_name"]
            final_page = hit["page_label"]
            
            if c_idx is None:
                # Even for un-indexed blocks, enforce structural banner framing
                hit["text"] = f"[METADATA ANCHOR | FILE: {final_file} | PAGE: {final_page}]\n{current_text}\n[END OF CONTEXT NODE]"
                final_context.append(hit)
                continue
                
            logger.info(f"Expanding context window for chunk index {c_idx} from file {final_file}...")
            
            neighbor_scroll, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="file_name", match=models.MatchValue(value=final_file)),
                        models.FieldCondition(key="chunk_index", match=models.MatchAny(any=[c_idx - 1, c_idx + 1]))
                    ]
                ),
                limit=2,
                with_payload=True,
                with_vectors=False
            )
            
            neighbors = sorted(neighbor_scroll, key=lambda x: x.payload.get("chunk_index", 0))
            
            stitched_text = ""
            for n in neighbors:
                n_idx = n.payload.get("chunk_index")
                n_text = n.payload.get("text", "")
                if n_idx == c_idx - 1:
                    stitched_text += n_text + "\n\n"
            
            stitched_text += current_text
            
            for n in neighbors:
                n_idx = n.payload.get("chunk_index")
                n_text = n.payload.get("text", "")
                if n_idx == c_idx + 1:
                    stitched_text += "\n\n" + n_text

            # --- THE CRITICAL FIX ---
            # Explicitly inject the citation source string layout directly into the text property.
            # This makes it impossible for the model to lose track of the file context.
            hit["text"] = (
                f"[METADATA ANCHOR | FILE: {final_file} | PAGE: {final_page}]\n"
                f"{stitched_text}\n"
                f"[END OF CONTEXT NODE]"
            )
            final_context.append(hit)

        logger.info(f"Selected top {len(final_context)} chunks with stitched neighbor contexts.")
        return {"status": "success", "context": final_context}