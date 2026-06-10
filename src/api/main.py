import logging
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import cleanly from decoupled exception workspace
from src.exceptions.governance import FinancialGovernanceViolation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Compliance RAG Ingress Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(..., description="The ingress audit query string.")
    role: str = Field("compliance_auditor", description="RBAC profile role of the caller.")

class QueryResponse(BaseModel):
    status: str = Field(..., description="Execution status pass.")
    response: str = Field(..., description="The synthesized text answer or guardrail message.")


@app.exception_handler(FinancialGovernanceViolation)
async def governance_violation_handler(request: Request, exc: FinancialGovernanceViolation):
    """
    Global interceptor translating internal system faults into standard HTTP 403 Forbidden.
    Unified payload structure ensures downstream consumers parse the response key seamlessly.
    """
    logger.warning(f"Financial Governance Guardrail triggered: {exc.message}")
    return JSONResponse(
        status_code=403,
        content={
            "status": "security_block",
            "response": f"Access Denied: Ingress query dropped by financial data governance guardrails. Reason: {exc.message}"
        }
    )

rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    global rag_pipeline
    try:
        from src.ingestion.generator import AsyncComplianceRAGPipeline
        rag_pipeline = AsyncComplianceRAGPipeline()
        logger.info("🚀 All model parameters safely resident in memory. System ready for ingress API traffic.")
    except Exception as e:
        logger.critical(f"Critical failure initializing RAG storage matrices: {str(e)}")

@app.get("/health")
def health_check():
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="Engine Warmup In Progress...")
    return {"status": "healthy", "engine": "hot"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def execute_compliance_query(payload: QueryRequest):
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG engine is initializing or unavailable.")
    
    logger.info(f"Received API request for role: {payload.role.upper()}")
    try:
        # Awaits the fully non-blocking asynchronous pipeline query execution sequence
        final_answer = await rag_pipeline.query(user_question=payload.question, user_role=payload.role)
        return QueryResponse(status="success", response=final_answer)
    except FinancialGovernanceViolation:
        # Re-raise to trigger the global FastAPI middleware exception handler cleanly
        raise
    except Exception as e:
        logger.error(f"API pipeline execution crash: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal execution fault: {str(e)}")