import streamlit as str_ui
import requests
import time
import logging

# Setup structural logging for UI layer tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://127.0.0.1:8000"

str_ui.set_page_config(
    page_title="Financial Governance RAG Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ─── SIDEBAR: APPLICATION INFORMATION & VECTOR ENGINE STATUS ───
with str_ui.sidebar:
    str_ui.title("🛡️ Governance Control Center")
    str_ui.markdown("---")
    
    str_ui.subheader("📌 Application Profiles")
    str_ui.info(
        "**Engine Type:** Enterprise Advanced RAG\n\n"
        "**Compliance Target:** SEC Form 10-K Disclosures\n\n"
        "**Security Layer:** Local Attribute-Based Access Control (ABAC)"
    )
    
    str_ui.markdown("---")
    str_ui.subheader("⚙️ Vector Engine Monitor")
    
    # Live health handshake checking system state
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if health_response.status_code == 200 and health_response.json().get("status") == "healthy":
            str_ui.success("● ENGINE STATUS: ONLINE / HOT")
            str_ui.caption("Qdrant Embedded DB and Local Transformers are fully active.")
        else:
            str_ui.warning("● ENGINE STATUS: WARMING UP")
            str_ui.caption("Models are initializing or caching layers are resident loading.")
    except Exception:
        str_ui.error("● ENGINE STATUS: OFFLINE")
        str_ui.caption("Unable to establish communication network with FastAPI router backend.")

    str_ui.markdown("---")
    str_ui.caption("v2.0.0 • Production Hardened State Enforced")


# ─── MAIN PRESENTATION FRAME ───
str_ui.title("Financial Compliance Audit Engine")
str_ui.write("Execute advanced cross-entity compliance sweeps with zero parametric hallucination.")
str_ui.markdown("---")

# 1. User Role Selector Layer placed directly ABOVE the query text input block
str_ui.subheader("🔐 Step 1: Enforce Security Profile")
user_role = str_ui.selectbox(
    "Select Current Identity Profile (ABAC Enforcement Routing Key):",
    options=["compliance_auditor", "guest_researcher"],
    index=0,
    help="Auditors have clearance for public and internal files; Researchers are strictly constrained to public records."
)

str_ui.markdown(" ")

# 2. Ingress Input Block Section
str_ui.subheader("🔍 Step 2: Formulate Audit Inquiry")
user_query = str_ui.text_area(
    "Enter Ingress Compliance Query:",
    placeholder="e.g., Explain how nvidia works or evaluate supply chain risks and revenue streams...",
    height=100
)

# Initialize persistent session tracking structures for metrics
if "final_report" not in str_ui.session_state:
    str_ui.session_state.final_report = None
if "execution_latency" not in str_ui.session_state:
    str_ui.session_state.execution_latency = None
if "status_code" not in str_ui.session_state:
    str_ui.session_state.status_code = None

# Action execution boundary
if str_ui.button("Execute Compliance Evaluation Sequence", type="primary"):
    if not user_query.strip():
        str_ui.error("Operational Block: Target inquiry text buffer cannot be empty.")
    else:
        with str_ui.spinner("Orchestrating multi-agent async retrieval loops and validation layers..."):
            payload = {
                "question": user_query.strip(),
                "role": user_role
            }
            
            # Start strict latency timing capture
            start_marker = time.time()
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/query",
                    json=payload,
                    timeout=None  # Allow local generation loops time to complete naturally
                )
                str_ui.session_state.execution_latency = time.time() - start_marker
                str_ui.session_state.status_code = response.status_code
                
                # Capture standard response matrices or security blocking payloads
                if response.status_code == 200:
                    str_ui.session_state.final_report = response.json().get("response", "No data returned.")
                elif response.status_code == 403:
                    # Capture security block reasons directly from the exception middleware
                    error_payload = response.json()
                    str_ui.session_state.final_report = error_payload.get("response", "Access Denied by guardrails.")
                else:
                    str_ui.session_state.final_report = f"Pipeline Processing Failure. Status Code: {response.status_code}"
                    
            except Exception as e:
                str_ui.session_state.execution_latency = time.time() - start_marker
                str_ui.session_state.status_code = 500
                str_ui.session_state.final_report = f"API Gate Connection Error: Failed to hit ingestion engine. Detail: {str(e)}"

str_ui.markdown("---")

# ─── DISPLAY RESULTS & INLINE METRICS ───
if str_ui.session_state.final_report:
    str_ui.subheader("📄 Finalized Compliance Report")
    
    # Check if the output contains a security violation signature or standard success
    if str_ui.session_state.status_code == 403:
        str_ui.error(str_ui.session_state.final_report)
    elif str_ui.session_state.status_code == 200:
        str_ui.info("The responses below were synthesized under zero-tolerance constraints using strictly isolated, verified local document contexts.")
        str_ui.write(str_ui.session_state.final_report)
    else:
        str_ui.warning(str_ui.session_state.final_report)
        
    str_ui.markdown("---")
    
    # 3. Execution Latency Metrics rendering explicitly at the BOTTOM of the visual viewport
    if str_ui.session_state.execution_latency is not None:
        str_ui.subheader("📊 Performance Observability Logs")
        col1, col2 = str_ui.columns(2)
        with col1:
            str_ui.metric(
                label="System Ingress-to-Egress Latency",
                value=f"{str_ui.session_state.execution_latency:.3f} seconds",
                delta="Cached Match" if str_ui.session_state.execution_latency < 1.0 else "Cold Compute Gen"
            )
        with col2:
            status_text = "SUCCESS (200 OK)" if str_ui.session_state.status_code == 200 else "BLOCKED (403 FORBIDDEN)"
            str_ui.metric(
                label="FastAPI Protocol Handshake Status",
                value=status_text
            )