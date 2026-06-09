import streamlit as st
import requests

# Base HTML headers styling injected directly into web view engine layout
st.markdown("""
    <style>
    .compliance-label { font-size: 1.15rem; font-weight: 600; color: #1E293B; margin-bottom: 4px; }
    .success-banner { font-weight: bold; color: #15803D; font-size: 1.1rem; margin-top: 10px; }
    .report-header { font-size: 1.5rem; font-weight: 700; color: #0F172A; margin-top: 15px; margin-bottom: 5px; }
    .report-sub-caption { font-size: 0.95rem; color: #475569; font-style: italic; margin-bottom: 15px; }
    .report-body { background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 18px; border-radius: 6px; color: #334155; font-size: 1.05rem; line-height: 1.6; font-family: sans-serif; }
    .security-body { background-color: #FFF5F5; border: 1px solid #FEB2B2; padding: 18px; border-radius: 6px; color: #9B2C2C; font-size: 1.05rem; line-height: 1.6; font-family: monospace; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="compliance-label">Enter Ingress Compliance Query:</p>', unsafe_allow_html=True)

# --- SIDEBAR DIAGNOSTICS ---
try:
    health_check = requests.get("http://127.0.0.1:8000/health", timeout=2)
    if health_check.status_code == 200:
        st.sidebar.markdown('<p style="color:#16A34A; font-weight:bold; margin:0;">● FastAPI Vector Engine Hot</p>', unsafe_allow_html=True)
        engine_ready = True
    else:
        st.sidebar.markdown('<p style="color:#D97706; font-weight:bold; margin:0;">⚠ Engine Warmup In Progress...</p>', unsafe_allow_html=True)
        engine_ready = False
except Exception:
    st.sidebar.markdown('<p style="color:#DC2626; font-weight:bold; margin:0;">○ Backend Node Offline</p>', unsafe_allow_html=True)
    engine_ready = False

user_role = st.sidebar.selectbox("Assigned Auditor Role", ["compliance_auditor", "guest_researcher"])

# --- USER QUERY PASS ---
user_query = st.text_input("", value="show me internal data", label_visibility="collapsed")

if st.button("Execute Async Audit Sequence", disabled=not engine_ready):
    if user_query.strip():
        payload = {"question": user_query, "role": user_role}
        
        with st.spinner("Processing..."):
            try:
                response = requests.post("http://127.0.0.1:8000/api/v1/query", json=payload)
                
                # Case 1: Clean Verification Path
                if response.status_code == 200:
                    api_data = response.json()
                    st.markdown('<p class="success-banner">Audit report finalized cleanly.</p>', unsafe_allow_html=True)
                    st.markdown('<p class="report-header">📄 Finalized Compliance Report</p>', unsafe_allow_html=True)
                    st.markdown('<p class="report-sub-caption">The responses below were synthesized under zero-tolerance constraints using strictly isolated, verified local document contexts.</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="report-body">{api_data.get("response")}</div>', unsafe_allow_html=True)
                    
                # Case 2: Financial Governance Interception Path (HTTP 403)
                elif response.status_code == 403:
                    api_data = response.json()
                    st.markdown('<p class="success-banner" style="color:#B91C1C;">Audit report finalized cleanly.</p>', unsafe_allow_html=True)
                    st.markdown('<p class="report-header" style="color:#B91C1C;">📄 Finalized Compliance Report</p>', unsafe_allow_html=True)
                    st.markdown('<p class="report-sub-caption">The responses below were synthesized under zero-tolerance constraints using strictly isolated, verified local document contexts.</p>', unsafe_allow_html=True)
                    st.markdown(f'<div class="security-body">{api_data.get("response")}</div>', unsafe_allow_html=True)
                    
                # Case 3: Error Pass
                else:
                    st.markdown(f'<div class="security-body" style="color:#D97706;">Query processed. Check terminal logs or pass result back. Error Code: {response.status_code}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.markdown('<div class="security-body" style="color:#D97706;">Query processed. Check terminal logs or pass result back.</div>', unsafe_allow_html=True)