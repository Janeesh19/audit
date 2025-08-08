# app.py
import json
import io
import os
import streamlit as st
import google.generativeai as genai
from datetime import datetime

# --------- Fixed config ----------
MODEL_ID = "gemini-1.5-pro"      # keep this static
OUTPUT_FILENAME_DEFAULT = "audit_output.txt"
AUTH_PASSWORD = "audit9099"
# ---------------------------------

st.set_page_config(page_title="Auditor", page_icon="📄", layout="centered")

# ---------- Password gate ----------
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    st.title("Audit Writer")
    st.subheader("Sign in")
    pwd = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pwd == AUTH_PASSWORD:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()
# -----------------------------------

st.title("Audit Writer")

with st.sidebar:
    st.header("Options")
    output_filename = st.text_input("Download filename", value=OUTPUT_FILENAME_DEFAULT)

uploaded = st.file_uploader("Upload COT JSON", type=["json"])

def get_api_key() -> str:
    # Prefer Streamlit secrets, fall back to environment variable for local dev
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_API_KEY", "")

def build_prompt(cot_obj: dict) -> str:
    cot_text = json.dumps(cot_obj, indent=2, ensure_ascii=False)
    return (
        "You are an expert financial auditor. Using the following chain-of-thought JSON "
        "about auditing standard SA-230 as your reasoning base:\n\n"
        f"{cot_text}\n\n"
        "Please produce:\n"
        "1. A concise scenario of auditing a mid-sized company.\n"
        "2. The high-level steps involved in the audit.\n"
        "3. A list of all documents and working papers to be considered.\n"
        "4. A detailed log mapping each output section back to the referenced COT entries. "
        "For each referenced entry, include both:\n"
        "   • The `triplet` text member\n"
        "   • A 2–3 sentence explanation that expands on why that triplet supports this part of the audit\n\n"
        "Use a clear, numbered structure and ensure each explanation is at least two sentences long."
    )

def call_gemini(prompt: str) -> str:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No API key configured. Add GOOGLE_API_KEY in Streamlit secrets or as an environment variable.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_ID)
    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip()

if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

if uploaded:
    try:
        cot_data = json.load(uploaded)
        with st.expander("Preview of uploaded JSON"):
            st.code(json.dumps(cot_data, indent=2, ensure_ascii=False), language="json")
    except Exception as e:
        st.error(f"Could not read JSON. {e}")
        cot_data = None
else:
    cot_data = None

if st.button("Generate audit output"):
    if not cot_data:
        st.error("Please upload a valid JSON file first.")
    else:
        with st.spinner("Generating text..."):
            try:
                prompt = build_prompt(cot_data)
                body = call_gemini(prompt)
                if not body:
                    st.error("The model returned no text. Check quotas and model availability.")
                else:
                    header = "=== Audit Scenario, Steps, Documents & COT Reference Log ==="
                    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                    st.session_state.generated_text = f"{header}\nGenerated: {timestamp}\n\n{body}"
            except Exception as e:
                st.error(f"Generation failed. {e}")

if st.session_state.generated_text:
    st.subheader("Preview")
    st.text_area("Generated text", st.session_state.generated_text, height=400)

    buf = io.BytesIO(st.session_state.generated_text.encode("utf-8"))
    st.download_button(
        label="Download .txt",
        data=buf,
        file_name=output_filename or "audit_output.txt",
        mime="text/plain",
        type="primary",
    )
