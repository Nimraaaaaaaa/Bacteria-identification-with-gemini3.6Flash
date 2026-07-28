"""
Bacteria Identification AI - Streamlit App (FREE - Google Gemini)
====================================================================
Everything in one file — UI + Gemini API call + JSON parsing.

Get a free API key: https://aistudio.google.com/apikey

Setup:
    pip install -r requirements.txt

Run:
    streamlit run app.py
"""

import os
import json
import re
import textwrap
import html as html_lib

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Bacteria Identification AI",
    page_icon="🔬",
    layout="centered",
)

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.6-flash"  # current GA model (as of July 2026)
MAX_IMAGE_DIMENSION = 1568

SYSTEM_PROMPT = """You are a microbiology expert analyzing a gram stain microscope image.
Respond ONLY with valid JSON (no markdown, no backticks):
{
  "classification": "Gram Positive" | "Gram Negative" | "Indeterminate",
  "confidence": <integer 0-100>,
  "morphology": "Cocci" | "Bacilli" | "Diplococci" | "Spirilli" | "Mixed" | "Unclear",
  "color_observed": <string>,
  "possible_organisms": [<string>, <string>],
  "explanation": <2-3 sentence plain English explanation>,
  "clinical_note": <1 sentence clinical relevance>,
  "image_quality": "Good" | "Fair" | "Poor"
}
If this is not a gram stain image, set classification to Indeterminate
and explain why. Respond with ONLY the JSON object, nothing else."""


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
# Palette is grounded in the actual gram-stain reagents:
#   - Crystal violet (deep purple) marks Gram-positive retention
#   - Safranin (brick red) marks the Gram-negative counterstain
# Typography: Inter for interface text, IBM Plex Mono for data/readouts
# (numbers, labels) — evokes a lab instrument display rather than a
# generic web app.
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background: #F7F7F4;
    }

    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 720px;
    }

    /* ---------- Header ---------- */
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8A6BB8;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #14161C;
        margin: 0 0 6px 0;
        line-height: 1.15;
    }
    .app-subtitle {
        color: #6B7080;
        font-size: 0.98rem;
        margin-bottom: 2rem;
        max-width: 480px;
    }
    .header-rule {
        height: 1px;
        background: linear-gradient(90deg, #14161C 0%, #E3E1DC 18%);
        margin-bottom: 2rem;
    }

    /* ---------- Upload zone ---------- */
    div[data-testid="stFileUploader"] {
        margin-bottom: 0.4rem;
    }

    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 6px;
        border: 1.5px dashed #C7C4BC;
        background: #FFFFFF;
        padding: 2.2rem 1.5rem;
        transition: border-color 0.15s ease, background 0.15s ease;
    }
    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #5C3D8C;
        background: #FBFAFD;
    }

    /* icon injected before the built-in instructions */
    section[data-testid="stFileUploaderDropzone"] > div::before {
        content: "";
        display: block;
        width: 34px;
        height: 34px;
        margin: 0 auto 14px auto;
        background-color: #5C3D8C;
        -webkit-mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='1.6'><path d='M12 3v12m0-12 4.5 4.5M12 3 7.5 7.5'/><path d='M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3'/></svg>");
        mask-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='1.6'><path d='M12 3v12m0-12 4.5 4.5M12 3 7.5 7.5'/><path d='M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3'/></svg>");
        -webkit-mask-repeat: no-repeat;
        mask-repeat: no-repeat;
        -webkit-mask-position: center;
        mask-position: center;
    }

    /* "Drag and drop file here" */
    section[data-testid="stFileUploaderDropzone"] span {
        font-family: 'Inter', sans-serif;
        font-size: 0.98rem;
        font-weight: 600;
        color: #14161C;
    }

    /* "Limit 200MB per file..." */
    section[data-testid="stFileUploaderDropzone"] small {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.02em;
        color: #9CA0AC;
    }

    /* "Browse files" button */
    section[data-testid="stFileUploaderDropzone"] button {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 600;
        color: #14161C;
        background: #FFFFFF;
        border: 1px solid #C7C4BC;
        border-radius: 3px;
        padding: 0.45rem 1rem;
        margin-top: 4px;
        transition: border-color 0.15s ease, color 0.15s ease;
    }
    section[data-testid="stFileUploaderDropzone"] button:hover {
        border-color: #5C3D8C;
        color: #5C3D8C;
    }

    /* ---------- Primary button ---------- */
    div.stButton > button[kind="primary"] {
        background: #14161C;
        border: none;
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 600;
        padding: 0.7rem 0;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #5C3D8C;
    }

    /* ---------- Report card ---------- */
    .report {
        margin-top: 2rem;
        border: 1px solid #E3E1DC;
        border-radius: 6px;
        background: #FFFFFF;
        overflow: hidden;
    }

    .verdict {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 24px 28px;
        border-left: 4px solid;
    }
    .verdict-positive { background: #F5F1FB; border-color: #5C3D8C; }
    .verdict-negative { background: #FCF1EF; border-color: #B23A2E; }
    .verdict-indeterminate { background: #FBF6EA; border-color: #A6690A; }

    .verdict-label {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 0;
    }
    .verdict-positive .verdict-label { color: #43296B; }
    .verdict-negative .verdict-label { color: #8C2A20; }
    .verdict-indeterminate .verdict-label { color: #7A4E08; }

    .verdict-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #6B7080;
        margin-top: 4px;
    }

    .verdict-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        line-height: 1;
        text-align: right;
    }
    .verdict-positive .verdict-score { color: #5C3D8C; }
    .verdict-negative .verdict-score { color: #B23A2E; }
    .verdict-indeterminate .verdict-score { color: #A6690A; }
    .verdict-score span {
        font-size: 0.7rem;
        display: block;
        font-weight: 500;
        color: #9CA0AC;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-top: 2px;
    }

    /* ---------- Data rows ---------- */
    .data-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        border-top: 1px solid #EEEDE9;
    }
    .data-cell {
        padding: 16px 28px;
    }
    .data-cell + .data-cell {
        border-left: 1px solid #EEEDE9;
    }
    .data-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9CA0AC;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .data-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: #14161C;
    }

    /* ---------- Full-width sections ---------- */
    .field-block {
        padding: 18px 28px;
        border-top: 1px solid #EEEDE9;
    }
    .field-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9CA0AC;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .field-value {
        font-size: 0.92rem;
        color: #2B2E38;
        line-height: 1.6;
    }

    .review-flag {
        margin: 0 28px 20px 28px;
        padding: 12px 16px;
        background: #FBF6EA;
        border-left: 3px solid #A6690A;
        border-radius: 3px;
        font-size: 0.85rem;
        color: #7A4E08;
    }

    /* ---------- Technical details (custom, replaces st.json) ---------- */
    .tech-panel {
        margin-top: 1.1rem;
        border: 1px solid #E3E1DC;
        border-radius: 6px;
        background: #14161C;
        overflow: hidden;
    }
    .tech-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 20px;
        background: #1D2029;
        border-bottom: 1px solid #2A2E3A;
    }
    .tech-panel-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #8A6BB8;
        font-weight: 600;
    }
    .tech-panel-dots span {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #3A3E4A;
        margin-left: 5px;
    }
    .tech-panel-body {
        padding: 18px 20px 20px 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        line-height: 1.85;
    }
    .tech-row {
        display: flex;
        padding: 2px 0;
    }
    .tech-key {
        color: #8A6BB8;
        white-space: nowrap;
        margin-right: 0.5em;
    }
    .tech-colon {
        color: #565B69;
        margin-right: 0.6em;
    }
    .tech-string { color: #D9A566; }
    .tech-number { color: #6FA8DC; }
    .tech-bool   { color: #E06C75; }
    .tech-bracket { color: #565B69; }
    .tech-array-item {
        display: flex;
        padding-left: 1.4em;
    }
    .tech-indent {
        padding-left: 1.4em;
    }

    /* ---------- Footer ---------- */
    .disclaimer {
        text-align: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.02em;
        color: #A3A7B0;
        margin-top: 2.5rem;
        padding-top: 18px;
        border-top: 1px solid #E3E1DC;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def preprocess_image(image: Image.Image) -> Image.Image:
    """Resize the image and convert it to RGB if needed."""
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
    return image


def extract_json(raw_text: str) -> dict:
    """Safely extract JSON from the model response (strip markdown fences if present)."""
    cleaned = re.sub(r"```json\s*|```\s*", "", raw_text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Model response was not valid JSON")


def analyze_image(image: Image.Image) -> dict:
    """Send the image to Gemini and return the parsed JSON result."""
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
    )
    response = model.generate_content(
        [image, "Analyze this gram stain image."],
        generation_config={
            "max_output_tokens": 2048,
            "response_mime_type": "application/json",
        },
    )
    st.session_state["last_raw_response"] = response.text
    result = extract_json(response.text)
    result["needs_human_review"] = result.get("confidence", 0) < 60
    return result


def verdict_class(classification: str) -> str:
    return {
        "Gram Positive": "verdict-positive",
        "Gram Negative": "verdict-negative",
        "Indeterminate": "verdict-indeterminate",
    }.get(classification, "verdict-indeterminate")


def render_tech_value(value) -> str:
    """Render a single JSON value as syntax-highlighted HTML (no key)."""
    if isinstance(value, bool):
        return f'<span class="tech-bool">{str(value).lower()}</span>'
    if isinstance(value, (int, float)):
        return f'<span class="tech-number">{value}</span>'
    if isinstance(value, str):
        safe = html_lib.escape(value)
        return f'<span class="tech-string">"{safe}"</span>'
    if value is None:
        return '<span class="tech-bool">null</span>'
    return html_lib.escape(str(value))


def render_tech_panel(result: dict) -> str:
    """Build a custom-styled, syntax-highlighted JSON viewer to replace st.json()."""
    rows = []
    for key, value in result.items():
        safe_key = html_lib.escape(str(key))

        if isinstance(value, list):
            if not value:
                body = '<span class="tech-bracket">[]</span>'
            else:
                items = "".join(
                    f'<div class="tech-array-item">{render_tech_value(item)}</div>'
                    for item in value
                )
                body = (
                    f'<span class="tech-bracket">[</span>{items}'
                    f'<div class="tech-bracket">]</div>'
                )
            rows.append(
                f'<div class="tech-row"><span class="tech-key">"{safe_key}"</span>'
                f'<span class="tech-colon">:</span><div>{body}</div></div>'
            )
        else:
            rows.append(
                f'<div class="tech-row"><span class="tech-key">"{safe_key}"</span>'
                f'<span class="tech-colon">:</span>{render_tech_value(value)}</div>'
            )

    body_html = "".join(rows)
    return textwrap.dedent(f"""\
    <div class="tech-panel">
        <div class="tech-panel-header">
            <span class="tech-panel-title">Raw Model Output · JSON</span>
            <span class="tech-panel-dots"><span></span><span></span><span></span></span>
        </div>
        <div class="tech-panel-body">{body_html}</div>
    </div>
    """).strip()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    textwrap.dedent("""\
    <div class="eyebrow">Specimen Analysis · AI-Assisted</div>
    <h1 class="app-title">Bacteria Identification AI</h1>
    <p class="app-subtitle">
        Upload a gram-stained microscope image to get an automated read on
        classification, morphology, and likely organisms.
    </p>
    <div class="header-rule"></div>
    """).strip(),
    unsafe_allow_html=True,
)

if not API_KEY:
    st.error(
        "⚠️ GEMINI_API_KEY not found. Create a `.env` file and add "
        "`GEMINI_API_KEY=your-key` to it.\n\n"
        "Get a free key here: https://aistudio.google.com/apikey"
    )
    st.stop()

st.markdown(
    '<div class="field-label" style="margin-bottom:8px;">Specimen Image</div>',
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader(
    "Upload specimen image",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=uploaded_file.name, use_column_width=True)

    analyze_clicked = st.button("Analyze Specimen", type="primary", use_container_width=True)

    if analyze_clicked:
        with st.spinner("Analyzing specimen..."):
            try:
                image_processed = preprocess_image(image.copy())
                result = analyze_image(image_processed)
            except ValueError as e:
                st.error(f"❌ Could not get valid JSON from the model: {e}")
                if "last_raw_response" in st.session_state:
                    with st.expander("Show technical details"):
                        st.code(st.session_state["last_raw_response"])
                st.stop()
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")
                st.stop()

        # -----------------------------------------------------
        # Report card
        # -----------------------------------------------------
        vclass = verdict_class(result["classification"])
        organisms = ", ".join(result.get("possible_organisms", []))

        report_html = textwrap.dedent(f"""\
        <div class="report">
            <div class="verdict {vclass}">
                <div>
                    <p class="verdict-label">{result['classification']}</p>
                    <p class="verdict-sub">IMAGE QUALITY: {result['image_quality'].upper()}</p>
                </div>
                <div class="verdict-score">{result['confidence']}<span>Confidence</span></div>
            </div>
            <div class="data-row">
                <div class="data-cell">
                    <div class="data-label">Morphology</div>
                    <div class="data-value">{result['morphology']}</div>
                </div>
                <div class="data-cell">
                    <div class="data-label">Color Observed</div>
                    <div class="data-value">{result['color_observed']}</div>
                </div>
            </div>
            <div class="field-block">
                <div class="field-label">Possible Organisms</div>
                <div class="field-value">{organisms}</div>
            </div>
            <div class="field-block">
                <div class="field-label">Findings</div>
                <div class="field-value">{result['explanation']}</div>
            </div>
            <div class="field-block">
                <div class="field-label">Clinical Note</div>
                <div class="field-value">{result['clinical_note']}</div>
            </div>
        </div>
        """).strip()
        st.markdown(report_html, unsafe_allow_html=True)

        if result.get("needs_human_review"):
            st.markdown(
                textwrap.dedent("""\
                <div class="review-flag">
                    Confidence below 60% — recommend manual verification by a
                    lab technician.
                </div>
                """).strip(),
                unsafe_allow_html=True,
            )

        with st.expander("Show technical details"):
            st.markdown(render_tech_panel(result), unsafe_allow_html=True)

st.markdown(
    textwrap.dedent("""\
    <div class="disclaimer">
        AI-ASSISTED DIAGNOSTIC AID · NOT A FINAL DIAGNOSIS · CONFIRM WITH A QUALIFIED MICROBIOLOGIST
    </div>
    """).strip(),
    unsafe_allow_html=True,
)