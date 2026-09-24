# 🔬 Bacteria Identification AI

An AI-assisted diagnostic tool for gram-stain microscopy analysis. Upload a gram-stained specimen image and receive a structured, automated read on Gram classification, cell morphology, likely organisms, and clinical relevance — with built-in confidence scoring and human-review flagging for low-confidence results.

---

## Overview

Gram staining is a foundational technique in clinical microbiology used to classify bacteria based on their cell wall properties. Manual interpretation under a microscope requires trained expertise and can be time-consuming in high-volume lab settings. This tool uses a multimodal vision-language model to provide a fast, structured first-pass read of gram-stain images, intended as a **diagnostic aid — not a replacement for expert review.**

## Features

- 🧫 **Gram Classification** — Positive / Negative / Indeterminate, with a confidence score (0–100)
- 🔍 **Morphology Detection** — Cocci, Bacilli, Diplococci, Spirilli, Mixed, or Unclear
- 🦠 **Possible Organisms** — Model-suggested candidate organisms based on visual features
- 📋 **Clinical Note** — One-line clinical relevance summary for quick reference
- ⚠️ **Human Review Flagging** — Automatically flags results with confidence below 60% for manual verification by a lab technician
- 🖥️ **Structured Output Panel** — Custom syntax-highlighted JSON viewer for inspecting the full raw model response
- 🎨 **Clinical-Grade UI** — Interface styling grounded in actual gram-stain reagent colors (crystal violet / safranin) for an intuitive, lab-instrument feel

## Tech Stack

- **Frontend**: Streamlit + custom CSS
- **Vision Model**: Google Gemini (multimodal, image + text input) via `google-generativeai`
- **Image Handling**: Pillow (PIL) for preprocessing and resizing
- **Output Parsing**: Structured JSON extraction with fallback regex parsing for malformed model responses
- **Config**: `python-dotenv` for API key management

## How It Works

1. User uploads a gram-stain microscope image (`.png`, `.jpg`, `.jpeg`, `.webp`)
2. Image is preprocessed (RGB conversion, resized to a max dimension for model input)
3. Image + a domain-specific system prompt are sent to the Gemini vision model, instructed to return a strict JSON schema (classification, confidence, morphology, possible organisms, explanation, clinical note, image quality)
4. Response is safely parsed (with fallback extraction if the model wraps JSON in markdown or extra text)
5. Results render as a clinical report card; any result with confidence below 60% is automatically flagged for human review
6. Full raw JSON output is available in an expandable technical panel for transparency

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Nimraaaaaaaa/Bacteria-identification-with-gemini3.6Flash
cd Bacteria-identification-with-gemini3.6Flash
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey), then create a `.env` file:
```
GEMINI_API_KEY=your-key-here
```

### 4. Run the app
```bash
streamlit run app.py
```

## Project Structure
```
.
├── app.py              # Full application: UI, Gemini API call, JSON parsing, report rendering
├── requirements.txt
├── .env                # (not committed) API key configuration
└── README.md
```

## Disclaimer

This tool is an **AI-assisted diagnostic aid** and is **not a substitute for diagnosis by a qualified microbiologist or clinician**. All results — especially low-confidence ones — should be verified by a trained professional before any clinical decision is made.

## License

MIT
