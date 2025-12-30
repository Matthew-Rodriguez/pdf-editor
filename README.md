# PDF Editor

A Streamlit-based PDF editor for cannabis lab reports (COAs) with intelligent cannabinoid value editing, automatic formula recalculation, and natural language instruction parsing.

## Features

- **Intelligent Cannabinoid Editing**: Automatically updates related values (% ↔ mg/g conversions, formula recalculation)
- **Natural Language Parsing**: Uses GPT-4o-mini (LLM-first) with rule-based fallback for simple instructions
- **Direct Text Search**: Edit any text in the PDF, even if not detected as a title
- **Priority Cannabinoids**: THCa %, Δ9 %, Total THC %, Total Cannabinoids % always shown at top
- **Domain-Specific Operations**:
  - THCa × 0.877 + Δ9 = Total THC formula recalculation
  - Reverse-engineer totals to ensure components add up correctly
  - Fix % ↔ mg/g conversions (1% = 10 mg/g)
  - Tier-based value adjustments (low/mid/high)
- **Dynamic Filenames**: Output PDFs named `COA_{strain-name}_{date}.pdf`
- **Visual Feedback**: Green highlighting for modified fields

## Installation

```bash
pip install -r requirements.txt
```

## Setup

1. Install dependencies:
```bash
pip install streamlit pymupdf rapidfuzz python-dateutil openai
```

2. Configure OpenAI API key (optional, for LLM parsing):
   - Create `.streamlit/secrets.toml`:
   ```toml
   [openai]
   api_key = "your-api-key-here"
   ```
   - Or set environment variable: `export OPENAI_API_KEY=your-key`

## Usage

```bash
streamlit run pdf_editor.py
```

1. Upload a PDF (works best with selectable text, not scanned images)
2. Enter natural language instructions like:
   - `"change thca to 29.44%"`
   - `"randomize LOD between 10-30%"`
   - `"set Sample ID to ABC123"`
   - `"ensure THCa × 0.877 + Δ9 = Total THC"`
   - `"reverse-engineer numbers so totals add up correctly"`
3. Preview edits and download the modified PDF

## Example Instructions

- **Set values**: `"change THCa % to 25.5"`
- **Randomize**: `"randomize Total THC between 20-30%"`
- **Formulas**: `"ensure THCa × 0.877 + Δ9 = Total THC"`
- **Conversions**: `"fix % ↔ mg/g conversions"`
- **Totals**: `"reverse-engineer totals"`
- **Tier adjustments**: `"adjust values to mid tier"`
- **Chained edits**: `"change THCa to 25% and set Strain to GG4"`

## Technical Details

- **PDF Processing**: PyMuPDF (fitz) for text extraction and in-place editing
- **Parsing Strategy**: LLM-first (GPT-4o-mini) with rule-based fallback
- **Fuzzy Matching**: rapidfuzz (with difflib fallback) for title resolution
- **Text Search**: Direct text search fallback for any text in PDF

## License

See LICENSE file for details.

