#!/usr/bin/env python3
"""
PDF COA Editor - Web Version using Streamlit
Run with: streamlit run pdf_editor.py
"""

import fitz  # PyMuPDF
import streamlit as st
from datetime import datetime
import re
import os
import io
import random


class PDFCOAEditor:
    """Editor for COA PDF files with precise text replacement"""
    
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.page = self.doc[0]
        
    def find_text_instances(self, search_text, case_sensitive=False):
        """Find all instances of text and return their positions and properties"""
        instances = []
        # Always get fresh text_dict (don't cache)
        text_dict = self.page.get_text("dict")
        
        for block in text_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        if case_sensitive:
                            if search_text in text:
                                instances.append({
                                    "text": text,
                                    "bbox": span["bbox"],
                                    "font": span["font"],
                                    "size": span["size"],
                                    "flags": span["flags"],
                                    "origin": span["origin"]
                                })
                        else:
                            if search_text.lower() in text.lower():
                                instances.append({
                                    "text": text,
                                    "bbox": span["bbox"],
                                    "font": span["font"],
                                    "size": span["size"],
                                    "flags": span["flags"],
                                    "origin": span["origin"]
                                })
        return instances
    
    def replace_text_precise(self, old_text, new_text, font_name=None, font_size=None, is_bold=False, exact_match=False):
        """
        Replace text precisely by finding exact matches and replacing them
        Uses white rectangle overlay + text insertion for safe replacement
        """
        # Find text instances - try multiple search strategies
        instances = self.find_text_instances(old_text, case_sensitive=False)
        
        # If no exact match, try searching for key parts
        if not instances:
            # Try searching for just the key part (e.g., "Gelato 45" instead of "Gelato 45 (Deps)")
            key_parts = old_text.split()
            if len(key_parts) > 1:
                # Try with first few words
                search_text = " ".join(key_parts[:2])
                instances = self.find_text_instances(search_text, case_sensitive=False)
        
        if not instances:
            return False
        
        # Try to find the best matching instance
        exact_instance = None
        old_text_lower = old_text.lower().strip()
        
        for inst in instances:
            inst_text = inst["text"]  # Keep original for exact match
            inst_text_lower = inst_text.lower().strip()
            
            if exact_match:
                # For exact match, compare the actual text (case-sensitive for exact match)
                if inst_text.strip() == old_text.strip():
                    exact_instance = inst
                    break
                # Also try case-insensitive
                elif inst_text_lower == old_text_lower:
                    exact_instance = inst
                    break
            else:
                # Check if old_text is contained in instance text, or vice versa
                if old_text_lower in inst_text_lower or inst_text_lower in old_text_lower:
                    exact_instance = inst
                    break
                # Check if they share significant words
                old_words = set(old_text_lower.split())
                inst_words = set(inst_text_lower.split())
                if len(old_words.intersection(inst_words)) >= len(old_words) * 0.5:
                    exact_instance = inst
                    break
        
        if not exact_instance:
            exact_instance = instances[0]  # Fallback to first match
        
        instance = exact_instance
        bbox = instance["bbox"]
        
        # CRITICAL: Protect address area - never redact anything in address area (X >= 380, Y between 115-130)
        # Address is at X=400, Y=116.3, so protect a wide area around it
        if 115 <= bbox[1] <= 130 and (bbox[0] >= 380 or bbox[2] >= 380):
            # This match is in the address area - skip replacement to protect address
            return False
        
        # Determine font properties
        if font_name is None:
            font_name = instance["font"]
        if font_size is None:
            font_size = instance["size"]
        
        # Calculate text width more accurately
        try:
            text_width = fitz.get_text_length(new_text, fontname=font_name, fontsize=font_size)
            # Get actual width of existing text from bbox
            old_text_width = bbox[2] - bbox[0]
        except:
            # Fallback estimation
            text_width = len(new_text) * font_size * 0.6
            old_text_width = bbox[2] - bbox[0]
        
        # Normalize font name for PyMuPDF FIRST (before calculating positions)
        # PyMuPDF uses short names: "helv" for Helvetica, "hebo" for Helvetica-Bold
        font_map = {
            "helvetica": "helv",
            "helvetica-bold": "hebo",
            "helvetica-oblique": "hevo",
            "helvetica-boldoblique": "hebo"
        }
        
        normalized_font = font_name.lower()
        for key, value in font_map.items():
            if key in normalized_font:
                normalized_font = value
                break
        
        # If still not mapped, try to extract base name
        if normalized_font not in ["helv", "hebo", "hevo"]:
            if "bold" in normalized_font:
                normalized_font = "hebo"
            else:
                normalized_font = "helv"
        
        # Determine if text should be bold (use hebo font instead of helv)
        # Don't use flags parameter - PyMuPDF insert_text doesn't support it
        should_be_bold = is_bold or (instance["flags"] & 16)
        
        # Calculate proper y position (baseline)
        # PyMuPDF insert_text uses bottom-left origin
        # Use the original baseline Y position from the instance to preserve exact alignment
        y_pos = instance["bbox"][3] - 2  # Use bottom of original bbox, adjusted slightly up
        
        # Determine if this is a number (for alignment)
        is_number = bool(re.match(r'^[\d.]+%?$', new_text.strip()))
        
        # INSERT FIRST, then redact (safer - don't lose original if insertion fails)
        simple_font = "hebo" if should_be_bold else "helv"
        insertion_success = False
        
        # Calculate insertion position
        if is_number:
            # Estimate text width for right alignment
            try:
                est_width = fitz.get_text_length(new_text, fontname=simple_font, fontsize=font_size)
            except:
                est_width = len(new_text) * font_size * 0.6
            x_pos = bbox[2] - est_width - 1  # Right align
        else:
            x_pos = bbox[0]  # Left align
        
        # Try to insert new text FIRST (before redacting old text)
        # This way if insertion fails, we don't lose the original
        try:
            self.page.insert_text(
                (x_pos, y_pos),
                new_text,
                fontname=simple_font,
                fontsize=font_size
            )
            insertion_success = True
        except Exception as e1:
            # Method 2: Try insert_textbox
            try:
                if is_number:
                    text_rect = fitz.Rect(
                        bbox[2] - text_width - 2,
                        bbox[1],
                        bbox[2] - 2,
                        bbox[3]
                    )
                    align = fitz.TEXT_ALIGN_RIGHT
                else:
                    text_rect = fitz.Rect(
                        bbox[0],
                        bbox[1],
                        bbox[0] + max(text_width, old_text_width) + 5,
                        bbox[3]
                    )
                    align = fitz.TEXT_ALIGN_LEFT
                
                self.page.insert_textbox(
                    text_rect,
                    new_text,
                    fontname=simple_font,
                    fontsize=font_size,
                    align=align,
                    flags=16 if should_be_bold else 0
                )
                insertion_success = True
            except Exception as e2:
                # Method 3: Try with adjusted y position
                try:
                    y_pos_adj = bbox[1] + (bbox[3] - bbox[1]) * 0.75
                    self.page.insert_text(
                        (bbox[0], y_pos_adj),
                        new_text,
                        fontname=simple_font,
                        fontsize=font_size
                    )
                    insertion_success = True
                except Exception as e3:
                    # Method 4: Last resort
                    try:
                        self.page.insert_text(
                            (bbox[0], bbox[3] - 1),
                            new_text,
                            fontname="helv",
                            fontsize=font_size
                        )
                        insertion_success = True
                    except Exception as e4:
                        # All insertion methods failed
                        return False
        
        # Use WHITE RECTANGLE to cover old text (NOT redaction - redaction destroys content!)
        if insertion_success:
            # Calculate exact width needed
            exact_width = bbox[2] - bbox[0]  # Actual width of text
            calculated_right = bbox[0] + exact_width + 1
            
            # ABSOLUTE PROTECTION: Never touch address area (X >= 380, Y between 115-130)
            if 115 <= bbox[1] <= 130 and (bbox[0] >= 380 or bbox[2] >= 380 or calculated_right >= 380):
                return insertion_success
            
            # Limit right edge in top area to protect address
            if bbox[1] < 150 and calculated_right > 350:
                calculated_right = min(calculated_right, 350)
            
            rect = fitz.Rect(
                bbox[0] - 1,
                bbox[1] - 0.3,
                calculated_right,
                bbox[3] + 0.1
            )
            
            # SAFE: Use white rectangle instead of redaction (doesn't destroy content)
            self.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
            
            # Text was already inserted before the rectangle - no need to re-insert
        
        return insertion_success
    
    def replace_cannabinoid_value(self, cannabinoid_name, new_percent, new_mg_g=None):
        """Replace cannabinoid values in the table"""
        calc = COACalculator()
        
        # Format the percentage value
        percent_val = None
        if new_percent and new_percent.strip() and new_percent.strip().upper() != "ND":
            try:
                percent_val = float(new_percent)
                percent_str = calc.format_number(percent_val, 4)
            except:
                percent_str = "ND"
        else:
            percent_str = "ND"
        
        # Calculate mg/g if not provided
        if new_mg_g is None and percent_val is not None:
            new_mg_g = calc.percent_to_mg_g(percent_val)
            if new_mg_g is not None:
                mg_g_str = calc.format_number(new_mg_g, 3)
            else:
                mg_g_str = "ND"
        elif new_mg_g:
            mg_g_str = calc.format_number(new_mg_g, 3)
        else:
            mg_g_str = "ND"
        
        # Find the cannabinoid row and replace values
        name_instances = self.find_text_instances(cannabinoid_name)
        
        if not name_instances:
            return False
        
        name_instance = name_instances[0]
        name_y = name_instance["bbox"][1]
        
        # Find all text on the same row
        # IMPORTANT: Get fresh text_dict after each replacement to ensure we find current state
        text_dict = self.page.get_text("dict")
        row_texts = []
        
        for block in text_dict["blocks"]:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        span_y = span["bbox"][1]
                        # Use tighter tolerance (1.5 instead of 2) to be more precise
                        if abs(span_y - name_y) < 1.5:
                            row_texts.append({
                                "text": span["text"],
                                "bbox": span["bbox"],
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"]
                            })
        
        row_texts.sort(key=lambda x: x["bbox"][0])
        
        # Find percentage and mg/g columns by X position (more reliable)
        percent_col = None
        mg_g_col = None
        
        for text_info in row_texts:
            text = text_info["text"].strip()
            bbox_x = text_info["bbox"][0]
            
            # Result % column is around X=310, Result mg/g is around X=395
            # Use wider ranges to catch slight variations
            if re.match(r'^[\d.]+$', text):
                if 300 < bbox_x < 325:  # Result % column (wider range)
                    if percent_col is None:
                        percent_col = text_info
                elif 385 < bbox_x < 405:  # Result mg/g column (wider range)
                    if mg_g_col is None:
                        mg_g_col = text_info
        
        # Debug: if not found, try to find by position relative to name
        if not percent_col or not mg_g_col:
            # Fallback: find any numeric values on the row, sorted by X
            numeric_values = []
            for text_info in row_texts:
                text = text_info["text"].strip()
                if re.match(r'^[\d.]+$', text):
                    numeric_values.append(text_info)
            
            # Sort by X position
            numeric_values.sort(key=lambda x: x["bbox"][0])
            
            # Result % should be around X=310, mg/g around X=395
            # Find closest to expected positions
            if not percent_col and numeric_values:
                for val in numeric_values:
                    if 300 < val["bbox"][0] < 325:
                        percent_col = val
                        break
            
            if not mg_g_col and numeric_values:
                for val in numeric_values:
                    if 385 < val["bbox"][0] < 405:
                        mg_g_col = val
                        break
        
        # Only return True if we actually found and replaced values
        if not percent_col and not mg_g_col:
            return False
        
        # Replace percentage value - use WHITE RECTANGLE (not redaction) to avoid destroying content
        if percent_col:
            old_percent = percent_col["text"].strip()
            percent_bbox = percent_col["bbox"]
            percent_font_size = percent_col["size"]
            original_x = percent_bbox[0]
            row_y_baseline = percent_bbox[3] - 2
            # Use white rectangle instead of redaction (SAFE - doesn't destroy content)
            rect = fitz.Rect(percent_bbox[0], percent_bbox[1], percent_bbox[2], percent_bbox[3])
            self.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
            # Insert new text
            insert_x = original_x if 300 < original_x < 325 else 310.0
            self.page.insert_text((insert_x, row_y_baseline), percent_str,
                                fontname="hebo", fontsize=percent_font_size)
        
        # Replace mg/g value - use WHITE RECTANGLE (not redaction)
        if mg_g_col:
            old_mg_g = mg_g_col["text"].strip()
            mg_g_bbox = mg_g_col["bbox"]
            mg_g_font_size = mg_g_col["size"]
            original_x_mg = mg_g_bbox[0]
            row_y_baseline = mg_g_bbox[3] - 2
            # Use white rectangle instead of redaction (SAFE)
            rect = fitz.Rect(mg_g_bbox[0], mg_g_bbox[1], mg_g_bbox[2], mg_g_bbox[3])
            self.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
            # Insert new text
            insert_x_mg = original_x_mg if 385 < original_x_mg < 405 else 395.0
            self.page.insert_text((insert_x_mg, row_y_baseline), mg_g_str,
                                fontname="helv", fontsize=mg_g_font_size)
        
        return True
    
    def get_pdf_bytes(self):
        """Get PDF as bytes"""
        return self.doc.tobytes()
    
    def save(self, output_path):
        """Save the edited PDF"""
        self.doc.save(output_path)
        self.doc.close()


class COACalculator:
    """Calculate dependent values for COA"""
    
    @staticmethod
    def percent_to_mg_g(percent):
        """Convert percentage to mg/g: mg/g = % × 10"""
        if percent is None or percent == "":
            return None
        try:
            return float(percent) * 10
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def calculate_total_thc(thca_percent, delta9_percent):
        """Total THC = THCa × 0.877 + Δ9"""
        try:
            thca = float(thca_percent) if thca_percent and thca_percent != "ND" else 0
            delta9 = float(delta9_percent) if delta9_percent and delta9_percent != "ND" else 0
            return thca * 0.877 + delta9
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def calculate_total_cannabinoids(total_thc, cbc, cbg, cbn, delta8, thcv):
        """Total Cannabinoids = Total THC + minor cannabinoids"""
        try:
            total = float(total_thc) if total_thc and total_thc != "ND" else 0
            cbc_val = float(cbc) if cbc and cbc != "ND" else 0
            cbg_val = float(cbg) if cbg and cbg != "ND" else 0
            cbn_val = float(cbn) if cbn and cbn != "ND" else 0
            delta8_val = float(delta8) if delta8 and delta8 != "ND" else 0
            thcv_val = float(thcv) if thcv and thcv != "ND" else 0
            return total + cbc_val + cbg_val + cbn_val + delta8_val + thcv_val
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def generate_sample_id(strain, year=None, month=None):
        """Generate Sample ID: YYMMHW-<truncated strain code>"""
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month
        
        yy = str(year)[-2:]
        mm = f"{month:02d}"
        
        strain_code = ""
        if strain:
            clean_strain = re.sub(r'\([^)]*\)', '', strain).strip()
            match = re.match(r'([A-Za-z])(\d+)', clean_strain)
            if match:
                letter = match.group(1).upper()
                numbers = match.group(2)
                strain_code = letter + numbers[:2]
            else:
                strain_code = clean_strain[:3].upper().replace(" ", "")
        
        return f"{yy}{mm}HW-{strain_code}D"
    
    @staticmethod
    def format_number(value, decimals=4):
        """Format number with specified decimal places"""
        if value is None:
            return "ND"
        try:
            formatted = f"{float(value):.{decimals}f}"
            # For Result % values (4 decimals), always show 4 decimal places (don't strip trailing zeros)
            # For mg/g values (3 decimals), strip trailing zeros for cleaner display
            if decimals == 4:
                return formatted  # Always show 4 decimal places for Result %
            else:
                return formatted.rstrip('0').rstrip('.')  # Strip trailing zeros for other values
        except (ValueError, TypeError):
            return "ND"


# Streamlit App
st.set_page_config(page_title="COA PDF Editor", page_icon="📄", layout="wide")

st.title("📄 COA PDF Editor")
st.markdown("Edit Certificate of Analysis PDFs while preserving layout and formatting")

# Initialize session state
if 'editor' not in st.session_state:
    st.session_state.editor = None
if 'pdf_loaded' not in st.session_state:
    st.session_state.pdf_loaded = False
if 'data' not in st.session_state:
    st.session_state.data = {
        "strain": "",
        "collected_date": "",
        "received_date": "",
        "completed_date": "",
        "thca": "",
        "delta9": "",
        "cbc": "",
        "cbg": "",
        "cbn": "",
        "delta8": "",
        "thcv": ""
    }

# Always load the default PDF template
pdf_filename = "COA_2511HW-G45D_v2.pdf"

# Try different paths (works locally and on Streamlit Cloud)
possible_paths = [
    "/Users/mini45/Desktop/PDF-Editor/COA_2511HW-G45D_v2.pdf",  # Local absolute path
    pdf_filename,  # Relative path (same directory as script)
]

default_pdf = None
for path in possible_paths:
    if os.path.exists(path):
        default_pdf = path
        break

# Load PDF on first run
if not st.session_state.pdf_loaded:
    if default_pdf:
        try:
            st.session_state.editor = PDFCOAEditor(default_pdf)
            st.session_state.pdf_loaded = True
            
            # Extract values from PDF
            text = st.session_state.editor.page.get_text()
            
            # Extract strain
            strain_match = re.search(r'Strain:\s*(.+)', text)
            if strain_match:
                st.session_state.data["strain"] = strain_match.group(1).strip()
            
            # Extract dates
            collected_match = re.search(r'Collected:\s*(\d+/\d+/\d+)', text)
            if collected_match:
                st.session_state.data["collected_date"] = collected_match.group(1)
            
            received_match = re.search(r'Received:\s*(\d+/\d+/\d+)', text)
            if received_match:
                st.session_state.data["received_date"] = received_match.group(1)
            
            completed_match = re.search(r'Completed:\s*(\d+/\d+/\d+)', text)
            if completed_match:
                st.session_state.data["completed_date"] = completed_match.group(1)
            
            # Extract cannabinoids
            cannabinoid_patterns = {
                "thca": r'THCa\s+[\d.]+\s+[\d.]+\s+(\d+\.\d+|ND)',
                "delta9": r'Delta 9 THC\s+[\d.]+\s+[\d.]+\s+(\d+\.\d+|ND)',
                "cbc": r'CBC\s+[\d.]+\s+[\d.]+\s+(\d+\.\d+|ND)',
                "cbg": r'CBG\s+[\d.]+\s+[\d.]+\s+(\d+\.\d+|ND)',
                "cbn": r'CBN\s+[\d.]+\s+[\d.]+\s+(\d+\.\d+|ND)',
                "delta8": r'Delta 8 THC\s+[\d.]+\s+[\d.]+\s+(\d+\.\d+|ND)',
                "thcv": r'THCV\s+[\d.]+\s+[\d.]+\s+(\d+\.\d+|ND)'
            }
            
            for key, pattern in cannabinoid_patterns.items():
                match = re.search(pattern, text)
                if match:
                    val = match.group(1)
                    if val != "ND":
                        st.session_state.data[key] = val
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            st.session_state.pdf_loaded = False
    else:
        st.error(f"Default PDF template not found: {pdf_filename}")
        st.info(f"Tried paths: {', '.join(possible_paths)}")
        st.session_state.pdf_loaded = False

# Sidebar info
with st.sidebar:
    st.header("📁 PDF Template")
    if st.session_state.pdf_loaded:
        st.success("✅ Template loaded: COA_2511HW-G45D_v2.pdf")
    else:
        st.error("❌ Template not loaded")

# Main content
if not st.session_state.pdf_loaded:
    st.error("❌ Unable to load PDF template. Please check that the file exists.")
    st.info(f"Expected file: `COA_2511HW-G45D_v2.pdf` (in the same directory as the app)")
else:
    calc = COACalculator()
    
    # Basic Information
    st.header("📋 Basic Information")
    col1, col2 = st.columns(2)
    
    with col1:
        strain = st.text_input("Strain", value=st.session_state.data["strain"], key="strain_input")
        st.session_state.data["strain"] = strain
        
        completed_date = st.text_input("Completed Date", value=st.session_state.data["completed_date"],
                                     placeholder="MM/DD/YYYY", key="completed_input")
        st.session_state.data["completed_date"] = completed_date
        
        # Auto-calculate Collected and Received as Completed - 1 day
        if completed_date:
            try:
                from datetime import datetime, timedelta
                completed_dt = datetime.strptime(completed_date, "%m/%d/%Y")
                one_day_before = completed_dt - timedelta(days=1)
                auto_collected = one_day_before.strftime("%m/%d/%Y")
                auto_received = one_day_before.strftime("%m/%d/%Y")
                st.session_state.data["collected_date"] = auto_collected
                st.session_state.data["received_date"] = auto_received
            except ValueError:
                st.session_state.data["collected_date"] = ""
                st.session_state.data["received_date"] = ""
    
    with col2:
        # Auto-generate Sample ID
        if strain:
            sample_id = calc.generate_sample_id(strain)
            st.text_input("Sample ID (auto-generated)", value=sample_id, disabled=True, key="sample_id")
        else:
            st.text_input("Sample ID (auto-generated)", value="", disabled=True, key="sample_id")
    
    # Cannabinoids
    st.header("🧪 Cannabinoids (%)")
    col1, col2, col3 = st.columns(3)
    
    cannabinoids = [
        ("THCa", "thca"),
        ("Δ9 THC", "delta9"),
        ("CBC", "cbc"),
        ("CBG", "cbg"),
        ("CBN", "cbn"),
        ("Δ8 THC", "delta8"),
        ("THCV", "thcv")
    ]
    
    cannabinoid_inputs = {}
    for i, (label, key) in enumerate(cannabinoids):
        col = col1 if i % 3 == 0 else (col2 if i % 3 == 1 else col3)
        with col:
            value = st.text_input(label, value=st.session_state.data[key], key=f"cannabinoid_{key}")
            st.session_state.data[key] = value
            cannabinoid_inputs[key] = value
    
    # Calculated Values
    st.header("📊 Calculated Values")
    calc_col1, calc_col2 = st.columns(2)
    
    total_thc = calc.calculate_total_thc(st.session_state.data["thca"], st.session_state.data["delta9"])
    total_cannabinoids = calc.calculate_total_cannabinoids(
        total_thc,
        st.session_state.data["cbc"],
        st.session_state.data["cbg"],
        st.session_state.data["cbn"],
        st.session_state.data["delta8"],
        st.session_state.data["thcv"]
    )
    
    with calc_col1:
        if total_thc is not None:
            st.metric("Total THC (%)", f"{calc.format_number(total_thc, 4)}%")
        else:
            st.metric("Total THC (%)", "N/A")
    
    with calc_col2:
        if total_cannabinoids is not None:
            st.metric("Total Cannabinoids (%)", f"{calc.format_number(total_cannabinoids, 4)}%")
        else:
            st.metric("Total Cannabinoids (%)", "N/A")
    
    # Generate PDF Button
    st.markdown("---")
    
    if st.button("🚀 Generate PDF", type="primary", use_container_width=True):
        if not st.session_state.editor:
            st.error("Please load a PDF template first")
        else:
            try:
                with st.spinner("Generating PDF..."):
                    # Create a copy of the editor for editing
                    editor_copy = PDFCOAEditor(st.session_state.editor.pdf_path)
                    
                    # Get current text from PDF to find what to replace
                    current_text = editor_copy.page.get_text()
                    
                    # Store original Collected/Received positions BEFORE any replacements
                    collected_bbox = None
                    collected_font_size = None
                    collected_y_baseline = None
                    received_bbox = None
                    received_font_size = None
                    received_y_baseline = None
                    
                    collected_instances_orig = editor_copy.find_text_instances("Collected:")
                    if collected_instances_orig:
                        collected_bbox = collected_instances_orig[0]['bbox']
                        collected_font_size = collected_instances_orig[0]['size']
                        # Calculate baseline to match original position: bbox[1] + font_size * 0.75 works well
                        collected_y_baseline = collected_bbox[1] + collected_font_size * 0.75
                    
                    received_instances_orig = editor_copy.find_text_instances("Received:")
                    if received_instances_orig:
                        received_bbox = received_instances_orig[0]['bbox']
                        received_font_size = received_instances_orig[0]['size']
                        # Calculate baseline to match original position: bbox[1] + font_size * 0.75 works well
                        received_y_baseline = received_bbox[1] + received_font_size * 0.75
                    
                    # Auto-calculate Collected and Received from Completed if not set
                    if st.session_state.data["completed_date"] and not st.session_state.data.get("collected_date"):
                        try:
                            from datetime import datetime, timedelta
                            completed_dt = datetime.strptime(st.session_state.data["completed_date"], "%m/%d/%Y")
                            one_day_before = completed_dt - timedelta(days=1)
                            st.session_state.data["collected_date"] = one_day_before.strftime("%m/%d/%Y")
                            st.session_state.data["received_date"] = one_day_before.strftime("%m/%d/%Y")
                        except:
                            pass
                    
                    # IMPORTANT: Replace dates FIRST (before strain) to avoid strain redaction covering them
                    # Dates first, then strain and Sample ID
                    # Use replace_text_precise (same method as Completed which works)
                    current_text = editor_copy.page.get_text()
                    
                    # Replace Collected date - use stored original position (dates handled at end, skip here)
                    # Dates will be replaced at the end (line ~1147) using stored positions
                    pass
                    
                    # Get fresh text
                    current_text = editor_copy.page.get_text()
                    
                    # OLD DATE CODE REMOVED - dates are now handled at the end of the flow (around line 1285)
                    # This prevents address overlap issues - all dates use draw_rect instead of apply_redactions()
                    
                    # Replace Strain - use WHITE RECTANGLE (not redaction) to avoid destroying content
                    if st.session_state.data.get("strain"):
                        current_text = editor_copy.page.get_text()
                        strain_match = re.search(r'Strain:\s*(.+)', current_text)
                        if strain_match:
                            current_strain = strain_match.group(1).strip()
                            strain_instances = editor_copy.find_text_instances("Strain:")
                            if strain_instances:
                                bbox = strain_instances[0]['bbox']
                                font_size = strain_instances[0]['size']
                                strain_value_instances = editor_copy.find_text_instances(current_strain)
                                if strain_value_instances:
                                    # Cover FULL original strain value + buffer
                                    strain_value_right = strain_value_instances[0]['bbox'][2]
                                    # Ensure we cover at least 145px from start to handle long names like "Gelato 45 (Deps)"
                                    max_right = max(strain_value_right + 5, bbox[0] + 105)
                                    max_right = min(max_right, 250)  # But don't go past X=250
                                else:
                                    # Fallback: use generous width for longer strain names
                                    max_right = min(bbox[0] + 105, 250)
                                # Use WHITE RECTANGLE instead of redaction (SAFE)
                                rect = fitz.Rect(bbox[0], bbox[1], max_right, bbox[3])
                                editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                                # Use ORIGINAL baseline position for exact alignment
                                editor_copy.page.insert_text((bbox[0], bbox[3]-2), 
                                                           f"Strain: {st.session_state.data['strain']}", 
                                                           fontname="helv", fontsize=font_size)
                        
                        # Also replace strain name in title area (around Y=88, X=40)
                        text_dict_temp = editor_copy.page.get_text('dict')
                        title_replaced = False
                        for block in text_dict_temp['blocks']:
                            if block['type'] == 0:
                                for line in block['lines']:
                                    for span in line['spans']:
                                        bbox = span['bbox']
                                        text_span = span['text'].strip()
                                        # Check if this is the title area strain name (around Y=88)
                                        if 85 < bbox[1] < 95 and 35 < bbox[0] < 200:
                                            if text_span and not text_span.startswith("Strain:") and len(text_span) > 3:
                                                # Cover FULL original title width + small buffer
                                                max_right = min(bbox[2] + 5, 250)
                                                rect = fitz.Rect(bbox[0], bbox[1], max_right, bbox[3])
                                                editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                                                title_font_size = span.get('size', 12.0)
                                                # Use ORIGINAL baseline position for exact alignment
                                                title_y_baseline = bbox[3] - 2
                                                editor_copy.page.insert_text((bbox[0], title_y_baseline), 
                                                                           st.session_state.data['strain'], 
                                                                           fontname="hebo", fontsize=title_font_size)
                                                title_replaced = True
                                                break
                                    if title_replaced:
                                        break
                                if title_replaced:
                                    break
                            if title_replaced:
                                break
                    
                    # Sample ID (after strain) - use REDACTION (Sample ID is isolated, safe to redact)
                    sample_id = calc.generate_sample_id(st.session_state.data["strain"]) if st.session_state.data["strain"] else "2511HW-G45D"
                    # Get current text BEFORE any replacements to find the full "Sample ID: [value]" text
                    current_text = editor_copy.page.get_text()
                    sample_id_match = re.search(r'Sample ID:\s*([A-Z0-9-]+)', current_text)
                    if sample_id_match:
                        current_sample_id = sample_id_match.group(1)
                        # Find the FULL "Sample ID: [value]" text instance (this gives us the complete bbox)
                        full_sample_id_instances = editor_copy.find_text_instances(f"Sample ID: {current_sample_id}")
                        if full_sample_id_instances:
                            # Use the FULL bbox of the complete text
                            full_bbox = full_sample_id_instances[0]['bbox']
                            sample_id_font_size = full_sample_id_instances[0]['size']
                            # Use REDACTION to actually remove old text (Sample ID is isolated, safe)
                            # Add 5px buffer to right edge to prevent any remnants
                            max_right = min(full_bbox[2] + 5, 250)
                            rect = fitz.Rect(full_bbox[0], full_bbox[1], max_right, full_bbox[3])
                            editor_copy.page.add_redact_annot(rect)
                            editor_copy.page.apply_redactions()
                            # Insert at EXACT original baseline position
                            editor_copy.page.insert_text((full_bbox[0], full_bbox[3] - 2), 
                                                       f"Sample ID: {sample_id}", 
                                                       fontname="helv", fontsize=sample_id_font_size)
                        else:
                            # Fallback: find just "Sample ID:" label
                            sample_id_instances = editor_copy.find_text_instances("Sample ID:")
                            if sample_id_instances:
                                sample_id_bbox = sample_id_instances[0]['bbox']
                                sample_id_font_size = sample_id_instances[0]['size']
                                # Estimate width for longer sample IDs
                                estimated_width = 80 + len(current_sample_id) * 5
                                max_right = min(sample_id_bbox[0] + estimated_width + 5, 250)
                                rect = fitz.Rect(sample_id_bbox[0], sample_id_bbox[1], max_right, sample_id_bbox[3])
                                editor_copy.page.add_redact_annot(rect)
                                editor_copy.page.apply_redactions()
                                editor_copy.page.insert_text((sample_id_bbox[0], sample_id_bbox[3] - 2), 
                                                           f"Sample ID: {sample_id}", 
                                                           fontname="helv", fontsize=sample_id_font_size)
                    
                    # Replace cannabinoids - IMPORTANT: Replace others FIRST, then THCa/THCV AFTER Total THC
                    # This ensures THCa/THCV aren't overwritten by Total THC replacement
                    cannabinoid_replacements_others = [
                        ("Delta 9 THC", st.session_state.data["delta9"]),
                        ("CBC", st.session_state.data["cbc"]),
                        ("CBG", st.session_state.data["cbg"]),
                        ("CBN", st.session_state.data["cbn"]),
                        ("Delta 8 THC", st.session_state.data["delta8"]),
                    ]
                    
                    cannabinoid_replacements_thca_thcv = [
                        ("THCa", st.session_state.data["thca"]),
                        ("THCV", st.session_state.data["thcv"])
                    ]
                    
                    replacement_results = []
                    # Replace others first
                    for name, value in cannabinoid_replacements_others:
                        if value and value.strip():
                            result = editor_copy.replace_cannabinoid_value(name, value)
                            replacement_results.append((name, result))
                    
                    # Show warnings for failed replacements
                    failed_replacements = [name for name, result in replacement_results if not result]
                    if failed_replacements:
                        st.warning(f"⚠️ Some cannabinoid values could not be replaced: {', '.join(failed_replacements)}")
                    
                    # Calculate and replace Total THC - find current value in table row
                    # Get fresh text after cannabinoid replacements (but before THCa/THCV)
                    current_text = editor_copy.page.get_text()
                    if total_thc is not None:
                        total_thc_str = calc.format_number(total_thc, 4)
                        # Find Total THC value in the cannabinoids table row (not the summary)
                        # Pattern: "Total THC" followed by LOD, LOQ, then percentage
                        total_thc_match = re.search(r'Total THC\s+0\.250\s+0\.500\s+(\d+\.\d+)', current_text)
                        if total_thc_match:
                            current_total_thc = total_thc_match.group(1)
                            # Find the exact instance to get bbox and preserve left edge alignment
                            total_thc_instances = editor_copy.find_text_instances(current_total_thc)
                            if total_thc_instances:
                                # Find the instance in the table (around Y=533)
                                for inst in total_thc_instances:
                                    if 532 < inst['bbox'][1] < 536:
                                        bbox = inst['bbox']
                                        font_size = inst['size']
                                        original_left_edge = bbox[0]
                                        original_y_baseline = bbox[3] - 2
                                        # Use WHITE RECTANGLE instead of redaction (SAFE)
                                        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                        editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                                        x_pos = original_left_edge
                                        y_pos = original_y_baseline
                                        editor_copy.page.insert_text((x_pos, y_pos), total_thc_str, 
                                                                   fontname="hebo", fontsize=font_size)
                                        break
                                else:
                                    # Fallback to replace_text_precise if instance not found
                                    editor_copy.replace_text_precise(current_total_thc, total_thc_str, is_bold=True, exact_match=True)
                            else:
                                editor_copy.replace_text_precise(current_total_thc, total_thc_str, is_bold=True, exact_match=True)
                        else:
                            # Fallback: try summary section
                            total_thc_match2 = re.search(r'21\.\d+%', current_text)
                            if total_thc_match2:
                                editor_copy.replace_text_precise("21.2000", total_thc_str, is_bold=True, exact_match=True)
                            else:
                                editor_copy.replace_text_precise("21.2000", total_thc_str, is_bold=True, exact_match=True)
                        
                        # Update mg/g AFTER updating the % value (get fresh text)
                        current_text = editor_copy.page.get_text()
                        total_thc_mg_g = calc.percent_to_mg_g(total_thc)
                        if total_thc_mg_g is not None:
                            mg_g_str = calc.format_number(total_thc_mg_g, 3)
                            # Find mg/g value directly by position (around Y=533, X=395) - values are on separate lines
                            text_dict_temp = editor_copy.page.get_text('dict')
                            found_mg_g = False
                            for block in text_dict_temp['blocks']:
                                if block['type'] == 0:
                                    for line in block['lines']:
                                        for span in line['spans']:
                                            bbox = span['bbox']
                                            text_span = span['text'].strip()
                                            # Check if this is the mg/g value at Total THC row position
                                            if 532 < bbox[1] < 536 and 390 < bbox[0] < 400:
                                                if re.match(r'^\d+\.\d+$', text_span):
                                                    # Use WHITE RECTANGLE instead of redaction (SAFE)
                                                    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                                    editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                                                    font_size = span.get('size', 9.0)
                                                    editor_copy.page.insert_text((bbox[0], bbox[3]-2), mg_g_str, 
                                                                               fontname="helv", fontsize=font_size)
                                                    found_mg_g = True
                                                    break
                                        if found_mg_g:
                                            break
                                    if found_mg_g:
                                        break
                            # Fallback: try to find by text search
                            if not found_mg_g:
                                mg_g_instances = editor_copy.find_text_instances("212.000")
                                if not mg_g_instances:
                                    mg_g_instances = editor_copy.find_text_instances("212")
                                for inst in mg_g_instances:
                                    if 532 < inst['bbox'][1] < 536 and 390 < inst['bbox'][0] < 400:
                                        bbox = inst['bbox']
                                        font_size = inst['size']
                                        # Use WHITE RECTANGLE instead of redaction (SAFE)
                                        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                        editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                                        editor_copy.page.insert_text((bbox[0], bbox[3]-2), mg_g_str, 
                                                                   fontname="helv", fontsize=font_size)
                                        found_mg_g = True
                                        break
                    
                    # Calculate and replace Total Cannabinoids - use direct redact+insert to preserve exact X position
                    if total_cannabinoids is not None:
                        total_cannabinoids_str = calc.format_number(total_cannabinoids, 4)
                        # Find Total Cannabinoids value in table row by position (around Y=569, X=310)
                        text_dict_temp = editor_copy.page.get_text('dict')
                        found_total_cannab_percent = False
                        for block in text_dict_temp['blocks']:
                            if block['type'] == 0:
                                for line in block['lines']:
                                    for span in line['spans']:
                                        bbox = span['bbox']
                                        text_span = span['text'].strip()
                                        # Check if this is the percentage value at Total Cannabinoids row position
                                        if 568 < bbox[1] < 572 and 305 < bbox[0] < 320:
                                            if re.match(r'^\d+\.\d+$', text_span):
                                                original_x = bbox[0]
                                                row_y_baseline = bbox[3] - 2
                                                # Use WHITE RECTANGLE instead of redaction (SAFE)
                                                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                                editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                                                font_size = span.get('size', 9.0)
                                                editor_copy.page.insert_text((original_x, row_y_baseline), total_cannabinoids_str, 
                                                                           fontname="hebo", fontsize=font_size)
                                                found_total_cannab_percent = True
                                                break
                                    if found_total_cannab_percent:
                                        break
                                if found_total_cannab_percent:
                                    break
                        
                        # Update mg/g AFTER updating the % value (get fresh text)
                        current_text = editor_copy.page.get_text()
                        total_cannabinoids_mg_g = calc.percent_to_mg_g(total_cannabinoids)
                        if total_cannabinoids_mg_g is not None:
                            mg_g_str = calc.format_number(total_cannabinoids_mg_g, 3)
                            # Find mg/g value for Total Cannabinoids in table row (around Y=569, X=395)
                            text_dict_temp = editor_copy.page.get_text('dict')
                            found_total_cannab_mg_g = False
                            for block in text_dict_temp['blocks']:
                                if block['type'] == 0:
                                    for line in block['lines']:
                                        for span in line['spans']:
                                            bbox = span['bbox']
                                            text_span = span['text'].strip()
                                            # Check if this is the mg/g value at Total Cannabinoids row position
                                            if 568 < bbox[1] < 572 and 390 < bbox[0] < 400:
                                                if re.match(r'^\d+\.\d+$', text_span):
                                                    original_x = bbox[0]
                                                    row_y_baseline = bbox[3] - 2
                                                    # Use WHITE RECTANGLE instead of redaction (SAFE)
                                                    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                                    editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                                                    font_size = span.get('size', 9.0)
                                                    editor_copy.page.insert_text((original_x, row_y_baseline), mg_g_str, 
                                                                               fontname="helv", fontsize=font_size)
                                                    found_total_cannab_mg_g = True
                                                    break
                                        if found_total_cannab_mg_g:
                                            break
                                    if found_total_cannab_mg_g:
                                        break
                    
                    # CRITICAL: Update summary boxes to match table values (Correlation 1: Summary % = Table %)
                    # Get fresh text after all table updates (but before THCa/THCV)
                    current_text = editor_copy.page.get_text()
                    
                    # Update Total THC summary box (around Y=258, X=97)
                    if total_thc is not None:
                        total_thc_str = calc.format_number(total_thc, 4)
                        text_dict_temp = editor_copy.page.get_text('dict')
                        for block in text_dict_temp['blocks']:
                            if block['type'] == 0:
                                for line in block['lines']:
                                    for span in line['spans']:
                                        bbox = span['bbox']
                                        text_span = span['text'].strip()
                                        if 255 < bbox[1] < 265 and 90 < bbox[0] < 110:
                                            if re.match(r'^\d+\.\d+%?$', text_span):
                                                # Use WHITE RECTANGLE instead of redaction (SAFE)
                                                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                                editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                                                font_size = span.get('size', 9.0)
                                                editor_copy.page.insert_text((bbox[0], bbox[3]-2), f"{total_thc_str}%", 
                                                                           fontname="hebo", fontsize=font_size)
                                                break
                                    else:
                                        continue
                                    break
                                else:
                                    continue
                                break
                    
                    # Update Total Cannabinoids summary box (around Y=258, X=452)
                    if total_cannabinoids is not None:
                        total_cannabinoids_str = calc.format_number(total_cannabinoids, 4)
                        text_dict_temp = editor_copy.page.get_text('dict')
                        for block in text_dict_temp['blocks']:
                            if block['type'] == 0:
                                for line in block['lines']:
                                    for span in line['spans']:
                                        bbox = span['bbox']
                                        text_span = span['text'].strip()
                                        if 255 < bbox[1] < 265 and 440 < bbox[0] < 460:
                                            if re.match(r'^\d+\.\d+%?$', text_span):
                                                # Use WHITE RECTANGLE instead of redaction (SAFE)
                                                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                                                editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                                                font_size = span.get('size', 9.0)
                                                editor_copy.page.insert_text((bbox[0], bbox[3]-2), f"{total_cannabinoids_str}%", 
                                                                           fontname="hebo", fontsize=font_size)
                                                break
                                    else:
                                        continue
                                    break
                                else:
                                    continue
                                break
                    
                    # CRITICAL: Replace THCa and THCV ABSOLUTELY LAST (after all summary box updates)
                    # This ensures they're not overwritten by any other operations
                    # Get fresh text before replacing THCa/THCV
                    current_text = editor_copy.page.get_text()
                    cannabinoid_replacements_thca_thcv = [
                        ("THCa", st.session_state.data["thca"]),
                        ("THCV", st.session_state.data["thcv"])
                    ]
                    for name, value in cannabinoid_replacements_thca_thcv:
                        if value and value.strip():
                            result = editor_copy.replace_cannabinoid_value(name, value)
                            replacement_results.append((name, result))
                    
                    # Show updated warnings
                    failed_replacements = [name for name, result in replacement_results if not result]
                    if failed_replacements:
                        st.warning(f"⚠️ Some cannabinoid values could not be replaced: {', '.join(failed_replacements)}")
                    
                    # DATES ABSOLUTELY LAST - right before saving to ensure they're not covered
                    # Use stored original positions (more reliable after other replacements)
                    
                    # Replace Collected date - use WHITE RECTANGLE (not redaction)
                    # CRITICAL: Insert at EXACT SAME position as original to maintain alignment
                    # Original uses 1 space after colon: "Collected: "
                    if st.session_state.data.get("collected_date") and collected_bbox:
                        # Cover FULL original text width (bbox[2] + 5) but protect address at X=395
                        max_right = min(collected_bbox[2] + 5, 395)
                        rect = fitz.Rect(collected_bbox[0], collected_bbox[1], max_right, collected_bbox[3])
                        editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                        # Use ORIGINAL Y position (bbox[3] - 2 = baseline) to maintain exact alignment
                        collected_y = collected_bbox[3] - 2  # Original baseline position
                        editor_copy.page.insert_text((collected_bbox[0], collected_y), 
                                                   f"Collected: {st.session_state.data['collected_date']}", 
                                                   fontname="helv", fontsize=collected_font_size)
                    
                    # Replace Received date - use WHITE RECTANGLE (not redaction)
                    # CRITICAL: Insert at EXACT SAME position as original to maintain alignment
                    # Original uses 2 spaces after colon: "Received:  "
                    if st.session_state.data.get("received_date") and received_bbox:
                        # Cover FULL original text width (bbox[2] + 5) but protect address at X=395
                        max_right = min(received_bbox[2] + 5, 395)
                        rect = fitz.Rect(received_bbox[0], received_bbox[1], max_right, received_bbox[3])
                        editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                        # Use ORIGINAL Y position (bbox[3] - 2 = baseline) to maintain exact alignment
                        received_y = received_bbox[3] - 2  # Original baseline position
                        editor_copy.page.insert_text((received_bbox[0], received_y), 
                                                   f"Received:  {st.session_state.data['received_date']}", 
                                                   fontname="helv", fontsize=received_font_size)
                    
                    # Replace Completed date - use WHITE RECTANGLE (not redaction)
                    # CRITICAL: Insert at EXACT SAME position as original to maintain alignment
                    # Original uses 1 space after colon: "Completed: "
                    completed_instances = editor_copy.find_text_instances("Completed:")
                    if st.session_state.data.get("completed_date") and completed_instances:
                        completed_bbox = completed_instances[0]['bbox']
                        completed_font_size = completed_instances[0]['size']
                        # Cover FULL original text width (bbox[2] + 5) but protect address at X=395
                        max_right = min(completed_bbox[2] + 5, 395)
                        # Use WHITE RECTANGLE instead of redaction (SAFE)
                        rect = fitz.Rect(completed_bbox[0], completed_bbox[1], max_right, completed_bbox[3])
                        editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                        # Use ORIGINAL Y position (bbox[3] - 2 = baseline) to maintain exact alignment
                        completed_y = completed_bbox[3] - 2  # Original baseline position
                        completed_text = f"Completed: {st.session_state.data['completed_date']}"
                        editor_copy.page.insert_text((completed_bbox[0], completed_y), 
                                                   completed_text, 
                                                   fontname="helv", fontsize=completed_font_size)
                    
                    # Replace "Matrix: Plant" - use WHITE RECTANGLE (not redaction)
                    # CRITICAL: Insert at EXACT SAME position as original to maintain alignment
                    matrix_instances = editor_copy.find_text_instances("Matrix:")
                    if matrix_instances:
                        matrix_bbox = matrix_instances[0]['bbox']
                        matrix_font_size = matrix_instances[0]['size']
                        current_text = editor_copy.page.get_text()
                        matrix_match = re.search(r'Matrix:\s*(.+)', current_text)
                        if matrix_match:
                            matrix_value = matrix_match.group(1).strip()
                            matrix_value_instances = editor_copy.find_text_instances(matrix_value)
                            if matrix_value_instances:
                                full_right = max(matrix_bbox[2], matrix_value_instances[0]['bbox'][2])
                            else:
                                full_right = matrix_bbox[2]
                            # Cover FULL original text width (full_right + 5) but stay in left column
                            max_matrix_right = min(full_right + 5, 250)
                            # Use WHITE RECTANGLE instead of redaction (SAFE)
                            rect = fitz.Rect(matrix_bbox[0], matrix_bbox[1], max_matrix_right, matrix_bbox[3])
                            editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                            # Use ORIGINAL Y position (bbox[3] - 2 = baseline) to maintain exact alignment
                            matrix_y = matrix_bbox[3] - 2  # Original baseline position
                            editor_copy.page.insert_text((matrix_bbox[0], matrix_y), 
                                                       f"Matrix: {matrix_value}", 
                                                       fontname="helv", fontsize=matrix_font_size)
                    
                    # Randomize moisture percentage (14.5% -> 13.8% to 14.6%)
                    moisture_percent = round(random.uniform(13.8, 14.6), 1)
                    moisture_text = f"{moisture_percent}% - Complete"
                    # Find "14.5% - Complete" text (around Y=217, X=320)
                    text_dict = editor_copy.page.get_text('dict')
                    for block in text_dict['blocks']:
                        if block['type'] == 0:
                            for line in block['lines']:
                                for span in line['spans']:
                                    if '14.5%' in span['text'] or '14.' in span['text'] and '% - Complete' in span['text']:
                                        bbox = span['bbox']
                                        # Check if it's in the Result column (around X=320, Y=217)
                                        if 315 < bbox[0] < 325 and 210 < bbox[1] < 225:
                                            font_size = span.get('size', 9.0)
                                            # Use WHITE RECTANGLE to cover old text
                                            rect = fitz.Rect(bbox[0], bbox[1], bbox[2] + 10, bbox[3])
                                            editor_copy.page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0)
                                            # Insert at original baseline
                                            editor_copy.page.insert_text((bbox[0], bbox[3] - 2), 
                                                                       moisture_text, 
                                                                       fontname="helv", fontsize=font_size)
                                            break
                    
                    # Replace Date Tested dates with Completed date
                    if st.session_state.data.get("completed_date"):
                        completed_date = st.session_state.data['completed_date']
                        # Find all dates in the Date Tested column (around X=200, Y=190-220)
                        text_dict = editor_copy.page.get_text('dict')
                        date_instances = []
                        
                        # First, collect all date instances in the Date Tested column
                        for block in text_dict['blocks']:
                            if block['type'] == 0:
                                for line in block['lines']:
                                    for span in line['spans']:
                                        bbox = span['bbox']
                                        text_span = span['text'].strip()
                                        # Check if it's a date in the Date Tested column (X=200, Y=190-220)
                                        if (195 < bbox[0] < 210 and 190 < bbox[1] < 225 and
                                            '/' in text_span and len(text_span) == 10):
                                            date_instances.append({
                                                'bbox': bbox,
                                                'text': text_span,
                                                'size': span.get('size', 9.0),
                                                'y': bbox[1]
                                            })
                        
                        # Sort by Y position and take the first 3 (should be the Date Tested dates)
                        date_instances.sort(key=lambda x: x['y'])
                        date_instances = date_instances[:3]  # Take first 3 dates
                        
                        # Replace each date - use REDACTION (dates are isolated in table, safe to redact)
                        for date_inst in date_instances:
                            bbox = date_inst['bbox']
                            font_size = date_inst['size']
                            # Use REDACTION to actually remove old date - ensure full coverage
                            rect = fitz.Rect(bbox[0], bbox[1], bbox[2] + 10, bbox[3])
                            editor_copy.page.add_redact_annot(rect)
                            editor_copy.page.apply_redactions()
                            # Insert at original baseline
                            editor_copy.page.insert_text((bbox[0], bbox[3] - 2), 
                                                       completed_date, 
                                                       fontname="helv", fontsize=font_size)
                    
                    # Validate PDF before saving
                    try:
                        # Get PDF bytes
                        pdf_bytes = editor_copy.get_pdf_bytes()
                        
                        # Validate PDF by trying to reopen it
                        import io
                        test_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        test_doc.close()
                        
                        editor_copy.doc.close()
                        
                        # Generate filename
                        filename = f"COA_{sample_id}.pdf"
                        
                        st.success("✅ PDF generated successfully!")
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as pdf_error:
                        editor_copy.doc.close()
                        st.error(f"❌ PDF validation failed: {str(pdf_error)}")
                        st.info("The PDF may have been corrupted during editing. Please try again or check the template PDF.")
                
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

