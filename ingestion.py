import base64
import os
import io
import re
import json
import time
from pathlib import Path

import anthropic
import fitz
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
# CLAUDE_API_KEY = ""
POPPLER_PATH   = r"C:\Program Files\poppler-25.12.0\Library\bin"
CHARTS_DIR     = "./stored_charts"
RENDER_DPI     = 200

# ── Init ──────────────────────────────────────────────────────────────────────
client   = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma    = chromadb.PersistentClient(path="./chroma_db")
text_col  = chroma.get_or_create_collection("text_chunks")
table_col = chroma.get_or_create_collection("table_chunks")
chart_col = chroma.get_or_create_collection("chart_chunks")

os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Caption patterns — works for any document type ────────────────────────────
CAPTION_PATTERNS = [
    r"^Figure\s+\d+",
    r"^Fig\.?\s*\d+",
    r"^Chart\s+\d+",
    r"^Graph\s+\d+",
    r"^Exhibit\s+\d+",
    r"^Diagram\s+\d+",
    r"^Illustration\s+\d+",
    r"^Plot\s+\d+",
    r"^Image\s+\d+",
    r"^Appendix\s+[A-Z]",
]
caption_pattern = re.compile("|".join(CAPTION_PATTERNS), re.IGNORECASE)

# ── Prompts ───────────────────────────────────────────────────────────────────
TEXT_PROMPT = """Extract all prose text from this page image exactly as written.
Do not include table content or chart descriptions.
Return only the paragraph text."""

TABLE_PROMPT = """Extract the table from this image as structured JSON.
Every row, every column header, every cell — nothing summarized.

Respond in this format:
DESCRIPTION: <one sentence about what this table shows>
JSON:
<JSON array of objects, one per row>"""

CHART_PROMPT = """Describe this chart or figure in complete detail.
Include: chart type, all axis labels, all data values, all legends, the main insight.
Return a thorough description that captures every piece of information visible."""


# ── Already processed check ───────────────────────────────────────────────────
def already_processed(document_name, page_number):
    for col in [text_col, table_col, chart_col]:
        results = col.get(where={"document_name": document_name})
        if any(f"_p{page_number}_" in id_ for id_ in results["ids"]):
            return True
    return False


# ── Image utilities ───────────────────────────────────────────────────────────
def pil_to_bytes(pil_image, max_size_kb=1500):
    """Convert PIL image to PNG bytes, compressing if needed."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    size_kb = buf.tell() / 1024

    if size_kb > max_size_kb:
        img = pil_image.copy()
        while size_kb > max_size_kb and img.width > 800:
            img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            size_kb = buf.tell() / 1024
            print(f"    Resized to {img.width}x{img.height} → {size_kb:.0f} KB")

    buf.seek(0)
    return buf.read()


def crop_region(pil_image, bbox, page_w, page_h):
    """Crop a region from PIL image using PDF top-origin coordinates."""
    scale = RENDER_DPI / 72.0
    px0 = int(bbox[0] * scale)
    py0 = int(bbox[1] * scale)
    px1 = int(bbox[2] * scale)
    py1 = int(bbox[3] * scale)

    img_w, img_h = pil_image.size
    px0 = max(0, min(px0, img_w))
    py0 = max(0, min(py0, img_h))
    px1 = max(0, min(px1, img_w))
    py1 = max(0, min(py1, img_h))

    if px1 <= px0 or py1 <= py0:
        return None

    return pil_image.crop((px0, py0, px1, py1))


# ── Claude extraction ─────────────────────────────────────────────────────────
def claude_extract(image_bytes, prompt):
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system="You are an academic document processor. Extract content from research paper pages for indexing purposes only.",
            messages=[
                {"role": "user", "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]}
            ]
        )
        time.sleep(1)
        if not response.content:
            return ""
        return response.content[0].text.strip()

    except anthropic.BadRequestError as e:
        print(f"    WARNING: Content filter blocked this page — skipping. ({e})")
        return ""


# ── Table detection ───────────────────────────────────────────────────────────
def detect_tables_robust(page):
    """Try multiple strategies, filter false positives, return real table bboxes."""
    found = {}

    for strategy in [
        {"vertical_strategy": "lines",    "horizontal_strategy": "lines"},
        {"vertical_strategy": "text",     "horizontal_strategy": "text"},
        {"vertical_strategy": "explicit", "horizontal_strategy": "explicit"},
    ]:
        try:
            for table in page.find_tables(strategy):
                bbox = tuple(round(x, 1) for x in table.bbox)
                if bbox not in found:
                    found[bbox] = table
        except Exception:
            continue

    real_tables = []
    for bbox in found:
        region_text = page.within_bbox(bbox).extract_text() or ""

        if len(region_text.strip()) < 20:
            continue

        # Filter reversed text (rotated word labels in visualizations)
        first_50    = region_text[:50].replace(" ", "")
        reversed_50 = first_50[::-1].lower()
        common      = ["the", "it", "is", "in", "this", "that", "and"]
        if any(w in reversed_50 for w in common):
            print(f"    Skipping reversed-text false table at y={bbox[1]:.0f}")
            continue

        # Filter tiny boxes
        x0, top, x1, bottom = bbox
        if (x1 - x0) < 50 or (bottom - top) < 20:
            continue

        real_tables.append(bbox)

    return real_tables


# ── Figure detection ──────────────────────────────────────────────────────────
def detect_figures_robust(fpage, blocks, page_w, page_h):
    """
    Strategy 1: caption-based (works for any labeled figure).
    Strategy 2: embedded raster images — only if no captions found.
    """
    chart_bboxes  = []
    found_regions = set()

    # Strategy 1 — caption-based
    for block in blocks:
        x0, y0, x1, y1, text, *_ = block
        text = text.strip()

        if caption_pattern.match(text):
            prev_bottom = 0
            for other in blocks:
                ox0, oy0, ox1, oy1, *_ = other
                if oy1 < y0 - 5:
                    prev_bottom = max(prev_bottom, oy1)

            figure_top    = prev_bottom + 5
            figure_bottom = y0

            if (figure_bottom - figure_top) < 100:
                figure_top = 0

            bbox = (40, figure_top, page_w - 40, figure_bottom)
            key  = (round(figure_top), round(figure_bottom))
            if key not in found_regions:
                found_regions.add(key)
                chart_bboxes.append(bbox)
                print(f"    Caption-based figure at y={figure_top:.0f}→{figure_bottom:.0f}")

    # Strategy 2 — embedded raster images (only if no captions found)
    if not chart_bboxes:
        for img in fpage.get_images(full=True):
            xref = img[0]
            for rect in fpage.get_image_rects(xref):
                x0 = rect.x0
                y0 = page_h - rect.y1
                x1 = rect.x1
                y1 = page_h - rect.y0

                # Strict size filter — real figures are large (200 PDF units)
                if (x1 - x0) < 200 or (y1 - y0) < 200:
                    continue

                key = (round(y0), round(y1))
                if key not in found_regions:
                    found_regions.add(key)
                    chart_bboxes.append((x0, y0, x1, y1))
                    print(f"    Embedded image at y={y0:.0f}→{y1:.0f}")

    return chart_bboxes


# ── Region detection entry point ──────────────────────────────────────────────
def get_regions(pdf_path, page_index):
    table_bboxes = []
    chart_bboxes = []

    with pdfplumber.open(pdf_path) as pdf:
        page   = pdf.pages[page_index]
        page_w = float(page.width)
        page_h = float(page.height)
        table_bboxes = detect_tables_robust(page)

    doc    = fitz.open(pdf_path)
    fpage  = doc[page_index]
    page_w = fpage.rect.width
    page_h = fpage.rect.height
    blocks = fpage.get_text("blocks")

    chart_bboxes = detect_figures_robust(fpage, blocks, page_w, page_h)

    doc.close()
    return table_bboxes, chart_bboxes, page_w, page_h


# ── Storage functions ─────────────────────────────────────────────────────────
def store_text(content, page_number, document_name, region_index=0):
    if not content.strip():
        return
    chunk_id  = f"{document_name}_p{page_number}_text_{region_index}"
    embedding = embedder.encode(content).tolist()
    text_col.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{
            "document_name": document_name,
            "page_number"  : page_number,
            "modality"     : "text"
        }]
    )
    print(f"    Stored text   | page {page_number} region {region_index}")


def store_table(description, json_data, page_number, document_name, region_index=0):
    if not description:
        return
    chunk_id  = f"{document_name}_p{page_number}_table_{region_index}"
    embedding = embedder.encode(description).tolist()
    table_col.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[description],
        metadatas=[{
            "document_name": document_name,
            "page_number"  : page_number,
            "modality"     : "table",
            "json_data"    : json.dumps(json_data)
        }]
    )
    print(f"    Stored table  | page {page_number} region {region_index}")


def store_chart(caption, image_bytes, page_number, document_name, region_index=0):
    if not caption:
        return
    chunk_id       = f"{document_name}_p{page_number}_chart_{region_index}"
    image_filename = f"{document_name}_p{page_number}_chart_{region_index}.png"
    image_path     = os.path.join(CHARTS_DIR, image_filename)

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    embedding = embedder.encode(caption).tolist()
    chart_col.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[caption],
        metadatas=[{
            "document_name": document_name,
            "page_number"  : page_number,
            "modality"     : "chart",
            "image_path"   : image_path
        }]
    )
    print(f"    Stored chart  | page {page_number} region {region_index} → {image_path}")


# ── Main ingestion pipeline ───────────────────────────────────────────────────
def ingest_pdf(pdf_path):
    document_name = Path(pdf_path).stem
    pdf_path      = str(pdf_path)

    print("Rendering pages...")
    page_images = convert_from_path(pdf_path, dpi=RENDER_DPI, poppler_path=POPPLER_PATH)
    print(f"Total pages: {len(page_images)}")

    for page_index, pil_page in enumerate(page_images):
        page_number = page_index + 1
        print(f"\n── Page {page_number}/{len(page_images)} ──")

        if already_processed(document_name, page_number):
            print(f"  Already processed, skipping.")
            continue

        table_bboxes, chart_bboxes, page_w, page_h = get_regions(pdf_path, page_index)
        print(f"  Tables detected : {len(table_bboxes)}")
        print(f"  Charts detected : {len(chart_bboxes)}")

        # Process tables
        for i, bbox in enumerate(table_bboxes):
            crop = crop_region(pil_page, bbox, page_w, page_h)
            if crop is None:
                continue
            image_bytes = pil_to_bytes(crop)
            raw         = claude_extract(image_bytes, TABLE_PROMPT)

            description = ""
            json_data   = []
            if "DESCRIPTION:" in raw:
                description = raw.split("DESCRIPTION:", 1)[1].split("JSON:", 1)[0].strip()
            if "JSON:" in raw:
                json_str = raw.split("JSON:", 1)[1].strip()\
                              .replace("```json", "").replace("```", "").strip()
                try:
                    json_data = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"    WARNING: JSON parse failed page {page_number}")

            store_table(description, json_data, page_number, document_name, region_index=i)

        # Process charts
        for i, bbox in enumerate(chart_bboxes):
            crop = crop_region(pil_page, bbox, page_w, page_h)
            if crop is None:
                continue
            image_bytes = pil_to_bytes(crop)
            caption     = claude_extract(image_bytes, CHART_PROMPT)
            store_chart(caption, image_bytes, page_number, document_name, region_index=i)

        # Process text — full page, Claude ignores tables/charts
        full_page_bytes = pil_to_bytes(pil_page)
        text_content    = claude_extract(full_page_bytes, TEXT_PROMPT)
        store_text(text_content, page_number, document_name, region_index=0)

    print(f"\n{'='*50}")
    print(f"Ingestion complete.")
    print(f"  Text chunks  : {text_col.count()}")
    print(f"  Table chunks : {table_col.count()}")
    print(f"  Chart chunks : {chart_col.count()}")


if __name__ == "__main__":
    PDF_PATH = r"C:\Users\hp\OneDrive\Desktop\Multimodel_doc_int\1706.03762v7.pdf"
    ingest_pdf(PDF_PATH)