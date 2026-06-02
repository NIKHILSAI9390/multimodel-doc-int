#imports and config
import os 
import base64
from pathlib import Path
from pdf2image import convert_from_path
import anthropic
from google.genai import types
import chromadb
from sentence_transformers import SentenceTransformer
import time
#poppler locationn
POPPLER_PATH = r"C:\Program Files\poppler-25.12.0\Library\bin"
#claude api key
# claude_API_KEY=""
#PATH to pdf 
PDF_PATH="C:\\Users\\hp\\OneDrive\\Desktop\\Multimodel_doc_int\\1706.03762v7.pdf"

#intilize everything 
client = anthropic.Anthropic(api_key=claude_API_KEY)
#embedder 
embedder=SentenceTransformer("all-MiniLM-L6-v2")
#chromadb
chroma_client=chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

#covert paages to images 
def pdf_to_images(pdf_path):
    print(f"Converting PDF to images: {pdf_path}")
    images=convert_from_path(pdf_path,poppler_path=POPPLER_PATH)
    print(f"Total pages found: {len(images)}")
    return images 



#ingest img to gemini

EXTRACTION_PROMPT="""You are a document content extractor.

You will be given an image of a single page from a document.
Your job is to extract ALL information from the page without missing any data values, labels, or text.

Respond in exactly this format:
TYPE: <one of: text, table, chart>
CONTENT: <complete extracted content — every word, every number, every row, every axis label, nothing summarized>

Rules:
- TYPE must be exactly one word: text, table, or chart
- CONTENT must contain every piece of information visible on the page
- Do not add explanations, preamble, or anything outside this format"""

def extract_from_image(pil_image,page_number):
    print(f"sendig page{page_number} to gemini vision")
    
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(pil_image).decode('utf-8')}},
                {"type": "text", "text": EXTRACTION_PROMPT}
            ]}
        ]
    )
    return response.text

#parse claude response 
def parse_claude_response(response_text):
    
    lines = response_text.strip().split("\n")
    
    source_type = "text"  # default fallback
    content_lines = []
    in_content = False
    
    for line in lines:
        if line.startswith("TYPE:"):
            source_type = line.replace("TYPE:", "").strip().lower()
        elif line.startswith("CONTENT:"):
            content_lines.append(line.replace("CONTENT:", "").strip())
            in_content = True
        elif in_content:
            content_lines.append(line)
    
    content = "\n".join(content_lines).strip()
    return source_type, content

#embed and store 

def store_chunk(content,source_type,page_number,document_name):
    chunk_id = f"{document_name}_page{page_number}"
    embedding = embedder.encode(content).tolist()
    collection.add(
        ids=[chunk_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{
            "document_name": document_name,
            "page_number": page_number,
            "source_type": source_type
        }]
    )
    
    print(f"  Stored: {chunk_id} | source_type: {source_type}")
    
    #pipline 
def ingest_pdf(pdf_path):

    
    
    document_name = Path(pdf_path).name  # e.g. "annual_report.pdf"
    images = pdf_to_images(pdf_path)
    
    for page_number, image in enumerate(images, start=1):
        chunk_id = f"{document_name}_page{page_number}"
        existing = collection.get(ids=[chunk_id])
    
        if existing["ids"]:
            print(f"  Page {page_number} already in DB, skipping.")
            continue
        print(f"\nProcessing page {page_number}/{len(images)}...")
        
        # Step 1: Get extracted text + type from Claude
        raw_response = extract_from_image(image, page_number)
        print(f"  Raw response:\n{raw_response[:200]}...")  # preview first 200 chars
        
        # Step 2: Parse into structured fields
        source_type, content = parse_claude_response(raw_response)
        
            # Step 3: Store in ChromaDB
        if content:
            store_chunk(content, source_type, page_number, document_name)
        
        else:
            print(f"  WARNING: Empty content on page {page_number}, skipping.")
        time.sleep(25)
    print(f"\nIngestion complete. {len(images)} pages processed.")


# Run it
if __name__ == "__main__":
    ingest_pdf(PDF_PATH)