import pdfplumber

PDF_PATH = r"C:\Users\hp\OneDrive\Desktop\Multimodel_doc_int\1706.03762v7.pdf"

with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(pdf.pages):
        tables = page.find_tables()
        if tables:
            print(f"Page {i+1}: {len(tables)} table(s) detected")
            for t in tables:
                text = page.within_bbox(t.bbox).extract_text() or ""
                preview = text[:80].replace("\n", " ")
                print(f"  bbox={t.bbox} | preview: {preview}")