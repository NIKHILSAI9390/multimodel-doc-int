import json
import base64
import os
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

# Claude_API_KEY=""

client=anthropic.Anthropic(api_key=Claude_API_KEY)
embedder=SentenceTransformer("all-MiniLM-L6-v2")
chroma=chromadb.PersistentClient(path="./chroma_db")
text_col=chroma.get_or_create_collection("text_chunks")
table_col=chroma.get_or_create_collection("table_chunks")
chart_col=chroma.get_or_create_collection("chart_chunks")


STYLE_INSTRUCTIONS={
    "technical":"Use precise techincal terminology.Be thorough and exact",
    "simple":"Use simple language and avoid jargon. Make it easy to understand",
    "eli5":"Explain like iam five years old.Use simple anaalogies.and very basic language."
    
}

def build_system_prompt(language,style):
    
    style_instruction=STYLE_INSTRUCTIONS.get(style,STYLE_INSTRUCTIONS["simple"])
    return f"""You are a document explanation assistant.
{style_instruction}
Always respond in {language}.
Use only the content provided — do not add outside knowledge."""


# answering specific question

def answer_question(chunk_text,metadata,question,language,style):
    modality=metadata.get("modality","text")
    system=build_system_prompt(language,style)
    
    #chart
    if modality=="chart":
        image_path=metadata.get("image_path","")
        
        if not os.path.exists( image_path):
            
            content = [{"type": "text", "text": f"Context:\n{chunk_text}\n\nQuestion: {question}"}]
        else:
            with open(image_path,"rb") as f:
                image_data=f.read()
                image_b64=base64.b64encode
                content = [
                {
                    "type": "image",
                    "source": {
                        "type"      : "base64",
                        "media_type": "image/png",
                        "data"      : image_b64
                    }
                },
                {"type": "text", "text": f"Question: {question}"}
            ]
    else:
        # Text or table — send chunk as context
        content = [{"type": "text", "text": f"Context:\n{chunk_text}\n\nQuestion: {question}"}]
        
    response=client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role":"user","content":content}]
    )
    return{
        "answer": response.content[0].text.strip(),
        "page_number": metadata.get("page_number"),
        "modality": modality,
        "language": language,
        "mode":"question"
    }

#single paragraph
def explain_paragraph(paragraph_text, page_number, language, style, paragraph_index):
    system=build_system_prompt(language, style)
    prompt = f"""Explain the following paragraph clearly.
    paragraph: {paragraph_text}
    """
    response=client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=[{"role":"user","content":prompt}]
    )
    return{
        "explanation": response.content[0].text.strip(),
        "page_number": page_number,
        "modality": "text",
        "language": language,
        "mode":"paragraph"
    }
# explain page

def explain_page(chunk_text,metadata,language, style):
    modality=metadata.get("modality","text")
    page_number=metadata.get("page_number")
    results=[]
    if modality=="text":
        paragraphs=[p.strip() for p in chunk_text.split("\n\n") if p.strip()]
        
        for i , para in enumerate(paragraphs):
            if len(para) < 30:  # skip very short fragments
                continue
            result = explain_paragraph(para, page_number, language, style, i)
            results.append(result)
    elif modality == "table":
        system = build_system_prompt(language, style)
        prompt = f"""Explain this table clearly. Describe what it shows and highlight the most important data points.

Table content:
{chunk_text}"""
        response = client.messages.create(
            model     = "claude-sonnet-4-6",
            max_tokens= 1024,
            system    = system,
            messages  = [{"role": "user", "content": prompt}]
        )
        results.append({
            "answer"     : response.content[0].text.strip(),
            "page_number": page_number,
            "modality"   : "table",
            "language"   : language,
            "mode"       : "page"
        })
        
        
    elif modality == "chart":
        image_path = metadata.get("image_path", "")
        system     = build_system_prompt(language, style)

        if not os.path.exists(image_path):
            content = [{"type": "text", "text": f"Describe and explain this chart:\n{chunk_text}"}]
        else:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
            content   = [
                {
                    "type": "image",
                    "source": {
                        "type"      : "base64",
                        "media_type": "image/png",
                        "data"      : image_b64
                    }
                },
                {"type": "text", "text": "Explain this figure in detail. What does it show and what is the key insight?"}
            ]

        response = client.messages.create(
            model     = "claude-sonnet-4-6",
            max_tokens= 1024,
            system    = system,
            messages  = [{"role": "user", "content": content}]
        )
        results.append({
            "answer"     : response.content[0].text.strip(),
            "page_number": page_number,
            "modality"   : "chart",
            "language"   : language,
            "mode"       : "page"
        })

    return results

# full doc read

def get_all_pages_ordered(document_name):
    all_chunks=[]
    for col,modality in [(text_col,"text"),(table_col,"table"),(chart_col,"chart")]:
        results=col.query(
            where={"document_name":document_name},
            include=["metadatas","documents"]
        )
        for doc,meta in zip(results["documents"],results["metadatas"]):
            all_chunks.append({
                "text": doc,
                "metadata": meta,
                "modality": modality,
                "page": meta.get("page_number", 0)
            })
    modality_order = {"text": 0, "table": 1, "chart": 2}
    all_chunks.sort(key=lambda x: (x["page"], modality_order.get(x["modality"], 3)))

    return all_chunks    
def read_full_document(document_name, language, style, granularity="paragraph"):
    chunks = get_all_pages_ordered(document_name)
    
    pages = {}
    for chunk in chunks:
        page = chunk["page"]
        if page not in pages:
            pages[page] = []
        pages[page].append(chunk)

    for page_number in sorted(pages.keys()):
        page_chunks  = pages[page_number]
        page_results = []

        for chunk in page_chunks:
            if granularity == "paragraph" and chunk["modality"] == "text":
                results = explain_page(chunk["text"], chunk["metadata"], language, style)
                page_results.extend(results)
            else:
                results = explain_page(chunk["text"], chunk["metadata"], language, style)
                page_results.extend(results)

        # Yield this page's results — Streamlit pauses here until user clicks Continue
        yield page_number, page_results
        
def detect_intent(question):
    q=question.lower().strip()
    
    full_read_triggers=[
        "read the document","read the whole","read everything",
        "summarize the document","summarize everything","summarize all"
        "read aloud","read it out","go through the document"
    ]
    
    section_triggers=[
        "explain the document","explain section","what does page","summarize page", "read page", "tell me about page"
    ]
    if any(t in q for t in full_read_triggers):
        return "full_read"
    elif any(t in q for t in section_triggers):
        return "section"
    else:
        return "question"
    

# function called by streamlit
def process(
    question,
    document_name,
    language      = "English",
    style         = "simple",
    granularity   = "paragraph",
    router_func   = None
):
    """
    Main entry point called by app.py.
    
    Returns:
      For 'question' mode  → single result dict
      For 'full_read' mode → generator (page by page)
      For 'section' mode   → list of result dicts
    """
    intent = detect_intent(question)

    if intent == "full_read":
        return "full_read", read_full_document(document_name, language, style, granularity)

    elif intent == "section":
        # Extract page number from question if mentioned
        import re
        match = re.search(r"page\s+(\d+)", question, re.IGNORECASE)
        if match:
            target_page = int(match.group(1))
            # Find chunks for that page
            for col, modality in [(text_col, "text"), (table_col, "table"), (chart_col, "chart")]:
                results = col.get(
                    where  = {"document_name": document_name, "page_number": target_page},
                    include= ["documents", "metadatas"]
                )
                if results["documents"]:
                    chunk   = results["documents"][0]
                    meta    = results["metadatas"][0]
                    results = explain_page(chunk, meta, language, style)
                    return "section", results
        return "section", []

    else:
        # Standard question — use router
        if router_func:
            chunk, meta, dist, used_modality = router_func(question, document_name)
        else:
            return "question", {"answer": "Router not connected.", "page_number": None}

        result = answer_question(chunk, meta, question, language, style)
        return "question", result
    
    #test
if __name__ == "__main__":
    from router import retrieve_with_fallback

    # Test question mode
    intent, result = process(
        question      = "What BLEU score did Transformer big achieve?",
        document_name = "1706.03762v7",
        language      = "English",
        style         = "simple",
        router_func   = retrieve_with_fallback
    )
    print(f"Intent: {intent}")
    print(f"Answer: {result['answer']}")
    print(f"Page  : {result['page_number']}")
    print(f"Modal : {result['modality']}")