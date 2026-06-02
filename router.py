import json
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

#init
# claude_API_key=""
client = anthropic.Anthropic(api_key=claude_API_key)
embedder=SentenceTransformer("all-MiniLM-L6-v2")
chroma=chromadb.PersistentClient(path="./chroma_db")
text_col=chroma.get_or_create_collection("text_chunks")
table_col=chroma.get_or_create_collection("table_chunks")
chart_col=chroma.get_or_create_collection("chart_chunks")

#classify the question
CLASSIFY_PROMPT="""you are a document query classifier.
Given a user question, decide which type of content in a document would best answer it. 
Reply with exactly one word:
- text   → if the answer is likely in a paragraph or prose section
- table  → if the answer requires specific numbers, comparisons, or structured data
- chart  → if the answer is about a diagram, figure, visualization, or image

One word only. No explanation."""

def classify_question(question):
    response=client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{
            "role":"user",
            "content": f"{CLASSIFY_PROMPT}\n\nQuestion: {question}"
        }]
        
        
    )
    modality=response.content[0].text.strip().lower()
    if modality not in ["text", "table", "chart"]:
        modality = "text"

    return modality
#retrieve relevant chunks from chromadb
def retrieve(question,modality,document_name=None):
    embedding=embedder.encode(question)
    collection_map={
        "text":text_col,
        "table":table_col,
        "chart":chart_col
    }
    collection=collection_map[modality]
    if collection.count() == 0:
        return None, None, None
    #build query
    query_kwargs={"query_embeddings":[embedding],
        "n_results":1,
        "include":["metadatas","documents","distances"]
    }
    
    if document_name:
        query_kwargs["where"]={"document_name":document_name}
    results=collection.query(**query_kwargs)
    if not results["ids"][0]:
        return None,None,None
    chunk_text=results["documents"][0][0]
    metadata=results["metadatas"][0][0]
    distance=results["distances"][0][0]
    if modality=="table"and "json_data" in metadata:
        json_data =json.loads(metadata["json_data"])
        if json_data:
            # Append structured data to chunk text for better answering
            chunk_text = chunk_text + "\n\nStructured data:\n" + json.dumps(json_data, indent=2)
    return chunk_text,metadata,distance
def retrieve_with_fallback(question,document_name=None):
    modality=classify_question(question)
    print(f"  Classified as: {modality}")
    chunk_text,metadata,distance=retrieve(question,modality,document_name)
    if distance is None or distance > 1.2:
        print(f"  Low confidence ({distance:.4f}) — trying all collections...")
        best_distance = distance or 9999
        best_result   = (chunk_text, metadata, distance)
        
        for fallback_modality in ["text", "table", "chart"]:
            if fallback_modality == modality:
                continue
            fb_chunk, fb_meta, fb_dist = retrieve(question, fallback_modality, document_name)
            if fb_dist is not None and fb_dist < best_distance:
                best_distance = fb_dist
                best_result   = (fb_chunk, fb_meta, fb_dist)
                modality      = fallback_modality
                print(f"  Better match in '{fallback_modality}' collection (distance: {fb_dist:.4f})")
        chunk_text, metadata, distance = best_result
    return chunk_text, metadata, distance, modality
    

#test
if __name__ == "__main__":
    test_questions = [
        "What BLEU score did Transformer big achieve on English-to-German?",
        "What optimizer was used to train the Transformer?",
        "What does the attention visualization show for the word making?",
        "How many encoder layers does the base Transformer have?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        chunk, meta, dist, modality = retrieve_with_fallback(q)
        print(f"  Modality used : {modality}")
        print(f"  Retrieved page: {meta['page_number'] if meta else 'None'}")
        print(f"  Distance      : {dist:.4f}" if dist else "  Distance: None")
        print(f"  Preview       : {chunk[:100] if chunk else 'None'}...")
