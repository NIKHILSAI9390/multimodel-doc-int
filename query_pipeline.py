import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

#setup
GEMINI_API_KEY="AIzaSyBYpK6dbCEbXRRedSk_OjiHS0wK46oz1pA"
client =genai.Client(api_key=GEMINI_API_KEY)
embedder=SentenceTransformer("all-MiniLM-L6-v2")
chroma_client=chromadb.PersistentClient(path="./chroma_db")
collection=chroma_client.get_or_create_collection(name="documents")


# retrevie
def retrieve(question):
    question_embedding = embedder.encode(question).tolist()
    results=collection.query(
        query_embeddings=[question_embedding],
        n_results=1,
        include=["metadatas","documents","distances"]
        
    )
    chunk_text=results["documents"][0][0]
    metadata=results["metadatas"][0][0] 
    distance=results["distances"][0][0]
    return chunk_text,metadata,distance

ANSWER_PROMPT = """You are a document question answering assistant.
Answer the question using ONLY the provided context. Do not use outside knowledge.
If the answer is not in the context, say "I cannot find this in the document."

At the end of your answer, on a new line, write exactly:
SOURCE: <one of: text, table, chart>

Choose SOURCE based on what type of content you used to answer:
- text: if you answered from prose paragraphs
- table: if you answered from tabular data with rows and columns  
- chart: if you answered from a visual diagram or figure"""
# question +chunk to gemini
def answer(question, chunk_text):
    
    
    prompt = f"""Context:
{chunk_text}

Question: {question}"""
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config={"system_instruction": ANSWER_PROMPT},
        contents=prompt
    )
    
    return response.text

def parse_source_tag(answer_text):
    
    lines = answer_text.strip().split("\n")
    
    for line in reversed(lines):  # SOURCE tag is always at the end
        if line.startswith("SOURCE:"):
            return line.replace("SOURCE:", "").strip().lower()
    
    return "unknown"

def check_faithfulness(predicted_source, stored_source_type):
    
    
    match = predicted_source == stored_source_type
    
    return {
        "stored_source_type": stored_source_type,   # what WE labeled during ingestion
        "predicted_source": predicted_source,         # what GEMINI claimed it used
        "faithful": match,                            # do they agree?
        "verdict": "FAITHFUL" if match else " MISMATCH — faithfulness error"
    }
    
def run(question):
   
    
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print('='*60)
    
    # Step 1: Retrieve
    chunk_text, metadata, distance = retrieve(question)
    print(f"\nRETRIEVED: Page {metadata['page_number']} | "
          f"stored source_type: {metadata['source_type']} | "
          f"distance: {distance:.4f}")
    print(f"CHUNK PREVIEW: {chunk_text[:150]}...")
    
    # Step 2: Answer
    answer_text = answer(question, chunk_text)
    print(f"\nANSWER:\n{answer_text}")
    
    # Step 3: Faithfulness check
    predicted_source = parse_source_tag(answer_text)
    result = check_faithfulness(predicted_source, metadata["source_type"])
    
    print(f"\nFAITHFULNESS RESULT:")
    print(f"  Stored source_type : {result['stored_source_type']}")
    print(f"  Gemini SOURCE tag  : {result['predicted_source']}")
    print(f"  Verdict            : {result['verdict']}")
    
    return result


# Test with your first question
if __name__ == "__main__":
    run("What optimizer was used to train the Transformer?")