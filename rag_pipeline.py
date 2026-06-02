import chromadb
from sentence_transformers import SentenceTransformer
from google import genai 
from google.genai import types 
#............setup.......
# embeeding model
embedder =SentenceTransformer("all-MiniLM-L6-v2")
db=chromadb.PersistentClient(path="./chroma_db")
#collection = it is used for like a grp of docs related staying in one 
collection=db.get_or_create_collection(name="documents")
#api
client = genai.Client(api_key="AIzaSyBYpK6dbCEbXRRedSk_OjiHS0wK46oz1pA")

#ingestion......

def ingest_text_chunk(chunk_id,text,metadata):
    
    vector= embedder.encode(text).tolist()
    collection.add(
        ids=[chunk_id],
        embeddings=[vector],
        metadatas=[metadata],
        documents=[text]
    )
    print(f"ingestion:{chunk_id}")
    
    #retrieval.....
def retrieve_relevant_chunks(question,top_k=3):
    
    query_vector=embedder.encode(question).tolist()
    results=collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
    )
    return results['documents'][0],results['metadatas'][0]

# answering .....

def answer_question(question):
    
    chunks,metadatas=retrieve_relevant_chunks(question)
    context=""
    for i,(chunk,meta) in enumerate(zip(chunks,metadatas)):
        context+=f"\n[Chunk {i+1}|Source:{meta.get('source_type', 'unknown')} "
        context += f"| Page: {meta.get('page', '?')}]\n{chunk}\n"
        
    # to model
    prompt=f"""You are a document analysis assistant.
    Answer the question using only the provided context.
End every answer with: SOURCE: [text/table/chart]
context:
{context}
Question:{question}"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0)
    )

    return response.text


# test it 

# ingest some fake document chunks
ingest_text_chunk(
    chunk_id="report_page1",
    text="Q3 revenue was $4.2 million, up 23% from Q2. Operating costs were $2.1 million.",
    metadata={"filename": "annual_report.pdf", "page": 1, "source_type": "text"}
)

ingest_text_chunk(
    chunk_id="report_page2",
    text="The Asia Pacific region contributed 40% of total revenue. APAC growth was driven by new enterprise contracts.",
    metadata={"filename": "annual_report.pdf", "page": 2, "source_type": "text"}
)

ingest_text_chunk(
    chunk_id="report_page3",
    text="Employee headcount increased from 200 to 267. Engineering team grew by 45%.",
    metadata={"filename": "annual_report.pdf", "page": 3, "source_type": "text"}
)


print("\n" + "="*50)
print("Question: What was Q3 revenue?")
print("="*50)
print(answer_question("What was Q3 revenue?"))

print("\n" + "="*50)
print("Question: Tell me about the Asia Pacific region")
print("="*50)
print(answer_question("Tell me about the Asia Pacific region"))