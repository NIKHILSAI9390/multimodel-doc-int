import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
import time

GEMINI_API_KEY = "AIzaSyBYpK6dbCEbXRRedSk_OjiHS0wK46oz1pA"
client = genai.Client(api_key=GEMINI_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# You defined ground truth by reading the PDF yourself
TEST_SET = [
    {
        "question": "What is the per-layer complexity of a self-attention layer?",
        "correct_modality": "table",
    },
    {
        "question": "What BLEU score did Transformer big achieve on English-to-German?",
        "correct_modality": "table",
    },
    {
        "question": "What F1 score did Transformer 4 layers achieve on WSJ in the semi-supervised setting?",
        "correct_modality": "table",
    },
    {
        "question": "How many parameters does the big Transformer model have in millions?",
        "correct_modality": "table",
    },
    {
        "question": "Which encoder layer is shown in the attention visualization for the word making?",
        "correct_modality": "chart",
    },
    {
        "question": "What word are attentions shown for in Figure 3?",
        "correct_modality": "chart",
    },
    {
        "question": "What optimizer was used to train the Transformer?",
        "correct_modality": "text",
    },
    {
        "question": "How many encoder layers does the base Transformer have?",
        "correct_modality": "text",
    },
]


ANSWER_PROMPT = """You are a document question answering assistant.
Answer the question using ONLY the provided context. Do not use outside knowledge.
If the answer is not in the context, say "I cannot find this in the document."

At the end of your answer, on a new line, write exactly:
SOURCE: <one of: text, table, chart>

Choose SOURCE based on what type of content you used to answer:
- text: if you answered from prose paragraphs
- table: if you answered from tabular data with rows and columns
- chart: if you answered from a visual diagram or figure"""


def retrieve(question):
    embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )
    return (
        results["documents"][0][0],
        results["metadatas"][0][0],
        results["distances"][0][0]
    )


def answer(question, chunk_text):
    prompt = f"Context:\n{chunk_text}\n\nQuestion: {question}"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config={"system_instruction": ANSWER_PROMPT},
        contents=prompt
    )
    return response.text


def parse_source_tag(answer_text):
    for line in reversed(answer_text.strip().split("\n")):
        if line.startswith("SOURCE:"):
            return line.replace("SOURCE:", "").strip().lower()
    return "unknown"


def evaluate():
    results = []
    
    print("Starting evaluation...\n")
    
    for i, item in enumerate(TEST_SET):
        question = item["question"]
        correct_modality = item["correct_modality"]
        
        print(f"[{i+1}/{len(TEST_SET)}] {question}")
        
        # Retrieve + Answer
        chunk_text, metadata, distance = retrieve(question)
        answer_text = answer(question, chunk_text)
        predicted_source = parse_source_tag(answer_text)
        
        # Faithfulness check against YOUR ground truth
        faithful = predicted_source == correct_modality
        
        result = {
            "question": question,
            "correct_modality": correct_modality,
            "predicted_source": predicted_source,
            "retrieved_page": metadata["page_number"],
            "retrieval_distance": round(distance, 4),
            "answer": answer_text.split("\nSOURCE:")[0].strip(),
            "faithful": faithful,
        }
        
        results.append(result)
        
        verdict = "yes" if faithful else "no"
        print(f"  Retrieved page {metadata['page_number']} | "
              f"correct: {correct_modality} | "
              f"predicted: {predicted_source} | {verdict}\n")
        
        time.sleep(25)  # rate limit
    
    return results

def print_report(results):
    total = len(results)
    faithful_count = sum(1 for r in results if r["faithful"])
    faithfulness_score = faithful_count / total * 100
    
    print("\n" + "="*60)
    print("FAITHFULNESS EVALUATION REPORT")
    print("="*60)
    
    # Per-modality breakdown
    for modality in ["text", "table", "chart"]:
        modality_results = [r for r in results if r["correct_modality"] == modality]
        if not modality_results:
            continue
        correct = sum(1 for r in modality_results if r["faithful"])
        print(f"{modality.upper():6} → {correct}/{len(modality_results)} faithful")
    
    print(f"\nOVERALL FAITHFULNESS SCORE: {faithful_count}/{total} = {faithfulness_score:.1f}%")
    
    # Show failures clearly
    failures = [r for r in results if not r["faithful"]]
    if failures:
        print(f"\nFAITHFULNESS ERRORS ({len(failures)}):")
        for r in failures:
            print(f"  Q: {r['question'][:60]}...")
            print(f"     Expected: {r['correct_modality']} | Got: {r['predicted_source']}")


if __name__ == "__main__":
    results = evaluate()
    print_report(results)
