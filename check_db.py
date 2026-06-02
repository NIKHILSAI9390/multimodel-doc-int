import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

# Delete the 3 old malformed chunks
collection.delete(ids=["report_page1", "report_page2", "report_page3"])
print("Deleted old chunks.")

# Now print all remaining chunks cleanly
results = collection.get(include=["metadatas", "documents"])
print(f"Total chunks: {len(results['ids'])}")

for i, metadata in enumerate(results["metadatas"]):
    preview = results["documents"][i][:80]
    print(f"Page {metadata['page_number']} | {metadata['source_type']} | {preview}...")