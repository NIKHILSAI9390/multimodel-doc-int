import chromadb, json

chroma = chromadb.PersistentClient(path="./chroma_db")
table_col = chroma.get_or_create_collection("table_chunks")

tables = table_col.get(include=["metadatas", "documents"])
for i, meta in enumerate(tables["metadatas"]):
    print(f"\nPage {meta['page_number']} | {tables['ids'][i]}")
    print(f"Description: {tables['documents'][i]}")
    json_data = json.loads(meta["json_data"])
    print(f"Rows: {len(json_data)}")
    if json_data:
        print(f"First row: {json_data[0]}")