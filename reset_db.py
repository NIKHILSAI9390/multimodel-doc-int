import chromadb
import shutil
import os

if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("chroma wiped successfully.")
    
client=chromadb.PersistentClient(path="./chroma_db")
client.get_or_create_collection("text_chunks")
client.get_or_create_collection("table_chunks")
client.get_or_create_collection("chart_chunks")

print("Three fresh collections created:")
print("  - text_chunks")
print("  - table_chunks")
print("  - chart_chunks")