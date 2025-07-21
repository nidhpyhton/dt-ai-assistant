import json
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load OpenAI API key from .env
load_dotenv()

# ----------- CONFIG -------------
CHUNK_FILE = "../data/capital_gains_chunks.jsonl"
VECTOR_STORE_DIR = "../data/vector_store/"
EMBED_MODEL = "text-embedding-3-small"
# --------------------------------

def load_chunks(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return chunks

def prepare_documents(chunks):
    docs = []
    for chunk in chunks:
        metadata = {
            "chunk_id": chunk.get("chunk_id"),
            "start_page": chunk.get("start_page"),
            "end_page": chunk.get("end_page"),
            "section_title": chunk.get("section_title", "Unknown")
        }
        doc = Document(page_content=chunk["text"], metadata=metadata)
        docs.append(doc)
    return docs

def build_and_save_vectorstore(docs):
    print("🔗 Generating embeddings using OpenAI...")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    print("📦 Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    print(f"💾 Saving vector store to {VECTOR_STORE_DIR} ...")
    vectorstore.save_local(VECTOR_STORE_DIR)

    print("✅ Embedding and storage complete.")

if __name__ == "__main__":
    if not os.path.exists(CHUNK_FILE):
        print(f"❌ Error: Chunk file not found at {CHUNK_FILE}")
        exit(1)

    chunks = load_chunks(CHUNK_FILE)
    docs = prepare_documents(chunks)
    build_and_save_vectorstore(docs)
