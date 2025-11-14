import os, re
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")
DB_DIR.mkdir(exist_ok=True)

def load_pdfs(data_dir: Path):
    docs = []
    for pdf_path in data_dir.glob("**/*.pdf"):
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                docs.append({"source": pdf_path.name, "page": i+1, "text": text})
    return docs

def chunk_text(text, chunk_size=800, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def main():
    print(">> Loading PDFs...")
    raw_docs = load_pdfs(DATA_DIR)
    if not raw_docs:
        raise SystemExit("No PDFs found in ./data. Put some PDFs and retry.")

    print(">> Chunking...")
    chunked = []
    for d in tqdm(raw_docs, ncols=80):
        for j, part in enumerate(chunk_text(d["text"], 800, 200)):
            chunked.append({
                "id": f'{d["source"]}-{d["page"]}-chunk{j}',
                "text": part,
                "source": d["source"],
                "page": d["page"]
            })
    print(f">> Total chunks: {len(chunked)}")

    print(">> Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(">> Connecting ChromaDB...")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.create_collection("docs", metadata={"hnsw:space": "cosine"})

    print(">> Embedding & upserting...")
    batch = 256
    ids, texts, metas = [], [], []
    for i, ch in enumerate(tqdm(chunked, ncols=80)):
        ids.append(ch["id"]); texts.append(ch["text"]); metas.append({"source": ch["source"], "page": ch["page"]})
        if len(ids) == batch or i == len(chunked) - 1:
            embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            col.upsert(ids=ids, documents=texts, embeddings=embs, metadatas=metas)
            ids, texts, metas = [], [], []
    print("✅ Index built at ./chroma_db")

if __name__ == "__main__":
    main()