import os
import pickle
import pandas as pd
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from app.services.preprocessing_service import PreprocessingService

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "./data/spotify_dataset.csv")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_ROWS = int(os.getenv("MAX_ROWS", "1000"))
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

PARQUET_PATH = os.path.join(ARTIFACT_DIR, "songs.parquet")
FAISS_PATH = os.path.join(ARTIFACT_DIR, "songs.faiss")
MODEL_NAME_PATH = os.path.join(ARTIFACT_DIR, "model_name.txt")
BM25_CORPUS_PATH = os.path.join(ARTIFACT_DIR, "bm25_corpus.pkl")
BM25_TOKENS_PATH = os.path.join(ARTIFACT_DIR, "bm25_tokenized.pkl")


def get_device():
    if USE_GPU:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


def build_index():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("[1/6] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = PreprocessingService.prepare_dataframe(df)

    if MAX_ROWS > 0 and len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)

    print(f"Rows used for indexing: {len(df)}")

    device = get_device()
    print(f"[2/6] Loading embedding model on device: {device}")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print("[3/6] Encoding dense documents...")
    docs = df["document"].tolist()
    batch_size = 256 if device == "cpu" else 512
    embeddings = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    print("[4/6] Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("[5/6] Building BM25 artifacts...")
    sparse_docs = df["sparse_text"].fillna("").tolist()
    tokenized_corpus = [PreprocessingService.tokenize(x) for x in sparse_docs]

    with open(BM25_CORPUS_PATH, "wb") as f:
        pickle.dump(sparse_docs, f)

    with open(BM25_TOKENS_PATH, "wb") as f:
        pickle.dump(tokenized_corpus, f)

    print("[6/6] Saving artifacts...")
    df.to_parquet(PARQUET_PATH, index=False)
    faiss.write_index(index, FAISS_PATH)

    with open(MODEL_NAME_PATH, "w", encoding="utf-8") as f:
        f.write(EMBEDDING_MODEL)

    print("Artifacts saved:")
    print(f"- {PARQUET_PATH}")
    print(f"- {FAISS_PATH}")
    print(f"- {BM25_CORPUS_PATH}")
    print(f"- {BM25_TOKENS_PATH}")
    print(f"- {MODEL_NAME_PATH}")


if __name__ == "__main__":
    build_index()