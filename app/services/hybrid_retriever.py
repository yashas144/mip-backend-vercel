import gc
import os
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from app.services.preprocessing_service import PreprocessingService

load_dotenv()

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

FAISS_PATH = os.path.join(ARTIFACT_DIR, "songs.faiss")
BM25_TOKENS_PATH = os.path.join(ARTIFACT_DIR, "bm25_tokenized.pkl")


class HybridRetriever:
    def __init__(self):
        self.model = None
        self.index = None
        self.bm25 = None
        self.initialized = False

    def _get_device(self):
        if USE_GPU:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    def initialize(self):
        if self.initialized:
            return

        device = self._get_device()

        # Load model with cache folder to avoid re-downloading on restart
        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=device,
            cache_folder="/tmp/model_cache",
        )

        # Load FAISS index
        self.index = faiss.read_index(FAISS_PATH)

        # Load BM25 tokenized corpus
        with open(BM25_TOKENS_PATH, "rb") as f:
            tokenized_corpus = pickle.load(f)

        self.bm25 = BM25Okapi(tokenized_corpus)

        # Free memory used during loading
        gc.collect()

        self.initialized = True

    def dense_search(self, query: str, top_k: int = 15):  # reduced from 50
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(q_emb, top_k)

        # Free embedding memory immediately
        del q_emb
        gc.collect()

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx >= 0:
                results.append({
                    "idx": int(idx),
                    "dense_score": float(score),
                    "dense_rank": rank,
                })
        return results

    def bm25_search(self, query: str, top_k: int = 15):  # reduced from 50
        tokens = PreprocessingService.tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            results.append({
                "idx": int(idx),
                "bm25_score": float(scores[idx]),
                "bm25_rank": rank,
            })

        # Free scores array memory
        del scores
        gc.collect()

        return results

    def hybrid_search(self, query: str, top_k: int = 15, rrf_k: int = 60):  # reduced from 30
        dense = self.dense_search(query, top_k=top_k)
        sparse = self.bm25_search(query, top_k=top_k)

        fused = {}

        for item in dense:
            idx = item["idx"]
            fused.setdefault(idx, {"idx": idx})
            fused[idx].update(item)
            fused[idx]["rrf_score"] = fused[idx].get("rrf_score", 0.0) + 1.0 / (rrf_k + item["dense_rank"])

        for item in sparse:
            idx = item["idx"]
            fused.setdefault(idx, {"idx": idx})
            fused[idx].update(item)
            fused[idx]["rrf_score"] = fused[idx].get("rrf_score", 0.0) + 1.0 / (rrf_k + item["bm25_rank"])

        results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
        return results[:top_k]


hybrid_retriever = HybridRetriever()