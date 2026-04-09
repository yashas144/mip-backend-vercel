import os
import pandas as pd
from dotenv import load_dotenv
from app.services.youtube_service import youtube_service
from app.services.hybrid_retriever import hybrid_retriever
from app.services.llm_service import generate_grounded_response

load_dotenv()

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")
PARQUET_PATH = os.path.join(ARTIFACT_DIR, "songs.parquet")

MIN_TOP_RRF_SCORE = float(os.getenv("MIN_TOP_RRF_SCORE", "0.020"))
MIN_TOP_DENSE_SCORE = float(os.getenv("MIN_TOP_DENSE_SCORE", "0.30"))
MIN_ITEM_RRF_SCORE = float(os.getenv("MIN_ITEM_RRF_SCORE", "0.012"))
MIN_ITEM_DENSE_SCORE = float(os.getenv("MIN_ITEM_DENSE_SCORE", "0.22"))
MIN_ITEM_FINAL_SCORE = float(os.getenv("MIN_ITEM_FINAL_SCORE", "0.80"))
MIN_RESULTS_REQUIRED = int(os.getenv("MIN_RESULTS_REQUIRED", "2"))


class RecommendationService:
    def __init__(self):
        self.df = None
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return

        if not os.path.exists(PARQUET_PATH):
            raise FileNotFoundError("Artifacts not found. Run: python build_index.py")

        self.df = pd.read_parquet(PARQUET_PATH)
        hybrid_retriever.initialize()
        self.initialized = True

    def _safe_float(self, value, default=0.0):
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    def _format_length_mins(self, value):
        """
        Converts dataset Length field into minutes.
        Handles:
        - milliseconds
        - seconds
        - already-minute-like values
        - missing/invalid values
        """
        try:
            if pd.isna(value):
                return None

            v = float(value)

            if v <= 0:
                return None

            # Likely milliseconds (common Spotify-style duration units)
            if v > 1000:
                mins = v / 60000.0
            else:
                # Likely seconds
                mins = v / 60.0

            return f"{mins:.2f}"
        except Exception:
            return None

    def _score_row(self, row, query: str) -> float:
        score = 0.0
        q = query.lower()

        work_study = self._safe_float(row.get("Good for Work/Study", 0))
        relax = self._safe_float(row.get("Good for Relaxation/Meditation", 0))
        exercise = self._safe_float(row.get("Good for Exercise", 0))
        party = self._safe_float(row.get("Good for Party", 0))
        driving = self._safe_float(row.get("Good for Driving", 0))
        energy = self._safe_float(row.get("Energy", 0))
        danceability = self._safe_float(row.get("Danceability", 0))
        acousticness = self._safe_float(row.get("Acousticness", 0))
        instrumentalness = self._safe_float(row.get("Instrumentalness", 0))
        speechiness = self._safe_float(row.get("Speechiness", 0))
        popularity = self._safe_float(row.get("Popularity", 0))

        if any(word in q for word in ["study", "coding", "focus", "work"]):
            score += work_study * 2.0
            score += instrumentalness * 0.8
            score -= speechiness * 0.7

        if any(word in q for word in ["relax", "calm", "sleep", "meditation"]):
            score += relax * 2.0
            score += acousticness * 0.8
            score -= energy * 0.4

        if any(word in q for word in ["workout", "gym", "run", "exercise"]):
            score += exercise * 2.0
            score += energy * 1.0
            score += danceability * 0.8

        if "party" in q:
            score += party * 2.0
            score += danceability * 1.0

        if any(word in q for word in ["drive", "driving", "road trip"]):
            score += driving * 2.0

        genre = str(row.get("Genre", "")).lower()
        emotion = str(row.get("emotion", "")).lower()

        if genre and genre in q:
            score += 1.5
        if emotion and emotion in q:
            score += 1.2

        score += min(popularity / 100.0, 1.0) * 0.25
        return score

    def _generate_reason(self, row, query: str) -> str:
        reasons = []
        q = query.lower()

        if any(word in q for word in ["study", "coding", "focus"]) and self._safe_float(row.get("Good for Work/Study", 0)) > 0:
            reasons.append("strong work/study context match")
        if any(word in q for word in ["relax", "calm"]) and self._safe_float(row.get("Good for Relaxation/Meditation", 0)) > 0:
            reasons.append("fits relaxation intent")
        if any(word in q for word in ["workout", "gym", "exercise"]) and self._safe_float(row.get("Energy", 0)) > 0.6:
            reasons.append("high energy profile")
        if self._safe_float(row.get("Danceability", 0)) > 0.65:
            reasons.append("danceable audio features")
        if self._safe_float(row.get("Acousticness", 0)) > 0.5:
            reasons.append("acoustic texture")
        if str(row.get("emotion", "")).strip():
            reasons.append(f"emotion tagged as {row.get('emotion')}")
        if str(row.get("Genre", "")).strip():
            reasons.append(f"genre: {row.get('Genre')}")

        if not reasons:
            reasons.append("hybrid lexical + semantic match")

        return ", ".join(reasons[:3])

    def _top_retrieval_is_confident(self, candidates: pd.DataFrame) -> bool:
        if candidates.empty:
            return False

        top = candidates.iloc[0]
        top_rrf = float(top.get("rrf_score", 0.0))
        top_dense = float(top.get("dense_score", 0.0))

        return (top_rrf >= MIN_TOP_RRF_SCORE) or (top_dense >= MIN_TOP_DENSE_SCORE)

    def _filter_strong_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        strong = candidates[
            (
                (candidates["rrf_score"] >= MIN_ITEM_RRF_SCORE) |
                (candidates["dense_score"] >= MIN_ITEM_DENSE_SCORE)
            ) &
            (candidates["final_score"] >= MIN_ITEM_FINAL_SCORE)
        ].copy()

        return strong.sort_values("final_score", ascending=False)

    def recommend(self, query: str, top_k: int = 5):
        if not self.initialized:
            self.initialize()

        retrieved = hybrid_retriever.hybrid_search(query, top_k=40)

        rows = []
        for item in retrieved:
            idx = item["idx"]
            row = self.df.iloc[idx].copy()
            row["rrf_score"] = item.get("rrf_score", 0.0)
            row["dense_score"] = item.get("dense_score", 0.0)
            row["bm25_score"] = item.get("bm25_score", 0.0)
            rows.append(row)

        candidates = pd.DataFrame(rows)
        if candidates.empty:
            return {
                "response": "I could not find enough grounded evidence for that request.",
                "songs": [],
                "evidence": [],
                "grounded": True,
            }

        if not self._top_retrieval_is_confident(candidates):
            return {
                "response": "I could not find strong enough grounded evidence in the Spotify dataset for that request.",
                "songs": [],
                "evidence": [],
                "grounded": True,
            }

        candidates["reason"] = candidates.apply(lambda row: self._generate_reason(row, query), axis=1)
        candidates["rule_score"] = candidates.apply(lambda row: self._score_row(row, query), axis=1)
        candidates["final_score"] = (
            candidates["rrf_score"] * 3.0 +
            candidates["rule_score"] * 1.5
        )

        strong_df = self._filter_strong_candidates(candidates)

        if len(strong_df) < MIN_RESULTS_REQUIRED:
            return {
                "response": "I could not find enough strong grounded recommendations from the Spotify dataset for that request.",
                "songs": [],
                "evidence": [],
                "grounded": True,
            }

        final_df = strong_df.head(top_k).copy()

        evidence = []
        songs = []

        for _, row in final_df.iterrows():
            title = str(row.get("song", "Unknown Song"))
            artist = str(row.get("Artist(s)", "Unknown Artist"))
            yt_query = f"{title} {artist} official audio"
            yt_data = youtube_service.search_video(yt_query) or {}

            score = float(row.get("final_score", 0.0))
            reason = str(row.get("reason", "hybrid match"))
            length_mins = self._format_length_mins(row.get("Length"))

            evidence_item = {
                "source_type": "spotify_dataset",
                "title": title,
                "artist": artist,
                "genre": str(row.get("Genre", "")) if pd.notna(row.get("Genre")) else None,
                "emotion": str(row.get("emotion", "")) if pd.notna(row.get("emotion")) else None,
                "score": score,
                "reason": reason,
            }
            evidence.append(evidence_item)

            songs.append({
                "title": title,
                "artist": artist,
                "genre": str(row.get("Genre", "")) if pd.notna(row.get("Genre")) else None,
                "emotion": str(row.get("emotion", "")) if pd.notna(row.get("emotion")) else None,
                "reason": reason,
                "image": yt_data.get("image"),
                "video_url": yt_data.get("video_url"),
                "embed_url": yt_data.get("embed_url"),
                "video_id": yt_data.get("video_id"),
                "score": score,
                "length_mins": length_mins,
            })

        llm_result = generate_grounded_response(query, final_df.to_dict(orient="records"))

        allowed_pairs = {(e["title"], e["artist"]) for e in evidence}
        songs = [s for s in songs if (s["title"], s["artist"]) in allowed_pairs]

        return {
            "response": llm_result["response"],
            "songs": songs,
            "evidence": evidence,
            "grounded": bool(llm_result.get("grounded", True)),
        }


recommendation_service = RecommendationService()

