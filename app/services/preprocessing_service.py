import re
import pandas as pd


class PreprocessingService:
    @staticmethod
    def safe_text(value):
        if pd.isna(value):
            return ""
        return str(value).strip()

    @staticmethod
    def safe_float(value, default=0.0):
        try:
            if pd.isna(value):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def tokenize(text: str):
        text = PreprocessingService.normalize_text(text)
        return re.findall(r"\b[a-z0-9]+\b", text)

    @staticmethod
    def build_document(row):
        parts = [
            f"song {PreprocessingService.safe_text(row.get('song'))}",
            f"artist {PreprocessingService.safe_text(row.get('Artist(s)'))}",
            f"genre {PreprocessingService.safe_text(row.get('Genre'))}",
            f"emotion {PreprocessingService.safe_text(row.get('emotion'))}",
            f"album {PreprocessingService.safe_text(row.get('Album'))}",
            f"popularity {PreprocessingService.safe_text(row.get('Popularity'))}",
            f"energy {PreprocessingService.safe_text(row.get('Energy'))}",
            f"danceability {PreprocessingService.safe_text(row.get('Danceability'))}",
            f"tempo {PreprocessingService.safe_text(row.get('Tempo'))}",
            f"speechiness {PreprocessingService.safe_text(row.get('Speechiness'))}",
            f"acousticness {PreprocessingService.safe_text(row.get('Acousticness'))}",
            f"instrumentalness {PreprocessingService.safe_text(row.get('Instrumentalness'))}",
            f"study {PreprocessingService.safe_text(row.get('Good for Work/Study'))}",
            f"relax {PreprocessingService.safe_text(row.get('Good for Relaxation/Meditation'))}",
            f"exercise {PreprocessingService.safe_text(row.get('Good for Exercise'))}",
            f"party {PreprocessingService.safe_text(row.get('Good for Party'))}",
            f"drive {PreprocessingService.safe_text(row.get('Good for Driving'))}",
        ]

        lyrics = PreprocessingService.safe_text(row.get("text"))
        if lyrics:
            parts.append(f"lyrics {lyrics[:300]}")

        return " | ".join(parts)

    @staticmethod
    def build_sparse_text(row):
        fields = [
            row.get("song"),
            row.get("Artist(s)"),
            row.get("Genre"),
            row.get("emotion"),
            row.get("Album"),
            row.get("text"),
        ]
        merged = " ".join(
            [PreprocessingService.safe_text(x) for x in fields if PreprocessingService.safe_text(x)]
        )
        return PreprocessingService.normalize_text(merged)

    @staticmethod
    def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop_duplicates()

        text_cols = [
            "song", "Artist(s)", "Genre", "emotion", "Album", "text",
            "Good for Party", "Good for Work/Study", "Good for Relaxation/Meditation",
            "Good for Exercise", "Good for Driving"
        ]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("")

        numeric_cols = [
            "Popularity", "Energy", "Danceability", "Acousticness",
            "Instrumentalness", "Tempo", "Speechiness", "Liveness"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["document"] = df.apply(PreprocessingService.build_document, axis=1)
        df["sparse_text"] = df.apply(PreprocessingService.build_sparse_text, axis=1)
        df = df[df["document"].str.len() > 0].reset_index(drop=True)
        return df