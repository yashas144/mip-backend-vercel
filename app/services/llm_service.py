import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def _build_evidence_payload(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence = []
    for row in rows:
        evidence.append({
            "title": str(row.get("song", "")),
            "artist": str(row.get("Artist(s)", "")),
            "genre": str(row.get("Genre", "")) if row.get("Genre") is not None else "",
            "emotion": str(row.get("emotion", "")) if row.get("emotion") is not None else "",
            "reason": str(row.get("reason", "")),
            "score": float(row.get("final_score", 0.0)),
        })
    return evidence


def generate_grounded_response(query: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "response": "I could not find enough grounded evidence in the Spotify dataset for that request.",
            "grounded": True,
        }

    if client is None:
        top = rows[:3]
        summary = []
        for row in top:
            title = row.get("song", "Unknown Song")
            artist = row.get("Artist(s)", "Unknown Artist")
            genre = row.get("Genre", "")
            emotion = row.get("emotion", "")
            parts = [f"{title} by {artist}"]
            if genre:
                parts.append(f"genre: {genre}")
            if emotion:
                parts.append(f"emotion: {emotion}")
            summary.append(", ".join(parts))

        return {
            "response": f"Based only on retrieved Spotify dataset evidence, the best matches for '{query}' are: " + "; ".join(summary),
            "grounded": True,
        }

    evidence = _build_evidence_payload(rows)

    system_prompt = """
You are a music recommendation assistant.

Rules:
1. Use ONLY the provided evidence.
2. Do NOT invent songs, artists, genres, emotions, or reasons.
3. Do NOT mention any song that is not present in the evidence list.
4. If the evidence is insufficient, say so.
5. Summarize the retrieved songs and why they match the query.
6. Keep the answer concise.
7. Return valid JSON only.
"""

    user_payload = {
        "query": query,
        "evidence": evidence,
    }

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "grounded_music_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "response": {"type": "string"},
                            "grounded": {"type": "boolean"},
                        },
                        "required": ["response", "grounded"],
                        "additionalProperties": False,
                    },
                },
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        return {
            "response": parsed.get("response", "I found grounded matches from the Spotify dataset."),
            "grounded": bool(parsed.get("grounded", True)),
        }

    except Exception:
        top = rows[:3]
        summary = []
        for row in top:
            title = row.get("song", "Unknown Song")
            artist = row.get("Artist(s)", "Unknown Artist")
            genre = row.get("Genre", "")
            emotion = row.get("emotion", "")
            parts = [f"{title} by {artist}"]
            if genre:
                parts.append(f"genre: {genre}")
            if emotion:
                parts.append(f"emotion: {emotion}")
            summary.append(", ".join(parts))

        return {
            "response": f"Based only on retrieved Spotify dataset evidence, the best matches for '{query}' are: " + "; ".join(summary),
            "grounded": True,
        }