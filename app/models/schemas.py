from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    message: str


class EvidenceItem(BaseModel):
    source_type: str = "spotify_dataset"
    title: Optional[str] = None
    artist: Optional[str] = None
    genre: Optional[str] = None
    emotion: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None


class SongItem(BaseModel):
    title: str
    artist: str
    genre: Optional[str] = None
    emotion: Optional[str] = None
    reason: str
    image: Optional[str] = None
    video_url: Optional[str] = None
    embed_url: Optional[str] = None
    video_id: Optional[str] = None
    score: Optional[float] = None
    length_mins: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    songs: List[SongItem]
    evidence: List[EvidenceItem]
    grounded: bool = True
