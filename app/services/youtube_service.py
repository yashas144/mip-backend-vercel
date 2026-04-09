import os
import requests
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


class YouTubeService:
    def __init__(self):
        self.cache = {}

    def search_video(self, query: str):
        if not YOUTUBE_API_KEY:
            return None

        if query in self.cache:
            return self.cache[query]

        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 1,
            "videoEmbeddable": "true",
            "key": YOUTUBE_API_KEY,
        }

        try:
            response = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            items = data.get("items", [])
            if not items:
                self.cache[query] = None
                return None

            item = items[0]
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            thumbnails = snippet.get("thumbnails", {})
            image = (
                thumbnails.get("high", {}).get("url")
                or thumbnails.get("medium", {}).get("url")
                or thumbnails.get("default", {}).get("url")
            )

            result = {
                "image": image,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "embed_url": f"https://www.youtube.com/embed/{video_id}",
                "video_id": video_id,
            }
            self.cache[query] = result
            return result
        except Exception:
            self.cache[query] = None
            return None


youtube_service = YouTubeService()
