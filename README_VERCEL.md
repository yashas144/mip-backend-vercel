# Vercel deployment notes

## Included changes
- Added `api/index.py` as the Vercel Python entrypoint.
- Added `vercel.json` routing all requests to the FastAPI function.
- Added `.vercelignore` to keep the deployment smaller.
- Converted `requirements.txt` to normal UTF-8 text.
- Removed `.env` from the deployable package.
- Added `.env.example` for environment variables.
- Added `/health` endpoint.

## Vercel settings
Set these environment variables in the Vercel dashboard:
- `OPENAI_MODEL`
- `OPENAI_API_KEY`
- `YOUTUBE_API_KEY`
- `ARTIFACT_DIR=./artifacts`
- `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `USE_GPU=false`
- `MIN_TOP_RRF_SCORE=0.020`
- `MIN_TOP_DENSE_SCORE=0.30`
- `MIN_ITEM_RRF_SCORE=0.012`
- `MIN_ITEM_DENSE_SCORE=0.22`
- `MIN_ITEM_FINAL_SCORE=0.80`
- `MIN_RESULTS_REQUIRED=2`

## Expected endpoints
- `GET /`
- `GET /health`
- `POST /chat`
