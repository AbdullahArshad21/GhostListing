# GhostListing

An agentic fraud-detection pipeline for online marketplaces. It catches stolen product photos reused across different sellers' listings, checks whether a listing's claimed text actually matches what's visible in its photo, and pauses for human moderator approval before anything is actioned.

Built as a portfolio project to demonstrate production-shaped AI system design: deterministic similarity search where it's the right tool, LLM reasoning only where judgment is actually needed, and a human-in-the-loop gate on every consequential decision.

## What it actually does

1. A seller uploads a listing (image + title + description + category).
2. The image is embedded with CLIP and searched against every other listing already in the system using FAISS.
3. If it's a near-duplicate of another seller's photo, a vision-language model (via Groq) describes what's actually in the image.
4. A second LLM call checks whether the listing's claimed text is consistent with that description (e.g. catches "brand new, sealed in box" claimed on a photo that clearly shows a mounted, used item).
5. The case pauses — a LangGraph agent, not a script — waiting for a human moderator to confirm fraud or mark it a false positive.
6. Every decision is logged to an audit trail.

This isn't hypothetical — it's been run end-to-end on real product images (a subset of the Stanford Online Products dataset) with deliberately injected duplicate photos, and it correctly flagged both duplicates while producing zero false positives on ~75 unrelated images.

## Architecture

```
Image upload → CLIP embedding → FAISS similarity search (deterministic)
                                        │
                              [match found?] ──No──→ done, no LLM cost spent
                                        │ Yes
                                        ▼
                        Groq vision model describes the photo
                                        │
                                        ▼
                  Groq text model checks listing text vs. photo
                                        │
                                        ▼
                      Paused for human moderator (LangGraph interrupt)
                                        │
                             [approve / reject]
                                        │
                                        ▼
                          Audit log (SQLite) + resolution
```

**Why LangGraph, not a plain script:** routing is genuinely conditional (the expensive LLM steps only run on the small fraction of listings FAISS actually flags), and the human-approval step is a durable pause — the agent can sit paused for hours waiting on a real moderator, not just a few seconds in memory. A linear script can't cleanly do either of those.

**Why FAISS for the fraud check, not an LLM:** near-duplicate image detection is a similarity-search problem with a clear right answer, not something requiring judgment. Using an LLM for this would be slower, more expensive, and less reliable than a vector index built for exactly this.

**Why an LLM for the consistency check, not a rule engine:** whether "brand new" text matches a photo showing visible wear is a judgment call, not something a fixed rule can reliably catch — this is where LLM reasoning earns its place in the pipeline.

## Stack

- **CLIP** (`open_clip_torch`, ViT-B-32, LAION-2B weights) — image embeddings
- **FAISS** — similarity search index
- **LangGraph** — agent orchestration, conditional routing, human-approval interrupts
- **Groq** (`qwen/qwen3.6-27b` for vision, `openai/gpt-oss-20b` for text reasoning) — free-tier LLM inference
- **FastAPI** — backend API
- **SQLite** — audit log
- **Vanilla HTML/JS** — moderator review dashboard

Every component here is free at portfolio scale — no paid APIs or hosted vector DB required.

## Running it locally

### 1. Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your-groq-api-key-here
```

Get a free key at [console.groq.com](https://console.groq.com) — no card required.

### 3. Get test images

`src/extract_dataset.py` pulls real images out of a locally downloaded copy of the Stanford Online Products dataset (Kaggle) and splits them into seller folders, deliberately duplicating a couple across sellers as known fraud test cases. Update `DATASET_DIR` in that file to point at your own download, then:

```powershell
cd src
python extract_dataset.py
```

### 4. Build the similarity index

```powershell
python build_index.py
```

This embeds every image with CLIP (uses your GPU automatically if available via CUDA) and scans for cross-seller duplicates.

### 5. Run the backend

```powershell
uvicorn api:app --reload --port 8000
```

### 6. Run the frontend

Open `frontend/index.html` via a local server (e.g. VS Code's Live Server extension) rather than double-clicking it directly — opening it as a raw `file://` URL triggers browser security restrictions that block requests to `localhost`.

### 7. Upload a test listing

```powershell
python test_upload.py
```

Flagged listings will appear in the browser dashboard for review.

## Known limitations / honest notes

- The FAISS distance threshold (`0.3`) is a starting guess, not yet validated against a large labeled evaluation set — a real precision/recall run with more data is a natural next step.
- No authentication on the API yet — this is a functional prototype, not hardened for multi-tenant production use.
- The dataset used for testing is a public academic benchmark, not real marketplace data — real deployment would need real seller upload patterns to validate against.
- Model IDs on Groq's free tier change over time; if `describe_image`/`check_consistency` start returning 404s, check [console.groq.com/docs/models](https://console.groq.com/docs/models) for current model names.

## What this project demonstrates

Multi-agent orchestration with genuine conditional branching (not agents bolted on for the sake of it), a deliberate choice of deterministic vs. ML vs. LLM components based on which is actually the right tool for each sub-problem, a non-bypassable human-approval gate on consequential decisions, and an end-to-end system validated against real data with a real (if small-scale) precision test — not just a demo that runs once.
