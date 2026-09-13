"""
GhostListing - FastAPI Backend
------------------------------------
Ties everything together into an actual running app:
  POST /listings/upload           -> submit a new listing, runs the graph up
                                       to the human-review pause (or straight
                                       through if clean)
  GET  /listings/pending           -> list of listings currently paused for review
  GET  /listings/resolved          -> recently decided listings (audit history)
  GET  /listings/{thread_id}/image -> serves the uploaded image for display
  POST /listings/{thread_id}/decide -> resume the graph with a moderator's decision

Run with:  uvicorn api:app --reload --port 8000
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sqlite3
import shutil
import json
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from embed import ListingImageEmbedder
from similarity_index import ListingSimilarityIndex
from graph import build_graph
import vision_check

app = FastAPI(title="GhostListing")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "data/audit_log.db"
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            thread_id TEXT PRIMARY KEY,
            listing_id TEXT,
            seller_id TEXT,
            listing_title TEXT,
            listing_description TEXT,
            listing_category TEXT,
            image_filename TEXT,
            flagged INTEGER,
            matches_json TEXT,
            consistency_verdict TEXT,
            consistency_reason TEXT,
            human_decision TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()

embedder = ListingImageEmbedder()
index = ListingSimilarityIndex(dim=512, index_path="data/listing_index")
if os.path.exists(f"{index.index_path}.faiss"):
    index.load()

graph = build_graph(
    embed_fn=embedder.embed_image,
    search_fn=index.find_similar,
    describe_fn=vision_check.describe_image,
    consistency_fn=vision_check.check_consistency,
)


@app.post("/listings/upload")
async def upload_listing(
    image: UploadFile,
    seller_id: str = Form(...),
    listing_title: str = Form(...),
    listing_description: str = Form(""),
    listing_category: str = Form(""),
):
    listing_id = f"{seller_id}_{image.filename}"
    save_path = os.path.join(UPLOAD_DIR, listing_id)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    thread_id = listing_id
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke({
        "listing_id": listing_id,
        "image_path": save_path,
        "seller_id": seller_id,
        "listing_title": listing_title,
        "listing_description": listing_description,
        "listing_category": listing_category,
    }, config)

    vec = embedder.embed_image(save_path)
    index.add_listing(vec, listing_id, seller_id)
    index.save()

    if result["flagged"]:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO audit_log "
            "(thread_id, listing_id, seller_id, listing_title, listing_description, "
            "listing_category, image_filename, flagged, matches_json, "
            "consistency_verdict, consistency_reason, human_decision) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (thread_id, listing_id, seller_id, listing_title, listing_description,
             listing_category, listing_id, 1, json.dumps(result["matches"]),
             result.get("consistency_verdict"), result.get("consistency_reason"), None),
        )
        conn.commit()
        conn.close()
        return {
            "status": "flagged_pending_review",
            "thread_id": thread_id,
            "matches": result["matches"],
            "consistency_verdict": result.get("consistency_verdict"),
            "consistency_reason": result.get("consistency_reason"),
        }

    return {"status": "clean", "thread_id": thread_id}


def _rows_to_dicts(rows):
    results = []
    for r in rows:
        row = dict(r)
        row["matches"] = json.loads(row["matches_json"]) if row["matches_json"] else []
        del row["matches_json"]
        results.append(row)
    return results


@app.get("/listings/pending")
async def get_pending():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM audit_log WHERE human_decision IS NULL ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return _rows_to_dicts(rows)


@app.get("/listings/resolved")
async def get_resolved():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM audit_log WHERE human_decision IS NOT NULL "
        "ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return _rows_to_dicts(rows)


@app.get("/listings/{thread_id}/image")
async def get_image(thread_id: str):
    path = os.path.join(UPLOAD_DIR, thread_id)
    return FileResponse(path)


@app.post("/listings/{thread_id}/decide")
async def decide(thread_id: str, decision: str = Form(...)):
    """decision should be 'confirmed_fraud' or 'false_positive'"""
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {"human_decision": decision})
    graph.invoke(None, config)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE audit_log SET human_decision = ? WHERE thread_id = ?",
        (decision, thread_id),
    )
    conn.commit()
    conn.close()
    return {"status": "recorded", "decision": decision}
