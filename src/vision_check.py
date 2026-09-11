"""
GhostListing - Vision Consistency Check (Groq, free tier)
------------------------------------------------------------
Given two images FAISS flagged as visually similar, this asks a vision
model to describe each one, then asks a second (text-only) call to judge
whether a listing's claimed text/category is consistent with what's
actually in its photo.

Get a free API key at https://console.groq.com (no card required).
Store it in a .env file at the project root as: GROQ_API_KEY=your-key-here
"""
import re
import os
import base64
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

VISION_MODEL = "qwen/qwen3.6-27b"
TEXT_MODEL = "openai/gpt-oss-20b"


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(image_path: str) -> str:
    """Ask the vision model what's actually in the photo."""
    b64_image = _encode_image(image_path)
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Describe this product photo factually in 2-3 sentences: "
                        "what the item is, its visible condition (new/used/damaged), "
                        "brand if visible, and any notable features. Be literal - "
                        "don't guess at things you can't actually see."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_image}"
                    }},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=300,
        reasoning_effort="none",
        reasoning_format="hidden",
    )
    content = response.choices[0].message.content
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content

def check_consistency(image_description: str, listing_title: str,
                       listing_description: str, listing_category: str) -> dict:
    """
    LLM reasoning step: does the listing's claimed text match what's
    actually visible in the photo? This is judgment, not similarity search -
    hence a text-reasoning LLM call rather than another embedding comparison.
    """
    prompt = f"""You are checking a marketplace listing for consistency between
its claimed text and what its photo actually shows.

PHOTO SHOWS: {image_description}

LISTING CLAIMS:
- Title: {listing_title}
- Description: {listing_description}
- Category: {listing_category}

Answer in this exact format:
VERDICT: [CONSISTENT / INCONSISTENT / UNCERTAIN]
REASON: [one sentence explaining why]

Flag as INCONSISTENT only for clear, factual mismatches (e.g. photo shows
a used item but listing claims "brand new", or the category doesn't match
the item type). Do not flag minor subjective differences in wording."""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        reasoning_effort="none",
        reasoning_format="hidden",
    )
    text = response.choices[0].message.content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    verdict = "UNCERTAIN"
    reason = text
    for line in text.splitlines():
        if line.startswith("VERDICT:"):
            verdict = line.replace("VERDICT:", "").strip()
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return {"verdict": verdict, "reason": reason, "raw": text}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        desc = describe_image(sys.argv[1])
        print("Image description:", desc)
    else:
        print("Usage: python vision_check.py path/to/image.jpg")