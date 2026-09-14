"""
GhostListing - Stage 2 Evaluation: Consistency Check Accuracy
--------------------------------------------------------------------
IMPORTANT SCOPE NOTE (read this before trusting the numbers):

This does NOT test whether the consistency check can tell "these are two
different items" apart (FAISS's job, and it can't - the consistency check
only ever looks at ONE listing's own text vs its own photo, never compares
two listings to each other). It tests a narrower, more honest question:

  "When a fraud case is misrepresented (photo doesn't match its claimed
   condition/category), does the consistency check catch it? And when a
   listing's text truthfully matches its own photo, does it correctly
   clear it?"

To test this fairly:
  - For every FAISS-flagged FRAUD case (a real duplicate/stolen photo),
    this script fabricates a deliberately misleading listing claim
    ("brand new, sealed, never used") - modeling the realistic thief
    behavior we actually observed in production (seller_c's real listing
    made exactly this false claim).
  - For every FAISS-flagged FALSE POSITIVE (a genuinely different but
    visually similar item), this script builds an honest listing
    description directly FROM the vision model's own description of that
    photo - modeling a legitimate seller accurately describing their item.

Run extract_dataset_eval.py and evaluate.py first - this script reuses
their output rather than re-running CLIP.

Usage:
    python stage2_evaluate.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import vision_check

RESULTS_PATH = "data/eval_results.json"
GROUND_TRUTH_PATH = "data/eval_ground_truth.json"


def fabricate_deceptive_listing():
    """Models the exact real-world lie we saw in production testing:
    claiming a reused/stolen photo shows a brand-new, sealed item."""
    return {
        "listing_title": "Brand New - Sealed, Never Used",
        "listing_description": "Brand new condition, never installed, still sealed in original packaging.",
        "listing_category": "General",
    }


def build_honest_listing(image_description: str):
    """Models a legitimate seller whose listing text is just an accurate
    reflection of their own photo - the honest baseline case."""
    return {
        "listing_title": "Item for sale - see photo",
        "listing_description": image_description,
        "listing_category": "General",
    }


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    fraud_ids = results["flagged_fraud_ids"]
    false_positive_ids = results["flagged_false_positive_ids"]
    path_by_id = results["path_by_id"]

    # Resume support: if a previous run got partway through (e.g. hit a
    # rate limit), don't re-spend quota re-answering cases we already have.
    outcomes = []
    already_done = set()
    output_path = "data/stage2_eval_results.json"
    if os.path.exists(output_path):
        with open(output_path) as f:
            prev = json.load(f)
        outcomes = prev.get("outcomes", [])
        already_done = {o["listing_id"] for o in outcomes}
        if already_done:
            print(f"Resuming: {len(already_done)} cases already answered in a previous run, skipping those.\n")

    print(f"Testing consistency check against {len(fraud_ids)} flagged fraud cases "
          f"(given a fabricated deceptive claim) and {len(false_positive_ids)} flagged "
          f"false positives (given an honest, photo-accurate claim).\n")

    def process(listing_id, is_fraud):
        if listing_id in already_done:
            return
        image_path = path_by_id[listing_id]
        description = vision_check.describe_image(image_path)

        if is_fraud:
            listing_text = fabricate_deceptive_listing()
        else:
            listing_text = build_honest_listing(description)

        result = vision_check.check_consistency(
            description,
            listing_text["listing_title"],
            listing_text["listing_description"],
            listing_text["listing_category"],
        )
        predicted_fraud = result["verdict"] in ("INCONSISTENT", "UNCERTAIN")
        outcomes.append({
            "listing_id": listing_id,
            "is_fraud": is_fraud,
            "verdict": result["verdict"],
            "predicted_fraud": predicted_fraud,
            "reason": result["reason"],
        })
        status = "correct" if predicted_fraud == is_fraud else "WRONG"
        print(f"  [{status:7}] {listing_id:40} true_fraud={is_fraud!s:5} verdict={result['verdict']:12} "
              f"reason={result['reason'][:70]}")

        # Save after every single case - if a rate limit or any other error
        # interrupts us, nothing already answered is lost.
        with open(output_path, "w") as f:
            json.dump({"outcomes": outcomes}, f, indent=2)

    print("--- Fraud cases (fabricated deceptive claims) ---")
    for lid in fraud_ids:
        try:
            process(lid, is_fraud=True)
        except Exception as e:
            print(f"\nStopped early due to an error ({e}). "
                  f"{len(outcomes)} cases saved - rerun this script later to resume.")
            return
        time.sleep(0.3)

    print("\n--- False positive cases (honest, photo-accurate claims) ---")
    for lid in false_positive_ids:
        try:
            process(lid, is_fraud=False)
        except Exception as e:
            print(f"\nStopped early due to an error ({e}). "
                  f"{len(outcomes)} cases saved - rerun this script later to resume.")
            return
        time.sleep(0.3)

    tp = sum(1 for o in outcomes if o["is_fraud"] and o["predicted_fraud"])
    fn = sum(1 for o in outcomes if o["is_fraud"] and not o["predicted_fraud"])
    fp = sum(1 for o in outcomes if not o["is_fraud"] and o["predicted_fraud"])
    tn = sum(1 for o in outcomes if not o["is_fraud"] and not o["predicted_fraud"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n--- Results ---")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"\nInterpretation: of the deceptive claims, {tp}/{tp+fn} were caught. "
          f"Of the honest claims, {tn}/{tn+fp} were correctly cleared.")

    with open("data/stage2_eval_results.json", "w") as f:
        json.dump({"precision": precision, "recall": recall, "f1": f1,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn, "outcomes": outcomes}, f, indent=2)
    print("\nFull results saved to data/stage2_eval_results.json")


if __name__ == "__main__":
    main()
