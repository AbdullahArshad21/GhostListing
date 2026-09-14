"""
GhostListing - Real Evaluation: Precision/Recall/Threshold Tuning
----------------------------------------------------------------------
Runs the actual CLIP+FAISS pipeline against the larger eval dataset built
by extract_dataset_eval.py, and computes real metrics against known
ground truth - not eyeballing a handful of print statements.

Usage:
    python evaluate.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import json
from embed import ListingImageEmbedder
from similarity_index import ListingSimilarityIndex

EVAL_DIR = "data/eval_listings"
GROUND_TRUTH_PATH = "data/eval_ground_truth.json"
THRESHOLDS = [round(t, 2) for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]]


def collect_listings():
    listings = []
    for seller_folder in glob.glob(os.path.join(EVAL_DIR, "*")):
        seller_id = os.path.basename(seller_folder)
        for img_path in glob.glob(os.path.join(seller_folder, "*.jpg")):
            filename = os.path.basename(img_path)
            listing_id = f"{seller_id}_{filename}"
            listings.append({"listing_id": listing_id, "seller_id": seller_id, "image_path": img_path})
    return listings


def main():
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    listings = collect_listings()
    print(f"Loaded {len(listings)} listings ({len(ground_truth)} known fraud cases: "
          f"{sum(1 for v in ground_truth.values() if v['method']=='exact')} exact, "
          f"{sum(1 for v in ground_truth.values() if v['method']!='exact')} manipulated)")

    embedder = ListingImageEmbedder()
    print(f"Embedding on: {embedder.device}")

    index = ListingSimilarityIndex(dim=512, index_path="data/eval_index")
    embeddings_cache = {}

    print("Embedding and indexing all listings...")
    for i, listing in enumerate(listings):
        vec = embedder.embed_image(listing["image_path"])
        embeddings_cache[listing["listing_id"]] = vec
        index.add_listing(vec, listing["listing_id"], listing["seller_id"])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(listings)} embedded")

    print("\nQuerying nearest cross-seller match for every listing...")
    nearest_distance = {}  # listing_id -> distance to nearest cross-seller match
    for listing in listings:
        vec = embeddings_cache[listing["listing_id"]]
        matches = index.find_similar(vec, k=1, exclude_seller=listing["seller_id"])
        nearest_distance[listing["listing_id"]] = matches[0]["l2_distance"] if matches else float("inf")

    fraud_ids = set(ground_truth.keys())
    all_ids = set(l["listing_id"] for l in listings)
    clean_ids = all_ids - fraud_ids

    # Map listing_id -> image_path so downstream scripts (stage2) don't need
    # to re-derive it or re-run CLIP embedding at all
    path_by_id = {l["listing_id"]: l["image_path"] for l in listings}

    print(f"\n{'Threshold':>10} | {'Precision':>10} | {'Recall':>8} | {'F1':>6} | {'FP Rate':>8} | TP/FP/FN/TN")
    print("-" * 80)

    best_f1 = -1
    best_threshold = None
    results_by_threshold = {}

    for threshold in THRESHOLDS:
        tp = fp = fn = tn = 0
        for lid in fraud_ids:
            if nearest_distance[lid] < threshold:
                tp += 1
            else:
                fn += 1
        for lid in clean_ids:
            if nearest_distance[lid] < threshold:
                fp += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        results_by_threshold[threshold] = {"precision": precision, "recall": recall, "f1": f1,
                                            "tp": tp, "fp": fp, "fn": fn, "tn": tn}

        print(f"{threshold:>10.2f} | {precision:>10.3f} | {recall:>8.3f} | {f1:>6.3f} | "
              f"{fp_rate:>8.3f} | {tp}/{fp}/{fn}/{tn}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nBest threshold by F1 score: {best_threshold} (F1={best_f1:.3f})")
    r = results_by_threshold[best_threshold]
    print(f"  At this threshold: precision={r['precision']:.3f}, recall={r['recall']:.3f}, "
          f"{r['tp']} true positives, {r['fp']} false positives, {r['fn']} missed frauds")

    print(f"\n--- Breakdown by manipulation method at threshold={best_threshold} ---")
    method_stats = {}
    for lid, info in ground_truth.items():
        method = info["method"]
        detected = nearest_distance[lid] < best_threshold
        method_stats.setdefault(method, {"total": 0, "detected": 0})
        method_stats[method]["total"] += 1
        if detected:
            method_stats[method]["detected"] += 1

    for method, stats in sorted(method_stats.items()):
        rate = stats["detected"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {method:>12}: {stats['detected']}/{stats['total']} detected ({rate:.1%})")

    output = {
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "results_by_threshold": results_by_threshold,
        "method_breakdown": method_stats,
        "flagged_fraud_ids": [lid for lid in fraud_ids if nearest_distance[lid] < best_threshold],
        "flagged_false_positive_ids": [lid for lid in clean_ids if nearest_distance[lid] < best_threshold],
        "path_by_id": path_by_id,
    }
    with open("data/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nFull results saved to data/eval_results.json")


if __name__ == "__main__":
    main()
