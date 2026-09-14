"""
GhostListing - Larger Evaluation Dataset Builder
----------------------------------------------------
Extracts a much bigger, harder test set than the original 75-image demo:
  - 5 sellers x 60 images each = 300 base listings (5 different product
    categories, so cross-seller "similar but not identical" items exist
    naturally - these are your hard negatives for false-positive testing)
  - 15 EXACT duplicate pairs injected across different sellers (easy positives)
  - 10 MANIPULATED duplicate pairs - the same photo but cropped, recolored,
    or recompressed before being reused by a different seller (hard
    positives - this is what a real photo thief would actually do, not
    just re-upload the identical file)

Every injected pair is recorded in ground_truth.json so evaluate.py can
compute real precision/recall against known answers, not guesses.

Usage:
    python extract_dataset_eval.py
"""

import os
import json
import random
import shutil
import tensorflow as tf
from PIL import Image, ImageEnhance
import io

DATASET_DIR = r"C:\Users\TOP WAY\Downloads\archive (2)\stanford_online_products\1.0.0"
OUTPUT_DIR = "data/eval_listings"
GROUND_TRUTH_PATH = "data/eval_ground_truth.json"

IMAGES_PER_SELLER = 60
SELLER_CATEGORY_MAP = {
    "seller_a": 0,
    "seller_b": 1,
    "seller_c": 2,
    "seller_d": 3,
    "seller_e": 4,
}

N_EXACT_DUPLICATES = 15
N_MANIPULATED_DUPLICATES = 10

feature_description = {
    'class_id': tf.io.FixedLenFeature([], tf.int64),
    'super_class_id': tf.io.FixedLenFeature([], tf.int64),
    'super_class_id/num': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
}


def _parse(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def manipulate_image(image_bytes: bytes, method: str) -> bytes:
    """Simulates what a real photo thief does: crop, recolor, or recompress
    a stolen photo rather than reuse the identical file - this is the
    actually-hard version of the detection problem."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if method == "crop":
        w, h = img.size
        img = img.crop((int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92)))
        img = img.resize((w, h))
    elif method == "recolor":
        img = ImageEnhance.Color(img).enhance(0.6)
        img = ImageEnhance.Brightness(img).enhance(1.15)
    elif method == "recompress":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=25)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    out = io.BytesIO()
    img.save(out, "JPEG", quality=90)
    return out.getvalue()


def main():
    print(f"Loading dataset from: {DATASET_DIR}")
    for seller_id in SELLER_CATEGORY_MAP:
        os.makedirs(os.path.join(OUTPUT_DIR, seller_id), exist_ok=True)

    counts = {seller_id: 0 for seller_id in SELLER_CATEGORY_MAP}
    saved = {seller_id: [] for seller_id in SELLER_CATEGORY_MAP}

    shard_path = os.path.join(
        DATASET_DIR, "stanford_online_products-test.tfrecord-00000-of-00016"
    )
    raw_dataset = tf.data.TFRecordDataset(shard_path)
    parsed_dataset = raw_dataset.map(_parse)

    print("Scanning for matching categories...")
    for example in parsed_dataset:
        super_class_id = int(example["super_class_id"].numpy())
        target_seller = None
        for seller_id, wanted_class in SELLER_CATEGORY_MAP.items():
            if super_class_id == wanted_class and counts[seller_id] < IMAGES_PER_SELLER:
                target_seller = seller_id
                break
        if target_seller is None:
            if all(c >= IMAGES_PER_SELLER for c in counts.values()):
                break
            continue

        image_bytes = example["image"].numpy()
        filename = f"item_{counts[target_seller]:03d}.jpg"
        save_path = os.path.join(OUTPUT_DIR, target_seller, filename)
        with open(save_path, "wb") as f:
            f.write(image_bytes)
        saved[target_seller].append((save_path, image_bytes))
        counts[target_seller] += 1

    print("\nBase extraction complete:")
    for seller_id, count in counts.items():
        print(f"  {seller_id}: {count} images")

    ground_truth = {}
    random.seed(7)
    seller_ids = list(SELLER_CATEGORY_MAP.keys())

    def listing_id_for(seller, filename):
        return f"{seller}_{filename}"

    print("\nInjecting exact duplicate pairs...")
    for i in range(N_EXACT_DUPLICATES):
        src_seller = random.choice(seller_ids)
        dst_seller = random.choice([s for s in seller_ids if s != src_seller])
        src_path, src_bytes = random.choice(saved[src_seller])
        src_filename = os.path.basename(src_path)

        dst_filename = f"exact_dup_{i:02d}.jpg"
        dst_path = os.path.join(OUTPUT_DIR, dst_seller, dst_filename)
        shutil.copy(src_path, dst_path)

        src_id = listing_id_for(src_seller, src_filename)
        dst_id = listing_id_for(dst_seller, dst_filename)
        ground_truth[dst_id] = {"duplicate_of": src_id, "method": "exact", "is_fraud": True}
        print(f"  {dst_id} <- exact copy of {src_id}")

    print("\nInjecting manipulated duplicate pairs (harder cases)...")
    methods = ["crop", "recolor", "recompress"]
    for i in range(N_MANIPULATED_DUPLICATES):
        src_seller = random.choice(seller_ids)
        dst_seller = random.choice([s for s in seller_ids if s != src_seller])
        src_path, src_bytes = random.choice(saved[src_seller])
        src_filename = os.path.basename(src_path)
        method = methods[i % len(methods)]

        manipulated_bytes = manipulate_image(src_bytes, method)
        dst_filename = f"manip_{method}_{i:02d}.jpg"
        dst_path = os.path.join(OUTPUT_DIR, dst_seller, dst_filename)
        with open(dst_path, "wb") as f:
            f.write(manipulated_bytes)

        src_id = listing_id_for(src_seller, src_filename)
        dst_id = listing_id_for(dst_seller, dst_filename)
        ground_truth[dst_id] = {"duplicate_of": src_id, "method": method, "is_fraud": True}
        print(f"  {dst_id} <- {method}-manipulated copy of {src_id}")

    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2)

    total_images = sum(counts.values()) + N_EXACT_DUPLICATES + N_MANIPULATED_DUPLICATES
    print(f"\nDone. {total_images} total images, {len(ground_truth)} known fraud cases.")
    print(f"Ground truth saved to {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
