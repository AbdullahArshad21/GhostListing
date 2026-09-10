"""
GhostListing - Extract real images from the downloaded Stanford Online
Products TFRecord dataset into seller folders for testing.

This reads the TFRecords directly (bypassing tfds.builder_from_directory,
which failed due to a version mismatch) since we now know the 'image'
field already contains ready-to-save JPEG bytes.
"""

import os
import glob
import shutil
import tensorflow as tf

DATASET_DIR = r"C:\Users\TOP WAY\Downloads\archive (2)\stanford_online_products\1.0.0"
OUTPUT_DIR = "data/listings"
IMAGES_PER_SELLER = 25

SELLER_CATEGORY_MAP = {
    "seller_a": 0,
    "seller_b": 1,
    "seller_c": 2,
}

feature_description = {
    'class_id': tf.io.FixedLenFeature([], tf.int64),
    'super_class_id': tf.io.FixedLenFeature([], tf.int64),
    'super_class_id/num': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
}


def _parse(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


def main():
    for seller_id in SELLER_CATEGORY_MAP:
        os.makedirs(os.path.join(OUTPUT_DIR, seller_id), exist_ok=True)

    counts = {seller_id: 0 for seller_id in SELLER_CATEGORY_MAP}
    saved_paths = {seller_id: [] for seller_id in SELLER_CATEGORY_MAP}

    # Just use the first shard of the test split - it alone has ~3781
    # examples, far more than we need
    shard_path = os.path.join(DATASET_DIR, "stanford_online_products-test.tfrecord-00000-of-00016")
    raw_dataset = tf.data.TFRecordDataset(shard_path)
    parsed_dataset = raw_dataset.map(_parse)

    print("Scanning first shard for matching categories...")
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

        image_bytes = example["image"].numpy()  # already valid JPEG bytes
        filename = f"item_{counts[target_seller]:03d}.jpg"
        save_path = os.path.join(OUTPUT_DIR, target_seller, filename)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        saved_paths[target_seller].append(save_path)
        counts[target_seller] += 1
        print(f"  saved {save_path}")

    print("\nExtraction complete:")
    for seller_id, count in counts.items():
        print(f"  {seller_id}: {count} images")

    if all(c > 1 for c in counts.values()):
        print("\nInjecting deliberate duplicate images as fraud test cases...")
        for i in range(2):
            src = saved_paths["seller_a"][i]
            dst = os.path.join(OUTPUT_DIR, "seller_b", f"stolen_from_a_{i}.jpg")
            shutil.copy(src, dst)
            print(f"  copied {src} -> {dst}")
    else:
        print("\nWarning: not enough images found in one or more categories - "
              "check that super_class_id values 0, 1, 2 actually appear in this shard.")

    print("\nDone.")


if __name__ == "__main__":
    main()