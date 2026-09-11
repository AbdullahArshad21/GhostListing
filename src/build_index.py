"""
GhostListing - Build & Query Index on Real Images
----------------------------------------------------
Run this AFTER you've downloaded sample images into data/listings/<seller_name>/*.jpg

Usage:
    python src/build_index.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import glob
from embed import ListingImageEmbedder
from similarity_index import ListingSimilarityIndex


def collect_listings(data_dir="data/listings"):
    """
    Expects a folder structure like:
        data/listings/seller_a/photo1.jpg
        data/listings/seller_a/photo2.jpg
        data/listings/seller_b/photo1.jpg
    """
    listings = []
    for seller_folder in glob.glob(os.path.join(data_dir, "*")):
        seller_id = os.path.basename(seller_folder)
        for img_path in glob.glob(os.path.join(seller_folder, "*.jpg")) + \
                         glob.glob(os.path.join(seller_folder, "*.png")):
            listing_id = os.path.splitext(os.path.basename(img_path))[0]
            listings.append({
                "seller_id": seller_id,
                "listing_id": f"{seller_id}_{listing_id}",
                "image_path": img_path,
            })
    return listings


def main():
    listings = collect_listings()
    if not listings:
        print("No images found. Put some .jpg/.png files under data/listings/<seller_name>/")
        return

    print(f"Found {len(listings)} listing images across "
          f"{len(set(l['seller_id'] for l in listings))} sellers.")

    embedder = ListingImageEmbedder()
    print(f"Loaded CLIP on: {embedder.device}")

    index = ListingSimilarityIndex(dim=512, index_path="data/listing_index")

    for listing in listings:
        vec = embedder.embed_image(listing["image_path"])
        index.add_listing(vec, listing["listing_id"], listing["seller_id"])
        print(f"  embedded: {listing['listing_id']}")

    index.save()
    print("\nIndex built and saved to data/listing_index.faiss")

    # Now check every listing against every other listing for cross-seller duplicates
    print("\n--- Scanning for cross-seller duplicate photos ---")
    found_any = False
    for listing in listings:
        vec = embedder.embed_image(listing["image_path"])
        matches = index.find_similar(vec, k=3, exclude_seller=listing["seller_id"])
        # A very tight distance threshold = near-identical image; tune this
        # once you've eyeballed real scores on your own sample data
        flagged = [m for m in matches if m["l2_distance"] < 0.3]
        if flagged:
            found_any = True
            print(f"\n⚠ Possible stolen photo: {listing['listing_id']}")
            for m in flagged:
                print(f"    matches {m['listing_id']} (distance: {m['l2_distance']:.4f})")

    if not found_any:
        print("No cross-seller duplicates found in this sample set — "
              "try deliberately copying one image into a different seller's "
              "folder to test that the detector actually catches it.")


if __name__ == "__main__":
    main()
