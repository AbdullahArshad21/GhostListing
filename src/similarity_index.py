"""
GhostListing - Similarity Search Index
----------------------------------------
Free, open-source, runs entirely on CPU - no paid vector DB needed.
This part of the pipeline is already tested and working (see test run below).
"""

import faiss
import numpy as np
import json
import os


class ListingSimilarityIndex:
    """
    Wraps a FAISS HNSW index (good default: no training step needed, fast
    approximate search, scales to hundreds of thousands of listings on a
    single machine for free before you'd ever need a paid vector DB).
    """

    def __init__(self, dim: int = 512, index_path: str = "data/listing_index"):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.id_map = []  # position in FAISS index -> listing metadata

    def add_listing(self, embedding: np.ndarray, listing_id: str, seller_id: str):
        embedding = embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
        self.id_map.append({"listing_id": listing_id, "seller_id": seller_id})

    def find_similar(self, embedding: np.ndarray, k: int = 5, exclude_seller: str = None):
        """
        Returns the top-k most visually similar listings, excluding matches
        from the same seller (a seller's own photo reused across their own
        listings isn't fraud - it's normal).
        """
        query = embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, k + 5)  # over-fetch to allow filtering

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.id_map[idx]
            if exclude_seller and meta["seller_id"] == exclude_seller:
                continue
            results.append({**meta, "l2_distance": float(dist)})
            if len(results) >= k:
                break
        return results

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, f"{self.index_path}.faiss")
        with open(f"{self.index_path}_meta.json", "w") as f:
            json.dump(self.id_map, f)

    def load(self):
        self.index = faiss.read_index(f"{self.index_path}.faiss")
        with open(f"{self.index_path}_meta.json") as f:
            self.id_map = json.load(f)


def _self_test():
    """Proves the pipeline logic works before you ever touch a real image."""
    np.random.seed(42)
    idx = ListingSimilarityIndex(dim=512, index_path="data/test_index")

    # Simulate 20 listings from 5 different sellers
    embeddings = {}
    for i in range(20):
        emb = np.random.randn(512).astype("float32")
        embeddings[i] = emb
        idx.add_listing(emb, listing_id=f"listing_{i}", seller_id=f"seller_{i % 5}")

    # Simulate seller_4 stealing seller_0's photo for listing_0 (tiny noise = "recompressed" copy)
    stolen_photo = embeddings[0] + np.random.randn(512).astype("float32") * 0.01

    matches = idx.find_similar(stolen_photo, k=3, exclude_seller="seller_4")
    print("Flagged matches for suspected stolen photo:")
    for m in matches:
        print(f"  {m['listing_id']} (seller: {m['seller_id']}) - distance: {m['l2_distance']:.4f}")

    assert matches[0]["listing_id"] == "listing_0", "Should have found the true source listing"
    print("\nSelf-test passed: correctly traced the stolen photo back to its original listing.")


if __name__ == "__main__":
    _self_test()
