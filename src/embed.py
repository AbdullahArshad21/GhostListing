"""
GhostListing - Image Embedding Module
--------------------------------------
Run this in Google Colab (free T4 GPU) or Kaggle Notebooks (free GPU) -
model weights download from HuggingFace/OpenCLIP's hub, which those
platforms can reach freely.

Model: ViT-B-32 pretrained on LAION-2B (via open_clip) - free, open-source,
~600MB download, runs fine on CPU too (just slower).
"""

import open_clip
import torch
from PIL import Image
import numpy as np


class ListingImageEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ViT-B-32 is the standard free/open starting point - good balance of
        # speed and quality. Swap for ViT-L-14 later if you want better
        # accuracy at the cost of speed once the MVP works.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalize
        return embedding.cpu().numpy().astype("float32").flatten()

    @torch.no_grad()
    def embed_batch(self, image_paths: list[str]) -> np.ndarray:
        """Batch version - much faster on GPU for building the initial index."""
        images = [self.preprocess(Image.open(p).convert("RGB")) for p in image_paths]
        batch = torch.stack(images).to(self.device)
        embeddings = self.model.encode_image(batch)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype("float32")


if __name__ == "__main__":
    # Quick smoke test once you have a few sample images downloaded
    embedder = ListingImageEmbedder()
    print(f"Running on: {embedder.device}")
    # vec = embedder.embed_image("sample_listing.jpg")
    # print("Embedding shape:", vec.shape)  # should be (512,)
