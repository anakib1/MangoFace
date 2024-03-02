from .BaseEmbedder import BaseEmbedder
import torch
import clip
from PIL import Image


class ClipEmbedder(BaseEmbedder):
    def __init__(self, clip_backbone="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_backbone, device=self.device)

    def transform(self, img_path):
        image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            return image_features.flatten().cpu()
