import torch

from .BaseEmbedder import BaseEmbedder
from deepface import DeepFace

models_deepface = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]


class DeepFaceWrapper(BaseEmbedder):
    """
    Wrapper around DeepFace library
    """

    def __init__(self, model_name: str):
        if not model_name in models_deepface:
            raise Exception('Backend not supported')
        self.backend = model_name

    def transform(self, img_path: str) -> torch.Tensor:
        embedding_objs = DeepFace.represent(img_path=img_path, model_name=self.backend, enforce_detection=False)
        return torch.tensor(embedding_objs[0]["embedding"])
