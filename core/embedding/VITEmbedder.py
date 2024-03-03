import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms.functional as vision_F
from .BaseEmbedder import BaseEmbedder
from PIL import Image

vit_embedding = None


def vit_hook(m, i, o):
    global vit_embedding
    vit_embedding = o.data.flatten().cpu()


class VitEmbedderModel(BaseEmbedder):
    def __init__(self):
        weights_config = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        self.processor = weights_config.transforms()
        self.model = vit_b_16(weights_config).to("cuda")
        self.model.get_submodule("encoder").register_forward_hook(vit_hook)

    def transform(self, img_path: str) -> torch.Tensor:
        global vit_embedding
        with torch.no_grad():
            image = Image.open(img_path)
            image_tensor = self.processor(vision_F.pil_to_tensor(image)).unsqueeze(0).to("cuda")

            self.model(image_tensor)
            return vit_embedding
