import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from .BaseEmbedder import BaseEmbedder

my_embedding = None


def copy_data(m, i, o):
    global my_embedding
    my_embedding = o.data.flatten().cpu()


class ResNetEmbedder(BaseEmbedder):
    def __init__(self):
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.to('cuda')
        model.eval()
        preprocess = weights.transforms()
        model.get_submodule('avgpool').register_forward_hook(copy_data)

        self.model, self.preprocess = model, preprocess

    def transform(self, img_path):
        global my_embedding
        img_tensor = read_image(img_path)
        with torch.no_grad():
            batch = self.preprocess(img_tensor).unsqueeze(0).cuda()
            self.model(batch).squeeze(0).softmax(0)
            return my_embedding
