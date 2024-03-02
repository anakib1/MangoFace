import torch


class BaseEmbedder:
    def transform(self, img_path: str) -> torch.Tensor:
        """
        Transforms image located at path to embedding
        :param img_path: string with path to image
        :return: embedding of image - torch.Tesnor of shape (D, )
        """
        pass
