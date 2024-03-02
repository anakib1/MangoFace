from core.clustering import BaseCluster
from core.embedding import BaseEmbedder
from tqdm.auto import tqdm
from glob import glob
import torch
from typing import List, Tuple


class ClusterizationPipeline:
    def __init__(self, embedder: BaseEmbedder, clustering: BaseCluster):
        self.embedder = embedder
        self.clustering = clustering

    def cluster(self, img_folder: str) -> Tuple[List[str], torch.Tensor]:
        """
        Cluster all images in the image folder into clusters according to clustering provided.
        :param img_folder: path to iamge folder. Only jpg images will be used
        :return: tensor with cluster ids
        """
        names = []
        embeddings = []
        for img in tqdm(glob(img_folder + '/**/*.jpg', recursive=True)):
            names.append(img)
            embeddings.append(self.embedder.transform(img))

        return names, self.clustering.fit_predict(torch.stack(embeddings))
