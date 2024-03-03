from typing import Union, List
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random
from tsnecuda import TSNE
import numpy as np


class DrawingUtility:

    def __init__(self, embeddings:torch.Tensor, predictions: torch.Tensor, filenames: List[str]):
        """
        Initializes visualisation module.
        :param embeddings:
        :param predictions:
        :param filenames:
        """
        self.cluster_to_id = None
        self.embeddings = embeddings
        self.predictions = predictions
        self.filenames = filenames
        self._generate_cluster_to_id()

    def _generate_cluster_to_id(self):
        cluster_to_id = {}
        for i, x in enumerate(self.predictions):
            x = int(x)
            if x not in cluster_to_id:
                cluster_to_id[x] = []
            cluster_to_id[x].append(i)
        self.cluster_to_id = cluster_to_id

    def draw_one_cluster(self, cluster_id: Union[int, None]):
        """
        Draws examples from one cluster.
        :param cluster_id:
        :return:
        """
        if cluster_id is None:
            cluster_id = random.choice(self.cluster_to_id.keys())

        choices = random.choices(self.cluster_to_id[cluster_id], k=16)

        fig, ax = plt.subplots(4, 4)
        for i, x in enumerate(choices):
            img = Image.open(f'{self.filenames[x]}')
            ax[i // 4][i % 4].imshow(img)

    def draw_all_clusters(self, max_images=None):
        """
        Draws at most some amount of images at TSNE-based plot.
        :param max_images: maximum number of images drawn. Optional.
        :return:
        """
        embeddings = self.embeddings
        filenames = self.filenames
        if max_images is not None:
            embeddings, filenames = embeddings[:max_images], filenames[:max_images]

        projections = TSNE(n_components=2, perplexity=25, learning_rate=10).fit_transform(torch.stack(embeddings))

        tx, ty = projections[:, 0], projections[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        width = 4000
        height = 3000
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        for img, x, y in zip(names, tx, ty):
            tile = Image.open(img)
            rs = max(1, tile.width / max_dim, tile.height / max_dim)
            tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
            full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))

        plt.figure(figsize=(16, 12))
        plt.imshow(full_image)
        plt.axis('off')
