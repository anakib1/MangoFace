from typing import Union, List
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random
from tsnecuda import TSNE
import numpy as np


class DrawingUtility:

    def __init__(self, embeddings: torch.Tensor, predictions: torch.Tensor, filenames: List[str]):
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

        projections = TSNE(n_components=2, perplexity=25, learning_rate=10).fit_transform(embeddings)

        tx, ty = projections[:, 0], projections[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        width = 4000
        height = 3000
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        for img, x, y in zip(filenames, tx, ty):
            tile = Image.open(img)
            rs = max(1, tile.width / max_dim, tile.height / max_dim)
            tile = tile.resize((int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
            full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))

        plt.figure(figsize=(16, 12))
        plt.imshow(full_image)
        plt.axis('off')

    def draw_average(self):
        num_clusters = len(self.cluster_to_id)
        n_cols = (num_clusters + 3) // 4

        fig, ax = plt.subplots(4, n_cols)
        for i, k in enumerate(self.cluster_to_id.keys()):
            avg_image = []
            for img_id in self.cluster_to_id[k]:
                avg_image.append(np.array(Image.open(self.filenames[img_id]).resize((124, 124))))

            avg_image = np.array(avg_image).mean(axis=0).astype(np.int8)

            ax[i // n_cols][i % n_cols].imshow(avg_image)
            ax[i // n_cols][i % n_cols].axis('off')

    def draw_best(self):
        num_clusters = len(self.cluster_to_id)
        n_cols = (num_clusters + 3) // 4

        fig, ax = plt.subplots(4, n_cols)
        for i, k in enumerate(self.cluster_to_id.keys()):
            best_img = None
            bst_dist = np.inf
            for img_id in self.cluster_to_id[k]:
                embed1 = self.embeddings[img_id]
                distance = np.inf
                for other_img in self.cluster_to_id[k]:
                    embed2 = self.embeddings[other_img]
                    distance = max(distance,
                                   np.dot(embed1, embed2) / np.linalg.norm(embed1) / np.linalg.norm(embed2))

                if distance < bst_dist:
                    bst_dist = distance
                    best_img = img_id

            ax[i // n_cols][i % n_cols].imshow(Image.open(self.filenames[best_img]))
            ax[i // n_cols][i % n_cols].axis('off')
