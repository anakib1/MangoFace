import torch

from .BaseCluster import BaseClusterModel
from fast_pytorch_kmeans import KMeans

class KmeansClusterModel(BaseClusterModel):
    """
    Class for clustering based on KMeans
    """
    def __init__(self, n_clusters=8, mode='euclidean'):
        """

        :param n_clusters: number of clusters to fit to.
        :param mode: distance mode
        """
        self.n_clusters = n_clusters
        self.mode = mode
        self.kmeans = KMeans(n_clusters=n_clusters, mode=mode, verbose=1)

    def fit_predict(self, X:torch.Tensor) -> torch.Tensor:
        labels = self.kmeans.fit_predict(X)
        return labels
