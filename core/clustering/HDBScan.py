import torch

from .BaseCluster import BaseClusterModel
from cuml.cluster import HDBSCAN
import numpy as np


class HDbScanClusterModel(BaseClusterModel):
    """
    Class for clustering based on HDBScan
    """

    def __init__(self, alpha=5, min_cluster_size=25, max_cluster_size=500, metric='l2'):
        """
        View HDBSCAN paramteres for details
        :param alpha:
        :param min_cluster_size: minimum cluster size
        :param max_cluster_size: maximum cluster size
        :param metric: clustering metric
        """
        self.hdbscan_float = HDBSCAN(alpha=alpha, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
                                     metric=metric)

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        labels = self.hdbscan_float.fit_predict(X.numpy().astype(np.float64))
        return torch.Tensor(labels)
