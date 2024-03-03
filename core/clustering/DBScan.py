import torch

from .BaseCluster import BaseClusterModel
from cuml.cluster import DBSCAN
import numpy as np


class DbScanClusterModel(BaseClusterModel):
    """
    Class for clustering based on DBScan
    """

    def __init__(self, eps=0.2, min_samples=25, metric='cosine'):
        """

        :param eps: min dist inside cluster elements
        :param min_samples: min cluster samples
        :param metric: ['cosine'/'euclidian']
        """
        self.dbscan_float = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        labels = self.dbscan_float.fit_predict(X.numpy().astype(np.float64))
        return torch.Tensor(labels)
