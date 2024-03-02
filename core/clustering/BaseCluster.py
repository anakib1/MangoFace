import torch


class BaseClusterModel:
    """
    Base class for clustering models.
    """

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts cluster assignments given embeddings.

        :param X: [N, D] tensor
        :return: [N, K] LongTensor of class id.
        """
        pass
