import torch
import torch.nn as nn
import numpy as np


class Eucl_Mat(nn.Module):
    """Pairwise Euclidean distance"""

    def __init_(self):
        super(Eucl_Mat, self).__init__()

    def forward(self, mapping):
        """
        Args:
            mapping (tensor): Tensor of shape N_vectors x Embedding_dimension
        Returns:
            distances: Tensor of shape N_vectors x N_vectors giving the pairwise Euclidean distances

        """
        return torch.norm(mapping[:, None, :] - mapping[None, :, :], dim=-1)


class Cosine_Mat(nn.Module):
    """Pairwise Cosine distance"""

    def __init__(self):
        super(Cosine_Mat, self).__init__()

    def forward(self, mapping):
        """
        Args:
            mapping (tensor): Tensor of shape N_vectors x Embedding_dimension
        Returns:
            distances: Tensor of shape N_vectors x N_vectors giving the pairwise Cosine distances

        """
        return 1 - nn.CosineSimilarity(dim=-1)(mapping[:, None, :], mapping[None, :, :])


class Pseudo_Huber(nn.Module):
    """Pseudo-Huber function"""

    def __init__(self, delta=1):
        super(Pseudo_Huber, self).__init__()
        self.delta = delta

    def forward(self, input):
        out = (input / self.delta) ** 2
        out = torch.sqrt(out + 1)
        out = self.delta * (out - 1)
        return out


class Distortion(nn.Module):
    """Distortion measure of the embedding of finite metric given by matrix D into another metric space"""

    def __init__(self, D, dist="euclidian"):
        """
        Args:
            D (tensor): 2D cost matrix of the finite metric, shape (NxN)
            dist: Distance to use in the target embedding space (euclidean or cosine)
        """
        super(Distortion, self).__init__()
        self.D = D
        if dist == "euclidian":
            self.dist = Eucl_Mat()
        elif dist == "cosine":
            self.dist = Cosine_Mat()

    def forward(self, mapping, idxs=None):
        """
        mapping (tensor):  Tensor of shape (N x Embedding_dimension) giving the mapping to the target metric space
        """
        d = self.dist(mapping)
        d = (d - self.D).abs() / (
            self.D + torch.eye(self.D.shape[0], device=self.D.device)
        )
        d = d.sum() / (d.shape[0] ** 2 - d.shape[0])
        return d


class ScaleFreeDistortion(nn.Module):
    def __init__(self, D):
        super(ScaleFreeDistortion, self).__init__()
        self.D = D
        self.disto = Distortion(D)
        self.dist = Eucl_Mat()

    def forward(self, prototypes):
        # Compute distance ratios
        d = self.em(prototypes)
        d = d / (self.D + torch.eye(self.D.shape[0], device=self.D.device))

        # Get sorted list of ratios
        alpha = d[d > 0].detach().cpu().numpy()
        alpha = np.sort(alpha)

        # Find optimal scaling
        cumul = np.cumsum(alpha)
        a_i = alpha[np.where(cumul >= alpha.sum() - cumul)[0].min()]
        scale = 1 / a_i

        return self.disto(scale * prototypes)


class DistortionLoss(nn.Module):
    """Scale-free squared distortion regularizer"""

    def __init__(self, D, dist="euclidian", scale_free=True):
        super(DistortionLoss, self).__init__()
        self.D = D
        self.scale_free = scale_free
        if dist == "euclidian":
            self.dist = Eucl_Mat()
        elif dist == "cosine":
            self.dist = Cosine_Mat()

    def forward(self, mapping, idxs=None):
        d = self.dist(mapping)

        if self.scale_free:
            a = d / (self.D + torch.eye(self.D.shape[0], device=self.D.device))
            scaling = a.sum() / torch.pow(a, 2).sum()
        else:
            scaling = 1.0

        d = (scaling * d - self.D) ** 2 / (
            self.D + torch.eye(self.D.shape[0], device=self.D.device)
        ) ** 2
        d = d.sum() / (d.shape[0] ** 2 - d.shape[0])
        return d
