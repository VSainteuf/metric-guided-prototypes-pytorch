import torch
import torch.nn as nn


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
    def __init__(self, D, dist='euclidian'):
        """
        Args:
            D (tensor): 2D cost matrix of the finite metric, shape (NxN)
            dist: Distance to use in the target embedding space (euclidean or cosine)
        """
        super(Distortion, self).__init__()
        self.D = D
        if dist == 'euclidian':
            self.dist = Eucl_Mat()
        elif dist == 'cosine':
            self.dist = Cosine_Mat()
    def forward(self, mapping, idxs=None):
        """
            mapping (tensor):  Tensor of shape (N x Embedding_dimension) giving the mapping to the target metric space
        """
        d = self.dist(mapping)
        d = (d - self.D).abs() / (self.D + torch.eye(self.D.shape[0], device=self.D.device))
        d = d.sum() / (d.shape[0] ** 2 - d.shape[0])
        return d

class DistortionLoss(nn.Module):
    """Squared distortion to serve as a regularizer"""
    def __init__(self, D, dist='euclidian'):
        super(DistortionLoss, self).__init__()
        self.D = D
        if dist == 'euclidian':
            self.dist = Eucl_Mat()
        elif dist == 'cosine':
            self.dist = Cosine_Mat()
    def forward(self, mapping, idxs=None):
        d = self.dist(mapping)
        d = (d - self.D) ** 2 / (self.D + torch.eye(self.D.shape[0], device=self.D.device)) ** 2
        d = d.sum() / (d.shape[0] ** 2 - d.shape[0])
        return d
