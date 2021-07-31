import torch
import torch.nn as nn
from torch_prototypes.metrics.distortion import Eucl_Mat, Cosine_Mat


class RankLoss(nn.Module):
    """Rank preserving loss for finite metric embedding"""

    def __init__(self, D, n_triplets, dist="eucl", ignore_index=None):
        """
        Args:
            D (tensor): 2D cost matrix of the finite metric to embed
            n_triplets (int): Number of triplets over which to compute the loss at each iteration
            dist (str): Euclidean or Cosine distance for the target embedding space (eucl/cos)
            ignore_index (int): index of the label to be ignore (if any)
        """
        super(RankLoss, self).__init__()
        self.D = D
        self.n_triplets = n_triplets
        if dist == "eucl":
            self.dist = Eucl_Mat()
        elif dist == "cos":
            self.dist = Cosine_Mat()
        self.ignore_index = ignore_index
        if self.ignore_index is not None:
            self.idxs = torch.tensor(
                [i for i in range(D.shape[0]) if i != ignore_index], device=D.device
            ).long()

    def forward(self, prototypes, idxs=None):
        if self.ignore_index is None:
            i, j, k = torch.stack(
                [torch.randperm(self.D.shape[0])[:3] for _ in range(self.n_triplets)],
                dim=1,
            )
        else:
            i, j, k = torch.stack(
                [
                    torch.randperm(self.idxs.shape[0])[:3]
                    for _ in range(self.n_triplets)
                ],
                dim=1,
            )
            i = self.idxs[i]
            j = self.idxs[j]
            k = self.idxs[k]

        S_hat_ijk = (self.D[i, j] > self.D[i, k]).float()

        dists = self.dist(prototypes)
        diff = dists[i, j] - dists[i, k]
        log_Sijk = -torch.log(1 + torch.exp(-diff))
        log_1_Sijk = -torch.log(1 + torch.exp(diff))

        l = S_hat_ijk * log_Sijk + (1 - S_hat_ijk) * log_1_Sijk
        l = -l.mean()
        return l
