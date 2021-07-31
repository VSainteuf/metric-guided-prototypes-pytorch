"""
Implementation of the methods proposed in Hyperspherical Prototype Network,  Mettes et al., NeurIPS 2019
https://arxiv.org/abs/1901.10514
"""

import torch
import torch.nn as nn


class SeparationLoss(nn.Module):
    """Large margin separation between hyperspherical protoypes"""

    def __init__(self):
        super(SeparationLoss, self).__init__()

    def forward(self, protos):
        """
        Args:
            protos (tensor): (N_prototypes x Embedding_dimension)
        """
        M = torch.matmul(protos, protos.transpose(0, 1)) - 2 * torch.eye(
            protos.shape[0]
        ).to(protos.device)
        return M.max(dim=1)[0].mean()


class HypersphericalLoss(nn.Module):
    """Training loss, minimizes the cosine distance between a samples embedding and its class prototype"""

    def __init__(self, ignore_label=None, class_weights=None):
        super(HypersphericalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.class_weights = class_weights

    def forward(self, scores, y):
        if len(scores.shape) == 4:  # Flatten 2D data
            b, c, h, w = scores.shape
            scores = (
                scores.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)
            )
            y_lab = y.view(b * h * w)
        else:
            y_lab = y
        loss = -scores.gather(dim=1, index=y_lab.long().view(-1, 1)).squeeze()
        if self.ignore_label is not None:
            loss = loss[y_lab != self.ignore_label]
            y_lab = y_lab[y_lab != self.ignore_label]
        if self.class_weights is not None:
            W = self.class_weights[y_lab.long()]
            loss = loss * W
        return loss.sum()
