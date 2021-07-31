import torch
import torch.nn as nn


class AverageCost(nn.Module):
    """Average Cost of predictions, according to given cost matrix D"""

    def __init__(self, D, ignore_index=None):
        """
        Args:
            D (tensor): 2D cost matrix (n_classes x n_classes)
            ignore_index (int): index of label to ignore (if any)
        """
        super(AverageCost, self).__init__()
        self.D = D
        self.ignore_index = ignore_index

    def forward(self, input, y_true):
        if len(input.shape) == 4:  # Flatten 2D data
            b, c, h, w = input.shape
            input = (
                input.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)
            )
            y_true = y_true.view(b * h * w)

        out = nn.Softmax(dim=-1)(input)
        b = torch.zeros(out.shape, device=out.device)
        b = b.scatter(1, out.argmax(dim=1).view(-1, 1), 1)
        Dists = self.D[y_true.long()]

        if self.ignore_index is None:
            return float((Dists * b).sum(dim=-1).mean().detach().cpu().numpy())
        else:
            return float(
                (Dists * b)[y_true.long() != self.ignore_index]
                .sum(dim=-1)
                .mean()
                .detach()
                .cpu()
                .numpy()
            )


class EMDLoss(nn.Module):
    """Squared Earth Mover regularization"""

    def __init__(self, l, mu, D):
        """
        Args:
            l (float): regularization coefficient
            mu (float): offset
            D (ground distance matrix): 2D cost matrix (n_classes x n_classes)
        """
        super(EMDLoss, self).__init__()
        self.l = l
        self.mu = mu
        self.D = D

    def forward(self, input, y_true):

        if len(input.shape) == 4:  # Flatten 2D data
            b, c, h, w = input.shape
            input = (
                input.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)
            )
            y_true = y_true.view(b * h * w)

        out = nn.Softmax(dim=-1)(input)
        Dists = self.D[y_true.long()]
        p2 = out ** 2
        E = p2 * (Dists - self.mu)
        E = E.sum(dim=-1)

        return self.l * E.mean()
