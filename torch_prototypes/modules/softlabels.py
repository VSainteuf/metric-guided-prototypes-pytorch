import torch.nn as nn
import torch


class SoftCrossEntropy(nn.Module):
    def __init__(
        self, D, beta=1, class_weights=None, ignore_label=None, focal_gamma=None
    ):
        """
        Single Module to compute the soft-labels and pass tham to a cross-entropy loss.
        Args:
            D (tensor): Cost matrix
            beta (float): Invert temperature in the softmax layer.
            class_weights (list): (optional) class weighting values
            ignore_label (int): (optional) label to ignore in the loss
            focal_gamma (float): (optional) focal loss parameter. If provided the log-probabilities are multiplied by
            a factor (1 - p)^gamma .
        """
        super(SoftCrossEntropy, self).__init__()
        self.D = D / D.max()
        self.beta = beta
        self.class_weights = class_weights
        self.ignore_label = ignore_label
        self.kldiv = nn.KLDivLoss(reduction="batchmean")
        self.gamma = focal_gamma

    def forward(self, input, y_true):
        if len(input.shape) == 4:  # Flatten 2D data
            b, c, h, w = input.shape
            input = (
                input.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)
            )
            y_true = y_true.view(b * h * w)

        if self.ignore_label is not None:
            input = input[y_true != self.ignore_label]
            y_true = y_true[y_true != self.ignore_label]

        out = torch.nn.functional.log_softmax(input, dim=1)
        soft_target = nn.Softmax(dim=-1)(-self.beta * self.D[y_true.long()])

        if self.gamma is not None:
            out = out * (1 - torch.exp(out)) ** self.gamma
        out = self.kldiv(out, soft_target)

        return out.mean()
