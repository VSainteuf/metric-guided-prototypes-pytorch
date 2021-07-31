import torch
import torch.nn as nn

from torch_scatter import scatter_softmax, scatter_sum, scatter_logsumexp


class HierarchicalInference(nn.Module):
    """Tree-graph hierarchical inference module"""

    def __init__(self, model, path_matrix, sibling_mask):
        """

        Args:
            model (nn.Module): backbone feature extracting network, should return an embedding for all nodes of the tree
            (n_nodes = num_classes (or leaf nodes) + internal nodes)
            path_matrix (tensor): 2D tensor of shape (depth x n_nodes) that specifies the parent-child relationships in
            the tree. For i in [1,depth] and for k in [1,n_nodes], path_matrix[d-i, k] gives the i-th parent of node k
            of 0 if the root has already been reached.
            sibling_mask (tensor): 1D mask over the nodes, that specifies the sibling groups (i.e. nodes with the same
            first parent). This mask is used to compute the softmax restricted to each  sibling group.
        """
        super(HierarchicalInference, self).__init__()
        self.model = model
        self.path_matrix = path_matrix
        self.sibling_mask = sibling_mask

    def forward(self, *input):
        """Returns the log-probability of the marginal probability of each node in the graph"""
        edge_logits = self.model(*input)
        if len(edge_logits.shape) == 4:  # Flatten 2D data
            two_dim_data = True
            b, c, h, w = edge_logits.shape
            edge_logits = (
                edge_logits.view(b, c, h * w)
                .transpose(1, 2)
                .contiguous()
                .view(b * h * w, c)
            )
        else:
            two_dim_data = False

        lse = scatter_logsumexp(edge_logits, self.sibling_mask, dim=1)
        scaled_logits = edge_logits - lse[:, self.sibling_mask]
        marginal_logits = scaled_logits.clone()
        depth = self.path_matrix.shape[0]
        for d in range(depth):
            parent_logits = scaled_logits[:, self.path_matrix[d]]
            parent_logits[:, self.path_matrix[d] == 0] = 0
            marginal_logits = marginal_logits + parent_logits

        if two_dim_data:  # Un-flatten 2D data
            _, n_out = marginal_logits.shape
            marginal_logits = (
                marginal_logits.view(b, h * w, n_out)
                .transpose(1, 2)
                .contiguous()
                .view(b, n_out, h, w)
            )

        return marginal_logits


class HierarchicalCrossEntropy(nn.Module):
    def __init__(
        self,
        path_matrix,
        alpha=1,
        class_weights=None,
        ignore_label=None,
        focal_gamma=None,
        eps=0.000001,
    ):
        """
        Hierarchical Cross-Entropy
        Args:
            path_matrix(tensor): 2D tensor of shape (depth x n_nodes) that specifies the parent-child relationships in
            the tree. For i in [1,depth] and for k in [1,n_nodes], path_matrix[d-i, k] gives the i-th parent of node k
            of 0 if the root has already been reached.:
            alpha (float): discounting strength parameter
            class_weights (list): (optional) class weighting values
            ignore_label (int): (optional) label to ignore in the loss
            focal_gamma (float): (optional) focal loss parameter. If provided the log-probabilities are multiplied by
            a factor (1 - p)^gamma .
            eps (float): default 10e-6, for numerical stability.
        """
        super(HierarchicalCrossEntropy, self).__init__()
        self.alpha = alpha
        self.class_weights = class_weights
        self.ignore_label = ignore_label
        self.M = torch.Tensor(path_matrix).long().cuda()
        self.eps = eps
        self.gamma = focal_gamma

    def forward(self, input, y_true):
        if len(input.shape) == 4:  # Flatten 2D data
            b, c, h, w = input.shape
            input = (
                input.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)
            )
            y_true = y_true.view(b * h * w)

        # Posterior class probabilities
        p = nn.Softmax(dim=-1)(input)

        # Compute cumulative probabilities of internal nodes
        cum_proba = torch.ones((p.shape[0], int(self.M.max() + 1)), device=p.device)
        for d in range(self.M.shape[0] - 1, 0, -1):
            cum_proba = cum_proba + scatter_sum(
                p, self.M[d, :], dim=1, dim_size=int(self.M.max()) + 1
            )

        cond_proba = (
            torch.cat([cum_proba[:, self.M[1:, :]], p[:, None, :]], dim=1) + self.eps
        ) / (cum_proba[:, self.M] + self.eps)

        # Discounting coefficients
        c = (
            self.alpha
            * torch.arange(self.M.shape[0], 0, -1, device=input.device).float()
        )
        c = torch.exp(-c).squeeze()

        # Focal or simple corss-entropy
        if self.gamma is not None:
            out = (
                c[None, :, None]
                * (1 - cond_proba) ** self.gamma
                * torch.log(cond_proba)
            )
        else:
            out = c[None, :, None] * torch.log(cond_proba)

        # Combine levels
        out = -out.sum(dim=1).squeeze()
        out = out.gather(dim=1, index=y_true.view(-1, 1).long()).squeeze()

        if self.class_weights is not None:
            W = self.class_weights[y_true.long()]
            out = out * W

        if self.ignore_label is not None:
            out = out[y_true != self.ignore_label]
        return out.mean()
