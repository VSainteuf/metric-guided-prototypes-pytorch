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
            edge_logits = edge_logits.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)
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
            marginal_logits = marginal_logits.view(b, h * w, n_out).transpose(1, 2).contiguous().view(b, n_out, h, w)

        return marginal_logits
