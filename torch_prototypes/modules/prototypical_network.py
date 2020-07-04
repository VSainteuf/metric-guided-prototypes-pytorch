import torch
import torch.nn as nn
from torch_prototypes.metrics.distortion import Pseudo_Huber

from torch_scatter import scatter_mean


class LearntPrototypes(nn.Module):
    """
    Learnt Prototypes Module. Classification module based on learnt prototypes, to be wrapped around a backbone
    embedding network.
    """
    def __init__(self, model, n_prototypes, embedding_dim, prototypes=None, squarred=False, ph=None,
                 dist='euclidean', device='cuda'):
        """

        Args:
            model (nn.Module): feature extracting network
            n_prototypes (int): number of prototypes to use
            embedding_dim (int): dimension of the embedding space
            prototypes (tensor): Prototype tensor of shape (n_prototypes x embedding_dim),
            squared (bool): Whether to use the squared Euclidean distance or not
            ph (float): if specified, the distances function is huberized with delta parameter equal to the specified value
            dist (str): default 'euclidean', other possibility 'cosine'
            device (str): device on which to declare the prototypes (cpu/cuda)
        """
        super(LearntPrototypes, self).__init__()
        self.model = model
        self.prototypes = nn.Parameter(torch.rand((n_prototypes, embedding_dim), device=device)).requires_grad_(
            True) if prototypes is None else nn.Parameter(prototypes).requires_grad_(
            False)
        self.n_prototypes = n_prototypes
        self.squarred = squarred
        self.dist = dist
        self.ph = None if ph is None else Pseudo_Huber(delta=ph)

    def forward(self, *input):
        embeddings = self.model(*input)

        if len(embeddings.shape) == 4:  # Flatten 2D data
            two_dim_data = True
            b, c, h, w = embeddings.shape
            embeddings = embeddings.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)
        else:
            two_dim_data = False

        if self.dist == 'cosine':
            dists = 1 - nn.CosineSimilarity(dim=-1)(embeddings[:, None, :], self.prototypes[None, :, :])
        else:
            dists = torch.norm(embeddings[:, None, :] - self.prototypes[None, :, :], dim=-1)
            if self.ph is not None:
                dists = self.ph(dists)
        if self.squarred:
            dists = dists ** 2

        if two_dim_data:  # Un-flatten 2D data
            dists = dists.view(b, h * w, self.n_prototypes).transpose(1, 2).contiguous().view(b, self.n_prototypes, h,
                                                                                              w)

        return -dists


class HypersphericalProto(nn.Module):
    """
    Implementation of Hypershperical Prototype Networks (Mettes et al., 2019)
    """
    def __init__(self, model, num_classes, prototypes):
        """
        Args:
            model (nn.Module): backbone feature extracting network
            num_classes (int): number of classes
            prototypes (tensor): pre-defined prototypes, tensor has shape (num_classes x embedding_dimension)
        """
        super(HypersphericalProto, self).__init__()
        self.model = model
        self.prototypes = nn.Parameter(prototypes).requires_grad_(False)
        self.num_classes = num_classes

    def forward(self, *input):
        embeddings = self.model(*input)

        if len(embeddings.shape) == 4:  # Flatten 2D data
            two_dim_data = True
            b, c, h, w = embeddings.shape
            embeddings = embeddings.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)
        else:
            two_dim_data = False

        dists = 1 - nn.CosineSimilarity(dim=-1)(embeddings[:, None, :], self.prototypes[None, :, :])
        scores = - dists.pow(2)

        if two_dim_data:  # Un-flatten 2D data
            scores = scores.view(b, h * w, self.num_classes).transpose(1, 2).contiguous().view(b, self.num_classes, h,
                                                                                               w)
        return scores


class DeepNCM(nn.Module):
    """
    Implementation of Deep Nearest Mean Classifiers (Gueriero et al., 2017)
    """
    def __init__(self, model, num_classes, embedding_dim):
        """
        Args:
            model (nn.Module): backbone feature extracting network
            num_classes (int): number of classes
            embedding_dim (int): number of dimensions of the embedding space
        """
        super(DeepNCM, self).__init__()
        self.model = model
        self.prototypes = nn.Parameter(torch.rand((num_classes, embedding_dim), device='cuda')).requires_grad_(False)
        self.num_classes = num_classes
        self.counter = torch.zeros(num_classes)
        self._check_device = True

    def forward(self, *input_target):
        """
        DeepNCM needs the target vector to update the class prototypes
        Args:
            *input_target: tuple of tensors (*input, target)
        """
        input = input_target[:-1]
        y_true = input_target[-1]
        embeddings = self.model(*input)
        if self._check_device:
            self.counter = self.counter.to(embeddings.device)
            self._check_device = False

        if len(embeddings.shape) == 4:  # Flatten 2D data
            two_dim_data = True
            b, c, h, w = embeddings.shape
            embeddings = embeddings.view(b, c, h * w).transpose(1, 2).contiguous().view(b * h * w, c)
            y_true = y_true.view(b * h * w)
        else:
            two_dim_data = False

        if self.training:
            represented_classes = torch.unique(y_true).detach().cpu().numpy()

            # Compute Prototypes
            new_prototypes = scatter_mean(embeddings, y_true.unsqueeze(1), dim=0, dim_size=self.num_classes).detach()
            # Updated stored prototype values
            self.prototypes[represented_classes, :] = (self.counter[represented_classes, None] * self.prototypes[
                                                                                                 represented_classes, :]
                                                       + new_prototypes[represented_classes, :]) / (
                                                              self.counter[represented_classes, None] + 1)
            # self.counter[represented_classes]
            self.counter[represented_classes] = self.counter[represented_classes] + 1
        dists = torch.norm(embeddings[:, None, :] - self.prototypes[None, :, :], dim=-1)
        if two_dim_data:  # Un-flatten 2D data
            dists = dists.view(b, h * w, self.num_classes).transpose(1, 2).contiguous().view(b, self.num_classes, h, w)

        return -dists.pow(2)
