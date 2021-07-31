import torch
from torch_prototypes.metrics.rank import RankLoss
from torch_prototypes.metrics.hyperspherical import SeparationLoss
from torch_prototypes.metrics.distortion import DistortionLoss


def embed_nomenclature(
    D,
    embedding_dimension,
    loss="rank",
    n_steps=1000,
    lr=10,
    momentum=0.9,
    weight_decay=1e-4,
    ignore_index=None,
):
    """
    Embed a finite metric into a target embedding space
    Args:
        D (tensor): 2D-cost matrix of the finite metric
        embedding_dimension (int): dimension of the target embedding space
        loss (str): embedding loss to use distortion base (loss='disto') or rank based (loss='rank')
        n_steps (int): number of gradient iterations
        lr (float): learning rate
        momentum (float): momentum
        weight_decay (float): weight decay

    Returns:
        embedding (tensor): embedding of each vertex of the finite metric space, shape n_vertex x embedding_dimension
    """
    n_vertex = D.shape[0]
    mapping = torch.rand(
        (n_vertex, embedding_dimension), requires_grad=True, device=D.device
    )

    if loss == "rank":
        crit = RankLoss(D, n_triplets=1000)
    elif loss == "disto":
        crit = DistortionLoss(D, scale_free=False)
    else:
        raise ValueError

    optimizer = torch.optim.SGD(
        [mapping], lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    print("Embedding nomenclature  . . .")
    for i in range(n_steps):
        loss = crit(mapping)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "Step {}: loss {:.4f} ".format(i + 1, loss.cpu().detach().numpy(), end="\r")
        )
    print("Final loss {:.4f}".format(crit(mapping).cpu().detach().numpy()))
    return mapping.detach()


def embed_on_sphere(
    D, embedding_dimension, lr=0.1, momentum=0.9, n_steps=1000, wd=1e-4
):
    """
    Embed finite metric on the hypersphere
    Args:
        D (tensor): 2D-cost matrix of the finite metric
        embedding_dimension (int): dimension of the target embedding space
        lr (float): learning rate
        momentum (float): momentum
        n_steps (int): number of gradient iterations
        wd (float): weight decay

    Returns:
        embedding (tensor): embedding of each vertex of the finite metric space, shape n_vertex x embedding_dimension

    """
    n_vertex = D.shape[0]
    mapping = torch.rand(
        (n_vertex, embedding_dimension), requires_grad=True, device=D.device
    )
    optimizer = torch.optim.SGD([mapping], lr=lr, momentum=momentum, weight_decay=wd)

    L_hp = SeparationLoss()
    L_pi = RankLoss(D, n_triplets=1000, dist="cos")

    print("Embedding nomenclature  . . .")
    for i in range(n_steps):
        with torch.no_grad():
            mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
        loss = L_hp(mapping) + L_pi(mapping)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "Step {}: loss {:.4f} ".format(i + 1, loss.cpu().detach().numpy(), end="\r")
        )
    with torch.no_grad():
        mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
        loss = L_hp(mapping) + L_pi(mapping)
    print("Final loss {:.4f} ".format(loss.cpu().detach().numpy()))
    return mapping.detach()
