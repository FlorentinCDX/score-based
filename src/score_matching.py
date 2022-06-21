import torch
import torch.autograd as autograd
from utils import *

#Basic score matching loss function
def score_matching(model, samples):
    """Score matching loss originally proposed by Hyvarinen et al.

    Args:
        model (nn.Module): pytorch model
        samples (torch.Tensor): tensor sample
        
    Returns:
        score torch.Tensor): score matching score of the input tensor
    """
    samples.requires_grad_(True)
    logp = model(samples)
    # Compute the norm loss
    norm_loss = torch.norm(logp, dim=-1) ** 2 / 2.
    # Compute the Jacobian loss
    jacob_mat = jacobian(model, samples)
    tr_jacobian_loss = torch.diagonal(jacob_mat, dim1=-2, dim2=-1).sum(-1)
    return (tr_jacobian_loss + norm_loss).mean(-1)

def sliced_score_matching(model, samples):
    """Sliced score matching loss originally proposed by Hyvarinen et al.

    Args:
        model (nn.Module): pytorch model
        samples (torch.Tensor): tensor sample
        
    Returns:
        score torch.Tensor): score matching score of the input tensor
    """
    samples.requires_grad_(True)
    # Construct random vectors
    vectors = torch.randn_like(samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)
    # Compute the optimized vector-product jacobian
    logp, jvp = autograd.functional.jvp(model, samples, vectors, create_graph=True)
    # Compute the norm loss
    norm_loss = (logp * vectors) ** 2 / 2.
    # Compute the Jacobian loss
    v_jvp = jvp * vectors
    jacob_loss = v_jvp
    loss = jacob_loss + norm_loss
    return loss.mean(-1).mean(-1)

def denoising_score_matching(model, samples, sigma=0.01):
    """ Denoising score matching discussed by Vincent et al in the context of autoencoder

    Args:
        model (nn.Module): pytorch model
        samples (torch.Tensor): tensor sample
        
    Returns:
        score torch.Tensor): score matching score of the input tensor
    """
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = model(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
    return loss
