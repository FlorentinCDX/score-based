import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons, make_swiss_roll
from utils import *
from score_matching import *
import torch
import torch.nn as nn
import torch.optim as optim

import argparse

data = sample_batch(10**4).T

def main():
    # train args
    parser = argparse.ArgumentParser(description='Score based model tutorial')
    parser.add_argument('--score', default='sliced', help='score matching function (base, sliced, denoising')
    parser.add_argument('--outdir', default='../img/', help='directory to output images')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 2000)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()
    # determine device and device dep. args
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Our approximation model
    model = nn.Sequential(
            nn.Linear(2, 128), nn.Softplus(),
            nn.Linear(128, 128), nn.Softplus(),
            nn.Linear(128, 2)
    )
    # Create ADAM optimizer over our model
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Score objective function
    if args.score == "base":
        score_function = score_matching
    elif args.score == "sliced":
        score_function = sliced_score_matching
    elif args.score == "denoising":
        score_function = denoising_score_matching
    

    print("________________Start Training________________")
    dataset = torch.tensor(data.T).float()
    for t in range(args.epochs):
        # Compute the loss.
        loss = score_matching(model, dataset)
        # Before the backward pass, zero all of the network gradients
        optimizer.zero_grad()
        # Backward pass: compute gradient of the loss with respect to parameters
        loss.backward()
        # Calling the step function to update the parameters
        optimizer.step()
        if ((t % 500) == 0):
            print("score matching : ", loss.data)

    #plot results
    x = torch.Tensor([1.5, -1.5])
    samples = sample_langevin(model, x).detach()
    plot_gradients(model, data)
    plt.scatter(samples[:, 0], samples[:, 1], color='green', edgecolor='black', s=150)
    # draw arrows for each mcmc step
    deltas = (samples[1:] - samples[:-1])
    deltas = deltas - deltas / np.linalg.norm(deltas, keepdims=True, axis=-1) * 0.04
    for i, arrow in enumerate(deltas):
            plt.arrow(samples[i,0], samples[i,1], arrow[0], arrow[1], width=1e-4, head_width=2e-2, color="green", linewidth=3)
    plt.savefig(args.outdir + 'score-field.png')

if __name__ == '__main__':
    main()
