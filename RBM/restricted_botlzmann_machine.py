"""
This file contains the implementation of Restricted Boltzmann Machine with Gibbs sampling.
The gradient computation can be verified from here: https://christian-igel.github.io/paper/TRBMAI.pdf
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class RBM:

    def __init__(self, n_visible, n_hidden, mcmc_steps=1000):
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible), requires_grad=False)
        self.hidden_variables = nn.Parameter(torch.randn(n_hidden), requires_grad=False)
        self.visible_bias = nn.Parameter(torch.randn(1, n_visible), requires_grad=False)
        self.hidden_bias = nn.Parameter(torch.randn(1, n_hidden), requires_grad=False)
        self.mcmc_steps = mcmc_steps
        self._v = torch.randn(n_visible, n_visible)

    def bernoulli_sample(self, p):
        random_n = torch.rand(p.size())
        return (p > random_n).type(torch.float)

    def sample_h_given_v(self, v):
        sigmoid = torch.sigmoid(v @ self.W.T + self.hidden_bias)
        sample = self.bernoulli_sample(sigmoid)
        return sigmoid, sample

    def sample_v_given_h(self, h):
        sigmoid = torch.sigmoid(h @ self.W + self.visible_bias)
        sample = self.bernoulli_sample(sigmoid)
        return sigmoid, sample

    def grad_w_partial(self, v):
        if len(v.shape) == 2:
            return torch.sigmoid(torch.matmul(v, self.W.T) + self.hidden_bias).T @ v
        else:
            y = torch.sigmoid(torch.matmul(v, self.W.T) + self.hidden_bias[None]).view(-1, 100, 1) @ v
            return y

    def grad_visible_bias_partial(self, v):
        """
        Since we are just returning the input this method is not needed
        but to make it consistent with other gradient method it is included $E=mc^2$
        """
        return v

    def grad_hidden_bias_partial(self, v):
        return torch.sigmoid(v @ self.W.T + self.hidden_bias)

    def update_params(self, lr, v_data, v_sampled: list):
        self.W += (lr * self.grad_w_partial(v_data)) - 1 / len(v_sampled) * (
            torch.sum(self.grad_w_partial(v_sampled), dim=0))

        self.visible_bias += (lr * self.grad_visible_bias_partial(v_data)) - 1 / len(v_sampled) * torch.sum(
            self.grad_visible_bias_partial(v_sampled), dim=0)

        self.hidden_bias += (lr * self.grad_hidden_bias_partial(v_data)) - 1 / len(v_sampled) * torch.sum(
            self.grad_hidden_bias_partial(v_sampled), dim=0)

    def gibbs_sampling(self, v):
        # Run Gibbs sampling
        sampled_v = []
        sampled_h = []
        reached_stationary = False
        for i in range(self.mcmc_steps):
            # Assume stationary distribution is reached after half
            if i > self.mcmc_steps // 2:
                reached_stationary = True

            _, h = self.sample_h_given_v(v)
            if reached_stationary:
                # sampled after reaching stationary distribution
                sampled_h.append(h)

            _, v = self.sample_v_given_h(h)
            if reached_stationary:
                # sampled after reaching stationary distribution
                sampled_v.append(v)
        sampled_v, sampled_h = torch.as_tensor(np.array(sampled_v)), torch.as_tensor(np.array(sampled_h))
        return sampled_v, sampled_h

    def _train(self, v, lr):
        sampled_v, sampled_h = self.gibbs_sampling(v)
        self.update_params(lr, v, sampled_v)


def train(rbm, train_data, epochs=2, lr=0.0001, ):
    print("Beginning of training")
    for epoch in (range(epochs)):
        print(f"Beginning of epoch {epoch}")
        for img, idx in tqdm(train_data):
            img = (img > 127).type(img.dtype)

            img = img.view(1, 784)
            rbm._train(img, lr, )

        _, h = rbm.sample_h_given_v(img)
        _, sampled_v = rbm.sample_v_given_h(h)
        plt.imshow(sampled_v.view(28, 28, 1), cmap="gray")
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.show()
    print("End of training")


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])))

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=False, transform=transforms.Compose([
#         transforms.ToTensor()
#     ])))

RBM_ = RBM(784, 100)

train(RBM_, train_loader)
