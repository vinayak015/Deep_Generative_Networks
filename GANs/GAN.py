"""
My experiments with GANs
Solution of  https://github.com/rll/deepul/tree/master/homeworks/hw4
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import OrderedDict
from sklearn.manifold import TSNE
import os
import torch.utils.data as data

from common_utils import save_images
from dataset_utils import gaussian_data_2mode


class MLP(nn.Module):
    def __init__(self, input_size, num_hidden, hidden_size, out_size):
        super(MLP, self).__init__()
        layers = []
        for hidden in range(num_hidden):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.LeakyReLU(0.2))
            input_size = hidden_size
        # add last layer
        layers.append(nn.Linear(input_size, out_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)


class GeneratorGauss2Mode(nn.Module):
    def __init__(self, latent_dim, num_hidden, hidden_size, out_size):
        super(GeneratorGauss2Mode, self).__init__()
        self.latent_dim = latent_dim
        self.mlp = MLP(latent_dim, num_hidden, hidden_size, out_size)

    def forward(self, latent):
        return self.mlp(latent)

    def sample(self, batch_size):
        latent = torch.normal(mean=torch.zeros([batch_size, self.latent_dim]),
                              std=torch.ones([batch_size, self.latent_dim]))
        return self.forward(latent)


class DiscriminatorGauss2Mode(nn.Module):
    def __init__(self, input_size, num_hidden, hidden_size, out_size=1):
        super(DiscriminatorGauss2Mode, self).__init__()
        self.mlp = MLP(input_size, num_hidden, hidden_size, out_size)

    def forward(self, data_):
        return torch.sigmoid(self.mlp(data_))


def train_validate_epoch(generator, discriminator, loader, epoch, optimizer_gen, optimizer_disc, verbose, mode,
                         grad_clip=False):
    if mode == "train":
        generator.train()
        discriminator.train()
    else:
        generator.eval()
        discriminator.eval()

    if verbose:
        pbar = tqdm(total=len(loader.dataset))

    train_losses = OrderedDict([('gen_loss_train', []), ('disc_loss_train', [])])
    val_losses = OrderedDict([('gen_loss_val', []), ('disc_loss_val', [])])

    for data_ in loader:
        batch_size = data_.size(0)
        fakes = generator.sample(batch_size)
        disc_loss = -(torch.log(discriminator(data_)).mean() + torch.log(1 - discriminator(fakes)).mean())
        optimizer_disc.zero_grad()
        if mode == "train":
            disc_loss.backward()
            optimizer_disc.step()
            train_losses[f"disc_loss_{mode}"].append(disc_loss.item())
        else:
            val_losses[f"disc_loss_{mode}"].append(disc_loss.item())
        optimizer_gen.zero_grad()
        gen_loss = torch.log(1 - discriminator(generator.sample(batch_size))).mean()
        if mode == "train":
            gen_loss.backward()
            optimizer_gen.step()
            train_losses[f"gen_loss_{mode}"].append(gen_loss.item())
        else:
            val_losses[f"gen_loss_{mode}"].append(gen_loss.item())

        desc = f"Epoch {epoch}-{mode}"
        if verbose:
            pbar.set_description(desc)
            pbar.update(batch_size)

    if verbose:
        pbar.close()

    return train_losses if mode == "train" else val_losses


def train(generator, discriminator, train_loader, test_loader, verbose, **train_args):
    epochs, lr, grad_clip = train_args["epochs"], train_args["lr"], train_args["grad_clip"]
    optimizer_gen = Adam(generator.parameters(), lr=lr)
    optimizer_disc = Adam(discriminator.parameters(), lr=lr)

    train_losses = OrderedDict(gen_loss_train=[], disc_loss_train=[])
    test_losses = OrderedDict(gen_loss_val=[], disc_loss_val=[])

    def add_loss(loss, mode):
        if mode == "train":
            for key in loss.keys():
                train_losses[key].extend(loss[key])
        else:
            for key in loss.keys():
                test_losses[key].extend(loss[key])

    for epoch in range(epochs):
        mode = "train"
        train_loss = train_validate_epoch(generator, discriminator, train_loader, epoch, optimizer_gen, optimizer_disc,
                                          verbose, mode, grad_clip=False)
        add_loss(train_loss, mode)
        mode = "val"
        test_loss = train_validate_epoch(generator, discriminator, test_loader, epoch, optimizer_gen, optimizer_disc,
                                         verbose, mode, grad_clip=False)
        add_loss(test_loss, mode)

    return train_losses, test_losses


def start():
    generator = GeneratorGauss2Mode(latent_dim=1, num_hidden=3, hidden_size=128, out_size=1)
    discriminator = DiscriminatorGauss2Mode(input_size=1, num_hidden=3, hidden_size=128, out_size=1)

    data_ = np.float32(gaussian_data_2mode())
    train_data = data_[:20000]
    test_data = data_[20000:]

    loader_args = dict(batch_size=64, shuffle=True)
    train_loader = data.DataLoader(train_data, **loader_args)
    test_loader = data.DataLoader(test_data, **loader_args)
    train_args = dict(epochs=25, lr=1e-4, grad_clip=1)

    def plot_losses(losses, title, fig_name):
        plt.figure()
        for key, loss in losses.items():
            n_itr = len(loss)
            xs = np.arange(n_itr)

            plt.plot(xs, loss, label=f'loss-{key}')

        plt.legend(loc='upper right')
        plt.title(title)
        plt.xlabel('Training Iteration')
        plt.ylabel('Loss')

        # Ensure the directory exists
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)

        # Save the figure
        fig_path = os.path.join(results_dir, fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()

    train_losses, test_losses = train(generator, discriminator, train_loader, test_loader, verbose=True, **train_args)
    # for key, val in train_losses:
    #     plot_losses(val, key, f"results/{}")
    plot_losses(train_losses, "train", f"results/train_losses.png")
    plot_losses(test_losses, "val", f"results/test_losses.png")
    generated = generator.sample(20000).detach().cpu().numpy()

    def plot_samples():
        xs = np.linspace(-1, 1, 1000)
        plt.figure()
        plt.hist(generated, bins=50, density=True, alpha=0.7, label='fake')
        plt.hist(train_data, bins=50, density=True, alpha=0.7, label='real')

        plt.legend()
        plt.title("real vs fake")
        plt.savefig("results/real_vs_fake.png")

    plot_samples()


start()
