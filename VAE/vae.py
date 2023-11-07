import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import OrderedDict
from visualize_vae import save_results

import os


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 64, 2 * self.latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.flatten(x)
        """
        Why log_std instead of std? 
        https://stats.stackexchange.com/questions/353220/why-in-variational-auto-encoder-gaussian-variational-family-we-model-log-sig
        """
        mu, log_std = self.fc(x).chunk(2, dim=-1)

        return mu, log_std


class Decoder(nn.Module):

    def __init__(self, laten_dim):
        self.latent_dim = laten_dim
        super(Decoder, self).__init__()
        self.linear = nn.Linear(self.latent_dim, 64 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x, apply_sigmoid=False):
        out = self.linear(x)
        out = F.relu(out)
        out = out.view(-1, 64, 7, 7)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if apply_sigmoid:
            out = F.relu(out)

        return out


class ConvVAE(nn.Module):

    def __init__(self, latent_dim):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)

        self.decoder = Decoder(latent_dim)

    def loss(self, x):
        mu, log_std = self.encoder(x)
        # log_std = torch.clamp(log_std, min=-4, max=1)
        # Re-parametrization trick z= mu + sigma * ε ~ N(0,I)
        z = torch.randn_like(mu) * log_std.exp() + mu
        x_recon = self.decoder(z)

        # mse of log(1/sqrt(2*pi))*exp(-(x-mu).T*(x-mu))= const - (x-mu).T*(x-mu)
        # since we want to maximize it, we minimize negative of it which is mse
        # Note std is I
        recon_loss = F.mse_loss(x, x_recon, reduction='none').view(x.shape[0], -1).sum(1).mean()

        kl_loss = 0.5 * torch.sum((log_std.exp() + mu ** 2 - 1 - log_std), dim=1)
        kl_loss = kl_loss.mean()

        return OrderedDict(loss=recon_loss + kl_loss,
                           recon_loss=recon_loss,
                           kl_loss=kl_loss)

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim)
            samples = self.decoder(z)
        return samples


def train_epoch(model, train_loader, epoch, optimizer, verbose, grad_clip):
    model.train()

    if verbose:
        pbar = tqdm(total=len(train_loader))

    losses = OrderedDict()
    logged = False
    for img, label in train_loader:
        img = (img > 0.5).to(torch.float)
        if not logged:
            logged = True
            mu, log_var = model.encoder(img)
            eps = torch.randn_like(mu)
            recon = model.decoder(mu + log_var.exp() * eps)
            save_images(recon.permute(0, 2, 3, 1), epoch, f"train_images/train")

        out = model.loss(img)
        optimizer.zero_grad()
        out['loss'].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        desc = f"Epoch {epoch}"
        for k, v in out.items():
            losses[k] = losses.get(k, [])
            losses[k].append(v.item())
            avg_loss = np.mean(losses[k][-50:])
            desc += f", {k} {avg_loss:.4f}"

        if verbose:
            pbar.set_description(desc)
            pbar.update(img.shape[0])

    if verbose:
        pbar.close()
    return losses


def eval_loss(model, data_loader, verbose, epoch):
    model.eval()
    total_loss = OrderedDict()
    logged = False
    with torch.no_grad():
        for img, labels in data_loader:
            if not logged:
                # log at the beginning of each epoch
                mu, log_var = model.encoder(img)
                eps = torch.randn_like(mu)
                recon = model.decoder(mu + log_var.exp() * eps)
                save_images(recon.permute(0, 2, 3, 1), epoch, "test_images/eval")
                logged = True
            out = model.loss(img)
            for k, v in out.items():
                total_loss[k] = out.get(k, 0) + v.item() * img.shape[0]

        desc = "test"

        for k in total_loss.keys():
            total_loss[k] /= len(data_loader)
            desc += f", {k} {total_loss[k]:.4f}"
        if verbose:
            print(desc)
    return total_loss


def train(model, train_loader, test_loader, verbose=False, **train_args):
    epochs, lr, grad_clip = train_args["epochs"], train_args["lr"], train_args["grad_clip"]
    optimizer = Adam(model.parameters(), lr=lr)
    # optimizer = SGD(model.parameters(), lr=lr)
    train_losses, test_losses = OrderedDict(), OrderedDict()

    for epoch in range(epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer=optimizer, epoch=epoch, verbose=True,
                                 grad_clip=grad_clip)
        test_loss = eval_loss(model, test_loader, verbose, epoch)

        for key in train_loss.keys():
            if key not in train_losses:
                train_losses[key] = []
                test_losses[key] = []
            train_losses[key].extend(train_loss.get(key, []))
            test_losses[key].append(test_loss.get(key, []))

    return train_losses, test_losses


def start_training():
    model = ConvVAE(latent_dim=10)

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor()
                                                                  # first, convert image to PyTorch tensor
                                                              ])), batch_size=64, shuffle=True)

    # download and transform test dataset
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True,
                                                             train=False,
                                                             transform=transforms.Compose([
                                                                 transforms.ToTensor()
                                                                 # first, convert image to PyTorch tensor
                                                             ])), batch_size=64, shuffle=True)

    train_args = dict(epochs=50, lr=1e-3, grad_clip=1)
    train_losses, test_losses = train(model=model, train_loader=train_loader, test_loader=test_loader,
                                      verbose=True, **train_args)

    # save the model
    torch.save(model.state_dict(), "results/vae_l_dim-10.pt")

    train_losses = np.stack((train_losses["loss"], train_losses["recon_loss"], train_losses["kl_loss"]), axis=1)
    test_losses = np.stack((test_losses['loss'], test_losses['recon_loss'], test_losses['kl_loss']), axis=1)

    samples = model.sample(100).clamp(0, 1) * 255.
    samples = samples.cpu().permute(0, 2, 3, 1).numpy()

    test_images, _ = next(iter(test_loader))[:50]
    with torch.no_grad():
        test_images = test_images
        mu, _ = model.encoder(test_images)
        test_recon = model.decoder(mu).clamp(0, 1)
        test_recon = torch.stack((test_images, test_recon), dim=1).view(-1, 1, 28, 28)
        test_recon = test_recon.permute(0, 2, 3, 1).cpu().numpy() * 255.

    test_images, _ = next(iter(test_loader))[:50]
    with torch.no_grad():
        test_images = test_images
        mu, _ = model.encoder(test_images)
        mu1, mu2 = mu.chunk(2, dim=0)
        interps = [model.decoder(mu1 * (1 - alpha) + mu2 * alpha) for alpha in np.linspace(0, 1, 10)]
        interps = torch.stack(interps, dim=1).view(-1, 1, 28, 28)
        interps = torch.clamp(interps, 0, 1) * 255.
    interps = interps.permute(0, 2, 3, 1).cpu().numpy()

    return train_losses, test_losses, samples, test_recon, interps


def save_images(predictions, epoch, name):
    fig = plt.figure(figsize=(8, 8))
    predictions = predictions.detach().numpy()
    for i in range(predictions.shape[0]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(predictions[i, ...], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(f'{name}_image_at_epoch_{epoch:04d}.png')
    plt.close(fig)


if not os.path.exists("train_images"):
    os.mkdir("train_images")
    os.mkdir("test_images")

if __name__ == '__main__':
    save_results(start_training)

