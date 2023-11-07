import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import OrderedDict
import os

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

from encoder import Encoder
from decoder import Decoder
from visualize_vqvae import save_results


class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        self.encoder = Encoder()
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=4)
        self.decoder = Decoder()
        self.beta = 0.2

    def loss(self, input_):
        features = self.encoder(input_)

        quant_out, quantize_loss = self.quantize(features)

        reconstruction = self.decoder(quant_out)

        # reconstruction loss
        recon_loss = F.mse_loss(input_, reconstruction, reduction='none').view(input_.shape[0], -1).sum(1).mean()
        loss = recon_loss + quantize_loss

        result = OrderedDict(loss=loss, quantize_loss=quantize_loss)
        return result

    def quantize(self, features):
        quant_in = features.permute(0, 2, 3, 1)
        b, h, w, c = features.shape
        quant_in = quant_in.view(b, -1, c)
        dist = torch.cdist(quant_in, self.embedding.weight.unsqueeze(0).repeat(b, 1, 1))

        # Find index of nearest embedding
        min_index = torch.argmin(dist, dim=-1)

        # Select the embedding weights
        quant_out = torch.index_select(self.embedding.weight, dim=0, index=min_index.view(-1))

        quantize_loss = self.quantize_loss(quant_in, quant_out)

        # make sure that gradient propagates to encoder out
        quant_in = quant_in.reshape(b*h*w, c)
        quant_out = quant_in + (quant_out - quant_in).detach()

        quant_out = quant_out.reshape(b, c, h, w)
        return quant_out, quantize_loss

    def quantize_loss(self, quant_in, quant_out):
        quant_in = quant_in.reshape(-1, quant_in.shape[-1])

        # compute loss
        commitment_loss = ((quant_out.detach() - quant_in) ** 2).mean()
        codebook_loss = ((quant_out - quant_in.detach()) ** 2).mean()

        quantize_loss = commitment_loss + self.beta * codebook_loss
        return quantize_loss


# x = torch.randn(16, 1, 28, 28)
# model = VQVAE()
# model.loss(x)


def train_epoch(model, train_loader, epoch, optimizer, verbose, grad_clip):
    model.train()

    if verbose:
        pbar = tqdm(total=len(train_loader))

    losses = OrderedDict()
    logged = False
    for img, label in train_loader:
        if not logged:
            logged = True
            feat = model.encoder(img)
            quant_out, _ = model.quantize(feat)
            recon = model.decoder(quant_out)
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
                feat = model.encoder(img)
                quant_out, _ = model.quantize(feat)
                recon = model.decoder(quant_out)
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
    train_losses, test_losses = OrderedDict(), OrderedDict()

    for epoch in range(epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer=optimizer, epoch=epoch, verbose=True, grad_clip=False)
        test_loss = eval_loss(model, test_loader, verbose, epoch)

        for key in train_loss.keys():
            if key not in train_losses:
                train_losses[key] = []
                test_losses[key] = []
            train_losses[key].extend(train_loss.get(key, []))
            test_losses[key].append(test_loss.get(key, []))

    return train_losses, test_losses


def start_training():
    model = VQVAE()

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True, train=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  # first, convert image to PyTorch tensor
                                                              ])), batch_size=64, shuffle=True)

    # download and transform test dataset
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../mnist_data', download=True,
                                                             train=False,
                                                             transform=transforms.Compose([
                                                                 transforms.ToTensor(),
                                                                 # first, convert image to PyTorch tensor
                                                             ])), batch_size=64, shuffle=True)

    train_args = dict(epochs=20, lr=1e-3, grad_clip=1)
    train_losses, test_losses = train(model=model, train_loader=train_loader, test_loader=test_loader,
                                      verbose=True, **train_args)

    train_losses = np.stack((train_losses["loss"], train_losses["recon_loss"], train_losses["kl_loss"]), axis=1)
    test_losses = np.stack((test_losses['loss'], test_losses['recon_loss'], test_losses['kl_loss']), axis=1)
    test_images, _ = next(iter(test_loader))[:50]
    with torch.no_grad():
        test_images = test_images
        feat = model.encoder(test_images)
        quant_out, _ = model.quantize(feat)
        test_recon = model.decoder(quant_out).clamp(0, 1)
        test_recon = torch.stack((test_images, test_recon), dim=1).view(-1, 1, 28, 28)
        test_recon = test_recon.permute(0, 2, 3, 1).cpu().numpy() * 255.


    return train_losses, test_losses, test_recon


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
save_results(start_training)
