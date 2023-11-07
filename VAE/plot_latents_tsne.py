"""Plot VAE latents"""
import torch
import numpy as np
from matplotlib import pyplot as plt
from vae import ConvVAE
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
from tqdm import tqdm


def plot_latent_space(pt_name):
    model = ConvVAE(latent_dim=10)
    model.load_state_dict(torch.load(f"results/{pt_name}"))
    print("Loading MNIST-test data for latent space visualization")
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../mnist_data', transform=transforms.Compose([transforms.ToTensor(), ])), batch_size=64,
        shuffle=True)
    # visualize latent space
    latents = []
    classes = []
    i = 0
    for im, class_ in tqdm(test_loader):
        mu, sigma = model.encoder(im)
        z = torch.randn_like(mu) * sigma.exp() + mu
        latents.append(z)
        classes.append(class_)
        i += 1
        if i == 1000:
            break

    # convert latent space to t-sne space
    latents = torch.cat(latents, dim=0).detach().cpu().numpy()
    classes = torch.cat(classes, dim=0)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    print("Computing t-SNE embedding")
    latents_tsne = tsne.fit_transform(latents)
    # plot latent space
    plt.figure(figsize=(10, 10))
    plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], c=classes, cmap='tab10')
    plt.colorbar()
    plt.savefig("results/latent_space.png")
    plt.show()


if __name__ == '__main__':
    plot_latent_space("vae_l_dim-10.pt")
