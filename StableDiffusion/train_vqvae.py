import os
import argparse
import yaml

import torch
import torch.functional as F
from torch.optim import Adam
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import numpy as np

from models.vqvae import VQVAE
from models.discriminator import Discriminator
from models.lpips import LPIPS
from dataset import CelebADataset
from collections import OrderedDict
from omegaconf import OmegaConf

device = torch.device("cuda:1")


def train(config_path):
    # with open(config_path, 'r') as file:
    #     config = OrderedDict(yaml.safe_load(file))
    config = OmegaConf.load(config_path)
    vq_conf = config['VQ-VAE']
    ds_conf = config.dataset
    train_conf = config.train

    ds = CelebADataset(**ds_conf)
    data_loader = DataLoader(ds, train_conf.batch, shuffle=True)

    model = VQVAE(**vq_conf).to(device)
    discriminator = Discriminator(**train_conf.d).to(device)
    lpips = LPIPS().eval().to(device)

    os.makedirs(train_conf.ckpt_dir, exist_ok=True)
    optim_encoder = Adam(model.parameters(), lr=train_conf.lr, betas=(0.5, 0.999))
    optim_d = Adam(discriminator.parameters(), lr=train_conf.lr, betas=(0.5, 0.999))

    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()

    d_step_start = train_conf.d.start

    steps = 0
    num_epochs = train_conf.epochs
    pbar = tqdm(total=len(ds))
    start_d = False
    for epoch in range(num_epochs):
        for batch in data_loader:
            steps += 1
            if steps > d_step_start:
                start_d = True
            recon_losses = []
            codebook_losses = []
            # commitment_losses = []
            perceptual_losses = []
            disc_losses = []
            gen_losses = []
            losses = []

            optim_encoder.zero_grad()
            optim_d.zero_grad()

            img_s = batch['img'].to(device)
            # text = batch['text']
            # mask = batch['mask']

            outs = model(img_s)
            img_g, z, quant_losses = outs

            quant_loss = quant_losses['commitment_loss'] + quant_losses['code_book_loss']

            recon_loss = recon_criterion(img_g, img_s)
            g_loss = recon_loss + quant_loss
            codebook_losses.append(quant_loss.item())

            if start_d:
                d_fake = discriminator(img_g)
                d_fake_loss = disc_criterion(d_fake, torch.ones(d_fake.shape, device=device))

                # gen_losses.append(d_fake_loss)
                g_loss = g_loss + train_conf.d.d_weight * d_fake_loss

            lpips_loss = torch.mean(lpips(img_s, img_g))
            g_loss = g_loss + train_conf.perceptual_weight * lpips_loss
            losses.append(g_loss.item())
            g_loss.backward()

            if start_d:
                d_fake_pred = discriminator(img_g.detach())
                d_real_pred = discriminator(img_s)

                d_fake_loss = disc_criterion(d_fake_pred, torch.zeros(d_fake_pred.shape, device=device))
                d_real_loss = disc_criterion(d_real_pred, torch.ones(d_real_pred.shape, device=device))

                disc_loss = train_conf.d.d_weight * (d_real_loss + d_fake_loss) / 2.0
                disc_losses.append(disc_loss.item())
                disc_loss.backward()

            desc = f"Epoch: {epoch}"
            pbar.set_description(desc)
            pbar.update(img_s.shape[0])

            save_images(img_s, img_g, train_conf.ckpt_dir, epoch)

        optim_d.step()
        optim_d.zero_grad()
        optim_encoder.step()
        optim_encoder.zero_grad()

        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(codebook_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(codebook_losses)))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(train_conf.ckpt_dir, f"vq_vae_{epoch}.pt"))
            torch.save(discriminator.state_dict(), os.path.join(train_conf.ckpt_dir, f"d_{epoch}.pt"))


def save_images(img_s, img_g, dir, epoch):
    # Image Saving Logic
    sample_size = min(8, img_s.shape[0])
    save_output = torch.clamp(img_g[:sample_size], -1., 1.).detach().cpu()
    save_output = ((save_output + 1) / 2)
    save_input = ((img_s[:sample_size] + 1) / 2).detach().cpu()

    grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
    img = torchvision.transforms.ToPILImage()(grid)
    if not os.path.exists(os.path.join(dir, 'vqvae_samples')):
        os.mkdir(os.path.join(dir, 'vqvae_samples'))
    img.save(os.path.join(dir, 'vqvae_samples',
                          f'sample_{epoch}.png'))
    img.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config',
                        default='configs/vqvae.yaml', type=str)
    args = parser.parse_args()
    train(args.config)
