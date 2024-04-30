import os
import torch
from tqdm import tqdm
from torch.optim import Adam
from dataset import MnistDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np

from models.blocks import Unet
from linear_scheduler import LinearScheduler

device = torch.device("cuda")

# Ref: https://github.com/explainingai-code/DDPM-Pytorch


def train():
    scheduler = LinearScheduler(num_time_steps=1000, beta_start=0.0001, beta_end=0.02, device=device)
    mnist = MnistDataset(is_train=True)
    dataloader = DataLoader(mnist, batch_size=32, shuffle=True, drop_last=True, num_workers=4)

    model = Unet(in_channels=1).to(device)
    model.train()

    ckpt_dir = "ckpt"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    num_epochs = 40

    optim = Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()
    with torch.autograd.set_detect_anomaly(True):
        pbar = tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            losses = []
            ckpt_name = f"train_{epoch}.pt"
            desc = f"Epoch {epoch}"
            for data in dataloader:

                labels, images = data['labels'], data['images']
                optim.zero_grad()
                images = images.float().to(device)

                # sample noise
                noise = torch.randn_like(images).to(device)

                # sample timestep
                t = torch.randint(0, 1000, size=(images.shape[0],)).to(device)

                noisy_images = scheduler.add_noise(images, noise, t)

                noise_pred = model(noisy_images, t)

                loss = criterion(noise_pred, noise)
                losses.append(loss.item())

                loss.backward()
                desc = f"Epoch: {epoch}"
                pbar.set_description(desc)
                pbar.update(images.shape[0])
                optim.step()
            print('Finished epoch:{} | Loss : {:.4f}'.format(
                epoch + 1,
                np.mean(losses)))
            torch.save(model.state_dict(), os.path.join(ckpt_dir,
                                                        ckpt_name))
    pbar.close()


train()
