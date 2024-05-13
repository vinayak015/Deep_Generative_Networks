import glob
import os
from collections import OrderedDict

import numpy as np
import torchvision
from PIL import Image
import random
import torch
from torch.utils.data.dataset import Dataset


class CelebADataset(Dataset):
    def __init__(self, **kwargs):
        self.dataset_root = kwargs['root']
        self.im_path = kwargs['im_path']
        self.txt_path = kwargs['txt_path']
        self.masks_path = kwargs['masks_path']
        self.im_size = kwargs['im_size']
        self.mask_h = kwargs['mask_h']
        self.mask_w = kwargs['mask_w']


        # Should we use latents or not
        self.latent_maps = None
        self.use_latents = False
        self.condition_types = kwargs["condition_types"]
        self.images, self.texts, self.masks = self.load_images(self.dataset_root)

    def __len__(self):
        return len(self.images)

    def load_images(self, im_path_list):
        if 'image' in self.condition_types:
            label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                          'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
            self.idx_to_cls_map = {idx: label for idx, label in enumerate(label_list)}
            self.cls_to_idx_map = {label: idx for idx, label in enumerate(label_list)}
            masks = glob.glob(f"{self.dataset_root}/{self.masks_path}/**/*.png", recursive=True)

        images = glob.iglob(f"{self.dataset_root}/{self.im_path}/**/*.jpg", recursive=True)
        images = sorted(images)

        texts = glob.iglob(f"{self.dataset_root}/{self.txt_path}/**/*.txt", recursive=True)
        texts = sorted(texts)
        texts = [open(text, 'r').readlines() for text in texts]

        return images, texts, masks

    def get_masks(self, idx):
        mask = np.array(Image.open(self.masks[idx]))
        if self.mask_h is None:
            self.mask_h, self.mask_w = mask.shape
        base_mask = np.zeros((self.mask_h, self.mask_w, len(self.cls_to_idx_map)))

        for idx in range(len(self.idx_to_cls_map)):
            base_mask[mask == idx+1, idx] = 1.0

        mask = torch.from_numpy(base_mask).permute(2, 0, 1).float()

        return mask

    def __getitem__(self, idx):
        text = random.sample(self.texts[idx], k=1)[0].strip()

        im = self.images[idx]
        im = Image.open(im)
        im_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.im_size),
            torchvision.transforms.CenterCrop(self.im_size),
            torchvision.transforms.ToTensor(),
        ])(im)
        im.close()
        im_tensor = (im_tensor * 2.0) - 1.0

        mask = self.get_masks(idx)

        return OrderedDict(img= im_tensor, text= text, mask=mask)


