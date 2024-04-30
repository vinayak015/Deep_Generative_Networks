import glob
from collections import OrderedDict
import torchvision
from PIL import Image

from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    def __init__(self, is_train, ):
        self.dataset_path = "/home/pc2/vinayak_dev/datasets/mnist"
        self.is_train = is_train
        self.images = glob.glob(f"{self.dataset_path}/{'train' if self.is_train else 'test'}/**/*.png", recursive=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = self.images[idx]
        label = im.split("/")[-2]
        img = Image.open(im)
        img = torchvision.transforms.ToTensor()(img)
        # convert to [-1,1]
        img = (2*img) - 1
        return OrderedDict(labels=label, images=img)

