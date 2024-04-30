import torch
from linear_scheduler import LinearScheduler
from torchvision.utils import make_grid
import torchvision
import os
import imageio.v2 as imageio
from tqdm import tqdm
import re

from models.blocks import Unet

device = "cuda"


class DDPMSample(LinearScheduler):
    def __init__(self, num_time_steps, beta_start, beta_end, device, model):
        super().__init__(num_time_steps=num_time_steps, beta_start=beta_start, beta_end=beta_end, device=device)
        self.model = model
        self.model.eval()

    def sample(self, ):
        x_t = torch.randn([64, 1, 28, 28]).to(self.device)  # static for time being now
        pbar = tqdm(total=self.num_time_steps)
        desc = f"TimeStep: "
        pbar.set_description(desc)
        for t in range(self.num_time_steps - 1, -1, -1):
            t = torch.tensor(t).repeat(64).to(self.device)
            with torch.no_grad():
                noise = self.model(x_t, t)
            x_t, x_0 = self.sample_previous(x_t, t, noise)
            pbar.update(n=1)
            # Visualization
            if t[0] % 50 == 0:
                self.visualize(t, x_t)
        self.visualize(t, x_0)
        pbar.close()

    def visualize(self, t, x_t):
        x_t_save = torch.clamp(x_t, -1., 1.).detach().cpu()
        x_t_save = (x_t_save + 1) / 2
        grid = make_grid(x_t_save, nrow=8)
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join("visualization", "samples")):
            os.makedirs(os.path.join("visualization", "samples"))
        img.save(os.path.join("visualization", "samples", f"x_{t[0]}.png"))
        img.close()


def numerical_sort(value):
    """
    Extracts numbers from a string and returns it to aid in sorting.
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])  # convert all numerical strings to integers
    return parts

def create_animation(input_folder, output_file):
    images = []
    for file_name in sorted(os.listdir(input_folder), reverse=True, key=numerical_sort):
        if file_name.endswith('.png'):
            file_path = os.path.join(input_folder, file_name)
            images.append(imageio.imread(file_path))
    if not os.path.exists(output_file):
        os.makedirs("/".join(output_file.split('/')[:-1]))
    imageio.mimsave(output_file, (images), fps=20)

def start_sampling():
    model = Unet(1).to(device)
    model.load_state_dict(torch.load("ckpt/train_39.pt", map_location=device))
    scheduler = DDPMSample(num_time_steps=1000, beta_start=0.0001, beta_end=0.02, device=device, model=model)
    scheduler.sample()

input_folder = 'visualization/samples'
output_file = 'visualization/animation/animation.gif'


start_sampling()
create_animation(input_folder, output_file)
