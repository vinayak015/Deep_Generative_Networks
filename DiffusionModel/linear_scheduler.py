import torch


class LinearScheduler:
    def __init__(self, num_time_steps, beta_start, beta_end, device):
        self.num_time_steps = num_time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_time_steps,device=device)
        self.alphas = (1. - self.betas).to(device)

        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar).to(device)
        self.device = device

    def add_noise(self, x_0, noise, t):
        # xt = sqrt(alpha_bar)x0 + sqrt(1-alpha_bar) * eps
        b, c, h, w = x_0.shape
        x_t = self.sqrt_alpha_bar[t].reshape(b, 1, 1, 1).to(x_0.device) * x_0 + torch.sqrt(1 - self.alpha_bar[t]).reshape(b, 1, 1, 1).to(x_0.device) * noise
        return x_t

    def sample_previous(self, x_t, t, noise_pred):
        b, c, h, w = x_t.shape
        # variance = (1-self.alpha[t])*(1-self.alpha_bar[t-1]) / (1 - self.alpha_bar[t])
        x_0 = (x_t - (torch.sqrt(1. - self.alpha_bar[t].view(b, 1, 1, 1)) * noise_pred)) / torch.sqrt(self.alpha_bar[t].view(b, 1, 1, 1))
        x_0 = torch.clamp(x_0, -1., 1.)

        mean = (x_t - (self.betas[t].view(b, 1, 1, 1) * noise_pred / torch.sqrt(1. - self.alpha_bar[t].view(b, 1, 1, 1)))) / torch.sqrt(self.alphas[t].view(b, 1, 1, 1))

        if t[0] == 0:
            return mean, x_0
        variance = ((self.betas[t]) * (1. - self.alpha_bar[t - 1])) / (1. - self.alpha_bar[t])
        sigma = variance ** 0.5
        z = torch.randn_like(x_t)
        return mean + sigma.view(b, 1, 1, 1) * z, x_0

