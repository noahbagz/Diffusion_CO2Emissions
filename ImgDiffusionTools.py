
import numpy as np
import json
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


from torch.optim import Adam

import sklearn.preprocessing as PP

class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def closed_form_solution(T, betas):
  # Pre-calculate different terms for closed form
  alphas = 1. - betas
  alphas = torch.tensor(alphas)
  alphas_cumprod = torch.cumprod(alphas, axis=0)
  alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
  sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
  sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
  sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

  return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod






class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
        self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self,image_size, mult = 1, t_emb_dim = 32):
        super().__init__()
        image_channels = 1
        #down_channels = (16*mult, 32*mult, 64*mult, 128*mult)
        #up_channels = (128*mult, 64*mult, 32*mult, 16*mult)
        self.down_channels = [8*mult,16*mult]
        self.up_channels = [16*mult, 8*mult]
        out_dim = 1
        time_emb_dim = t_emb_dim 

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, self.down_channels[0], kernel_size=3, padding=2)

        # Downsample
        self.downs = nn.ModuleList([DownBlock(self.down_channels[i], self.down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(self.down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([UpBlock(self.up_channels[i], self.up_channels[i+1], \
                                        time_emb_dim) \
                    for i in range(len(self.up_channels)-1)])

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(self.up_channels[-1], out_dim, 3, padding=0)

    def forward(self, x, timestep):

        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        
        x = x #[:, :, :, :self.down_channels[0]]

        # Unet
        residual_inputs = []
        for down in self.downs:
           
            x = down(x, t)
            
            residual_inputs.append(x)
            

        for up in self.ups:
            residual_x = residual_inputs.pop()
            #print(residual_x.shape)

            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        out = self.output(x)
        return out#[:, :, :, :self.down_channels[0]+1]
    


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class Img_DDPM_Env:
    def __init__(self, img_size, mult, t_emb_dim, timesteps, device):
        self.timesteps = timesteps
        self.img_size = img_size
        self.device = device

        

        self.model = SimpleUnet(img_size, mult=mult, t_emb_dim = t_emb_dim).to(device)
        self.ema = EMA(0.99)
        self.ema.register(self.model)

        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        # Pre-calculate different terms for closed form
        self.betas = torch.linspace(0.001, 0.2, self.timesteps).to(self.device)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    
    def forward_diffusion_sample(self,x_0, t):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        x_0: (batch_size, 1, 30, 65)

        """
        
        # create noise tensor
        noise = torch.randn_like(x_0)

        # for calculation purpose take everything to the gpu or cpu --> same device
        noise, t = noise.to(self.device), t.to(self.device)



        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)


        # mean
        forward_mean = sqrt_alphas_cumprod_t * x_0

        # variance
        forward_variance = sqrt_one_minus_alphas_cumprod_t

        # reparameterization
        sample = forward_mean + forward_variance * noise

        return sample, noise
    

    def sample_timestep(self, x, t):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        # Define beta schedule
        T = self.timesteps  # total diffusion steps


        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

    

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    @torch.no_grad()
    def sample_images(self,num_samples):
        
        self.model.eval()
        
        # Sample noise
        img = torch.randn((num_samples, 1, self.img_size,self.img_size), device=self.device)

        

        for i in tqdm(range(0,self.timesteps)):
            t = torch.full((1,), 0, device=self.device, dtype=torch.int64)
            img = self.sample_timestep(img, t)
            # Edit: This is to maintain the natural range of the distribution
            img = torch.clamp(img, -1.0, 1.0)

        return ((img.cpu().numpy() +1.0)/2.0 *255).astype(np.uint8)

    def train_loop_patience(self,patience,train_loader):
        # Set Up training loop variables
        min_loss = 1e10
        min_loss_epoch = 1e10
        patience_counter = 0
        epochs = 0
        
        self.model.train()

        while (patience_counter < patience):


            
            for batch in train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)

                t = torch.randint(0, self.timesteps, (batch.size()[0],), device=self.device)
                loss = self.get_loss( batch, t)
                loss.backward()
                self.optimizer.step()
                min_loss_epoch = min(loss.item(), min_loss_epoch)
           

            if min_loss_epoch < min_loss:
                min_loss = min_loss_epoch
                patience_counter = 0
            else:
                patience_counter += 1

            self.ema.update(self.model)
            if epochs % 500 == 0:
                print("Epoch: {} Loss: {}".format(epochs, min_loss_epoch))
            min_loss_epoch = 1e10
            epochs += 1
        return epochs
    
    def get_loss(self, x_0, t):
        
        x_noisy, noise = self.forward_diffusion_sample(x_0, t)
        x_noisy = x_noisy.float()
        noise_pred = self.model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    

    




