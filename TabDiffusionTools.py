#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:10:55 2023

@author: shannon

I am trying my best to reconstruct the tabular diffusion, tab-ddpm,
to allow for multi output tabular diffusion


Code borrowed from https://www.kaggle.com/code/grishasizov/simple-denoising-diffusion-model-toy-1d-example
"""

import numpy as np
import json
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sklearn.preprocessing as PP


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def timestep_embedding(timesteps, dim, max_period=10000, device=torch.device('cuda:0')):
    """
    From https://github.com/rotot0/tab-ddpm
    
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1,device=device)
    return embedding

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def permute_idx(n):
   #n is the size of the array
   arr = list(range(0,n))
   for i in range(n - 1, 0, -1):
        # Generate a random index from 0 to i (inclusive)
        j = np.random.randint(0, i)
        
        # Swap the elements at positions i and j
        arr[i], arr[j] = arr[j], arr[i]
    
   return arr

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))



# Now lets make a Denoise Model:
    
class Denoise_MLP_Model(torch.nn.Module):
    def __init__(self, DDPM_Dict, device=torch.device('cuda:0')):
        nn.Module.__init__(self)
        
        self.xdim = DDPM_Dict['xdim']
        self.ydim = DDPM_Dict['ydim']
        self.cdim = DDPM_Dict['cdim']
        self.tdim  = DDPM_Dict['tdim']
        self.net = DDPM_Dict['net']
        self.device = device
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        
        self.fc.append(nn.Sequential(nn.Linear(self.tdim, self.xdim)))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        
        '''
        self.Y_embed = nn.Sequential(
            nn.Linear(self.ydim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
        
        self.Con_embed = nn.Sequential(
                    nn.Linear(self.cdim, self.tdim),
                    nn.SiLU(),
                    nn.Linear(self.tdim, self.tdim))
        
        '''
        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
       
        
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.LayerNorm(dimo),
                             nn.Dropout(p=0.1))
        

    def forward(self, x, timesteps):
        a = self.X_embed(x)
        #print(a.dtype)
        x = a +  self.time_embed(timestep_embedding(timesteps, self.tdim))
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    

        return x
    
class Denoise_ResNet_Model(torch.nn.Module):
    def __init__(self, DDPM_Dict, device=torch.device('cuda:0')):
        nn.Module.__init__(self)
        
        self.xdim = DDPM_Dict['xdim']
        self.ydim = DDPM_Dict['ydim']
        self.tdim  = DDPM_Dict['tdim']
        self.cdim = DDPM_Dict['cdim']
        self.net = DDPM_Dict['net']
        self.device = device
        
        self.fc = nn.ModuleList()
        
        self.fc.append(self.LinLayer(self.tdim,self.net[0]))
        
        for i in range(1, len(self.net)):
            self.fc.append(self.LinLayer(self.net[i-1],self.net[i]))
            
        self.fc.append(self.LinLayer(self.net[-1], self.tdim))
        
        
        self.finalLayer = nn.Sequential(nn.Linear(self.tdim, self.xdim))
        
    
        self.X_embed = nn.Linear(self.xdim, self.tdim)
        
        '''
        self.Y_embed = nn.Sequential(
            nn.Linear(self.ydim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
        
        self.Con_embed = nn.Sequential(
                    nn.Linear(self.cdim, self.tdim),
                    nn.SiLU(),
                    nn.Linear(self.tdim, self.tdim))
        
        '''
        self.time_embed = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.SiLU(),
            nn.Linear(self.tdim, self.tdim))
       
        
    def LinLayer(self, dimi, dimo):
        
        return nn.Sequential(nn.Linear(dimi,dimo),
                             nn.SiLU(),
                             nn.BatchNorm1d(dimo),
                             nn.Dropout(p=0.1))
        


    def forward(self, x, timesteps):
        
                
        x = self.X_embed(x) + self.time_embed(timestep_embedding(timesteps, self.tdim)) 
        res_x = x
        
        for i in range(0,len(self.fc)):
            x = self.fc[i](x)
    
        x = torch.add(x,res_x)
        
        x = self.finalLayer(x)
        
        return x


    
'''
==============================================================================
EMA - Exponential Moving Average: Helps with stable training
========================================================================
EMA class from: https://github.com/azad-academy/denoising-diffusion-model/blob/main/ema.py

'''
# Exponential Moving Average Class
# Orignal source: https://github.com/acids-ircam/diffusion_models


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
        
        
'''
==========================================
Set up the data normalizer class
==========================================

'''        

class Data_Normalizer:
    def __init__(self, X_LL_Scaled, X_UL_Scaled,datalength):
        
        self.normalizer = PP.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(datalength // 30, 1000), 10),
            subsample= int(1e9)
            )
        
        self.X_LL_Scaled = X_LL_Scaled
        self.X_UL_Scaled = X_UL_Scaled
        
        self.X_LL_norm = np.zeros((1,len(X_LL_Scaled)))
        self.X_UL_norm = np.zeros((1,len(X_LL_Scaled)))
        
        self.X_mean = np.zeros((1,len(X_LL_Scaled)))
        self.X_std = np.zeros((1,len(X_LL_Scaled)))
        
    def fit_Data(self,X):
        
        
        
        x = 2.0*(X-self.X_LL_Scaled)/(self.X_UL_Scaled- self.X_LL_Scaled) - 1.0
        
        self.normalizer.fit(x)
        x = self.normalizer.transform(x) # Scale Dataset between 
        #x = (X-self.X_LL_Scaled)/(self.X_UL_Scaled- self.X_LL_Scaled)
        

        '''
        self.normalizer.fit(x)
        x = self.normalizer.transform(x) # Scale Dataset between 
        #self.X_LL_norm = np.amin(x,axis = 0)
        #self.X_UL_norm = np.amax(x,axis=0)
        
        # Fit x to be between -1 and 1
        
        #x = 2.0*(x-self.X_LL_norm)/(self.X_UL_norm - self.X_LL_norm) - 1.0
        #x = (x-self.X_LL_norm)/(self.X_UL_norm - self.X_LL_norm)
        '''
        # lets try z score normalization
        

        '''
        self.X_mean = np.mean(X,axis=0)
        
        self.X_std = np.std(X,axis=0)
        
        z = (X - self.X_mean)/self.X_std
        
        #self.X_LL_norm = np.amin(z,axis = 0)
        #self.X_UL_norm = np.amax(z,axis=0)
       
        Z = 2.0*(z-4.0)/(8.0) - 1.0
        '''
        return x
        

    def scale_X(self,z):
        #rescales data
        z = self.normalizer.inverse_transform(z)
        scaled = (z + 1.0) * 0.5 * (self.X_UL_Scaled - self.X_LL_Scaled) + self.X_LL_Scaled
        #scaled = z* (self.X_UL_Scaled - self.X_LL_Scaled) + self.X_LL_Scaled

        '''
        x = self.normalizer.inverse_transform(x)
        
        #scaled = x* (self.X_UL_norm - self.X_LL_norm) + self.X_LL_norm
        '''
        #z = (z + 1.0) * 0.5 * (8.0) + 4.0
       
        #scaled = z*self.X_std + self.X_mean
        #scaled = self.normalizer.inverse_transform(scaled)
        return scaled     
        
        
        
        
'''
=======================================================================
Trainer class modified from Tab-ddpm paper code with help from hugging face
=====================================================================
'''
class DiffusionEnv:
    def __init__(self, DDPM_Dict, X):
        
        
        self.DDPM_Dict = DDPM_Dict
        
        self.device =torch.device(self.DDPM_Dict['device_name'])
        
        #Build the Diffusion Network
        self.diffusion = Denoise_ResNet_Model(self.DDPM_Dict,device=self.device)

        self.dataLength = self.DDPM_Dict['datalength']
        self.batch_size = self.DDPM_Dict['batch_size']
        
        self.data_norm = Data_Normalizer(np.array(self.DDPM_Dict['X_LL']),np.array(self.DDPM_Dict['X_UL']),self.dataLength)
        
        
        
        self.X = self.data_norm.fit_Data(X)
        
        #X and Y are numpy arrays - convert to tensor
        self.X = torch.from_numpy(self.X.astype('float32'))
        #self.Y = torch.from_numpy(Y.astype('float32'))
        #self.Cons = torch.from_numpy(Cons.astype('float32'))
        
        
         
        self.diffusion.to(self.device)
        
        
    
        self.X = self.X.to(self.device)
        #self.Y = self.Y.to(self.device)
        #self.Cons = self.Cons.to(self.device)
        
        self.eps = 1e-8
        
        self.ema = EMA(0.99)
        self.ema.register(self.diffusion)
        
        
        #set up optimizer 
        self.timesteps = self.DDPM_Dict['Diffusion_Timesteps']
        self.num_epochs = self.DDPM_Dict['Training_Epochs']
        self.init_lr = self.DDPM_Dict['lr']
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=self.init_lr, weight_decay=self.DDPM_Dict['weight_decay'])
       
        
        self.log_every = 100
        self.print_every = 500
        self.loss_history = []
        
        
        
        #Set up alpha terms
        self.betas = torch.linspace(0.001, 0.2, self.timesteps).to(self.device)
        
        #self.betas = betas_for_alpha_bar(self.timesteps, lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,)
        #self.betas = torch.from_numpy(self.betas.astype('float32')).to(self.device)
        
        self.alphas = 1. - self.betas
        
        self.log_alpha = torch.log(self.alphas)
        self.log_cumprod_alpha = np.cumsum(self.log_alpha.cpu().numpy())
        
        self.log_cumprod_alpha = torch.tensor(self.log_cumprod_alpha,device=self.device)
        
        self.log_1_min_alpha = log_1_min_a(self.log_alpha)
        self.log_1_min_cumprod_alpha = log_1_min_a(self.log_cumprod_alpha)
        
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1],[1,0],'constant', 0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod =  torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        a = torch.clone(self.posterior_variance)
        a[0] = a[1]                 
        self.posterior_log_variance_clipped = torch.log(a)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev)* torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))
         
        self.mean_zeros = torch.full((self.diffusion.xdim,),0.0).to(self.device) 
        self.var_sample = torch.full((self.diffusion.xdim,),1.0).to(self.device)
    """++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Start the training model functions
    """
    def extract(self,a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)
        out = a.gather(-1, t)
        while len(out.shape) < len(x_shape):
            out = out[..., None]
        return out.expand(x_shape)
    
    def _anneal_lr(self, epoch_step):
        #Update the learning rate
        frac_done = epoch_step / self.num_epochs
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
    '''
    GAUSSIAN DIFFUSION PARAMETERS FROM TAB-DDPM
    
    '''        
        # Gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = self.extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self.extract(
            self.log_1_min_cumprod_alpha, t, x_start.shape
        )
        return mean, variance, log_variance
    
    def gaussian_q_sample(self, x_start, t, noise):

      
        return (
            self.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self, model_output, x, t, clip_denoised=False, denoised_fn=None, model_kwargs=None
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        
        
        assert t.shape == (B,)

        a = torch.clone(self.betas)
        
        a[0] = self.posterior_variance[1]
        
        model_variance = a
        # model_variance = self.posterior_variance.to(x.device)
        model_log_variance = torch.log(model_variance)
        

        model_variance = self.extract(model_variance, t, x.shape)
        
        
        model_log_variance = self.extract(model_log_variance, t, x.shape)

    
 
        pred_xstart = model_output
        #pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

            
        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f'{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}'

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }        
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            -self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def gaussian_p_sample(
        self,
        model_out,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
    ):
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
    
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return sample
        
    def gaussian_loss(self, x,t):

        noise = torch.randn_like(x)
        
        
        x_num_t = self.gaussian_q_sample(x, t, noise=noise)
    
        
        model_out = self.diffusion(x_num_t,t)
        
        #model_out = model_out.clamp(-3,3)
        
        
        loss = F.mse_loss(noise, model_out)
        #loss = mean_flat((noise - model_out) ** 2)
        #loss = self._vb_terms_bpd(model_out, x, x_num_t, t)['output']

        return loss
    
    def _vb_terms_bpd(self, model_output, x_start, x_t, t,clip_denoised=False, model_kwargs=None):
        true_mean, _, true_log_variance_clipped = self.gaussian_q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = torch.mean(kl) / np.log(2.0)
    
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = torch.mean(decoder_nll) / np.log(2.0)
    
        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"], "out_mean": out["mean"], "true_mean": true_mean}

   
    def train_step(self,x):
        
        self.optimizer.zero_grad()
        t = torch.randint(0,self.timesteps,(self.batch_size,),device=self.device)
        loss = self.gaussian_loss(x,t)
        loss.backward()
        self.optimizer.step()
        return loss
        
    @torch.no_grad()
    def sample(self,num_samples):
        b = num_samples
    
        z_norm = torch.randn((b, self.diffusion.xdim), device=self.device)
    

        for i in reversed(range(0, self.timesteps)):
            #print(f'Sample timestep {i:4d}', end='\r')
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            model_out = self.diffusion(z_norm,t)
            
            #model_out = model_out.clamp(-3,3)
            z_norm = self.gaussian_p_sample(model_out, z_norm, t)
            #z_norm = z_norm.clamp(-3,3)

        return z_norm
    
    
    def sample_all(self, num_samples):
        self.diffusion.eval()
        
        x_gen = self.sample(num_samples)
        
        output = x_gen.cpu().detach().numpy()
            
            
        output_scaled = self.data_norm.scale_X(output) 
        
        return output_scaled, output   
        
                
    
    '''
    =========================================================================
    Vanilla Diffusion
    ==========================================================================
    '''
    def q_sample(self,x_start, t, noise=None):
        """
        qsample from https://huggingface.co/blog/annotated-diffusion
        """
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
    
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
    
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    
    def p_loss(self,x_start,t, noise=None,loss_type='l1'):
        '''
        from https://huggingface.co/blog/annotated-diffusion
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.diffusion(x_noisy, t)
        
        #predicted_noise = predicted_noise.clamp(-3,3)
        
        if loss_type == 'l1':
            loss1 = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss1 = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss1 = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        
        #loss2 = F.mse_loss(predicted_noise.mean(dim=0),self.mean_zeros) #batch mean loss
        
        #loss3 = F.mse_loss(predicted_noise.var(dim=0),self.var_sample) #batch mean variance loss suggesting tha
        
        
        return loss1 #, loss2 #, loss3
        


    def run_step(self, x):
        self.optimizer.zero_grad()
        
        t = torch.randint(0,self.timesteps,(self.batch_size,),device=self.device)
        loss1 = self.p_loss(x,t,loss_type='l2')
        
        loss = loss1 #+ loss2 #+ loss3
        loss.backward()
        self.optimizer.step()

        return loss

    def run_train_loop(self, batches_per_epoch=100):
        
        self.diffusion.train()
        
        num_batches = self.dataLength // self.batch_size
        
        batches_per_epoch = min(num_batches,batches_per_epoch)
        
    
        x_batch = torch.full((batches_per_epoch,self.batch_size,self.diffusion.xdim), 0, dtype=torch.float32,device=self.device)
        #y_batch = torch.full((batches_per_epoch,self.batch_size,self.diffusion.ydim), 0, dtype=torch.float32,device=self.device)
        #cons_batch = torch.full((batches_per_epoch,self.batch_size,self.diffusion.cdim), 0, dtype=torch.float32,device=self.device)
        
        for i in tqdm(range(self.num_epochs)):
            
            #IDX = permute_idx(self.dataLength) # get randomized list of idx for batching
            
            for j in range(0,batches_per_epoch):
                
                A = np.random.randint(0,self.dataLength,self.batch_size)
                x_batch[j] = self.X[A] 
                #y_batch[j] = self.Y[IDX[j*self.batch_size:(j+1)*self.batch_size]] 
                #cons_batch[j] = self.Cons[IDX[j*self.batch_size:(j+1)*self.batch_size]]
            
            for j in range(0,batches_per_epoch):
                       
                loss = self.run_step(x_batch[j])
                '''
                Gaussian Diffusion (oooohhhh ahhhhhh) from TabDDPM:
                '''
                #loss = self.train_step(x_batch[j])

            self._anneal_lr(i)

            if (i + 1) % self.log_every == 0:
                self.loss_history.append([i+1,float(loss.to('cpu').detach().numpy())])
        
            if (i + 1) % self.print_every == 0:
                    print(f'Step {(i + 1)}/{self.num_epochs} Loss: {loss}')
                

            self.ema.update(self.diffusion)
        #Make Loss History an np array
        self.loss_history = np.array(self.loss_history)

    def run_train_loop_patience(self, batches_per_epoch=100, patience = 500):

        min_loss = 1e10
        min_loss_epoch = 1e10
        patience_counter = 0
        
        self.diffusion.train()
        
        num_batches = self.dataLength // self.batch_size
        
        batches_per_epoch = min(num_batches,batches_per_epoch)
        
    
        x_batch = torch.full((batches_per_epoch,self.batch_size,self.diffusion.xdim), 0, dtype=torch.float32,device=self.device)
        #y_batch = torch.full((batches_per_epoch,self.batch_size,self.diffusion.ydim), 0, dtype=torch.float32,device=self.device)
        #cons_batch = torch.full((batches_per_epoch,self.batch_size,self.diffusion.cdim), 0, dtype=torch.float32,device=self.device)
        
        #for i in tqdm(range(self.num_epochs)):
        i = 0
        while patience_counter < patience:

            
            #IDX = permute_idx(self.dataLength) # get randomized list of idx for batching
            
            for j in range(0,batches_per_epoch):
                
                A = np.random.randint(0,self.dataLength,self.batch_size)
                x_batch[j] = self.X[A] 
                #y_batch[j] = self.Y[IDX[j*self.batch_size:(j+1)*self.batch_size]] 
                #cons_batch[j] = self.Cons[IDX[j*self.batch_size:(j+1)*self.batch_size]]
            
            for j in range(0,batches_per_epoch):
                       
                loss = self.run_step(x_batch[j])
                min_loss_epoch = min(min_loss_epoch,loss)
                '''
                Gaussian Diffusion (oooohhhh ahhhhhh) from TabDDPM:
                '''
                #loss = self.train_step(x_batch[j])
            i+=1
            patience_counter += 1
            if min_loss_epoch < min_loss:
                min_loss = min_loss_epoch
                patience_counter = 0

            self._anneal_lr(i)

            if (i + 1) % self.log_every == 0:
                self.loss_history.append([i+1,float(loss.to('cpu').detach().numpy())])
        
            if (i + 1) % self.print_every == 0:
                    print(f'Step {(i + 1)}/{self.num_epochs} Loss: {loss}')
                

            self.ema.update(self.diffusion)
        #Make Loss History an np array
        self.loss_history = np.array(self.loss_history)

        return i
    
    @torch.no_grad()
    def p_sample(self, x, t):
        
        time= torch.full((x.size(dim=0),),t,dtype=torch.int64,device=self.device)
        
        betas_t = self.extract(self.betas, time, x.shape)
        
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, time, x.shape
        )
        
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, time, x.shape)
        
        # Equation 11 in the paper
        
        X_diff = self.diffusion(x, time) 
        #X_diff.clamp(-3,3)
        
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * X_diff/ sqrt_one_minus_alphas_cumprod_t
        )
    
        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, time, x.shape)
            noise = torch.randn_like(x,device=self.device)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
       
    @torch.no_grad()
    def gen_samples(self, num_samples):
        #COND is a numpy array of the conditioning it is shape (num_samples,conditioning terms)
        #num_samples = len(y)
        
        #y = torch.from_numpy(y.astype('float32'))
        #y = y.to(self.device)
        
        #cons = torch.from_numpy(cons.astype('float32'))
        #cons = cons.to(self.device)
        
        
        x_gen = torch.randn((num_samples,self.diffusion.xdim),device=self.device)
        
        self.diffusion.eval()
   
        for i in tqdm(range(self.timesteps - 1, 0, -1)):
            
          
            x_gen = self.p_sample(x_gen, i)
            #x_gen = x_gen.clamp(-1,1)
        
        output = x_gen.cpu().detach().numpy()
            
            
        output_scaled = self.data_norm.scale_X(output)
        
        return output_scaled, output
          
    

    def load_trained_model(self,PATH):
        #PATH is full path to the state dictionary, including the file name and extension
        self.diffusion.load_state_dict(torch.load(PATH))
    
    def Load_DDPM_Dict(PATH):
        #returns the dictionary for the DDPM_Dictionary to rebuild the model
        #PATH is the path including file name and extension of the json file that stores it. 
        f = open(PATH)
        return json.loads(f)
        
    
    def Save_model(self,PATH,name):
        '''
        PATH is the path to the folder to store this in, including '/' at the end
        name is the name of the model to save without an extension
        '''
        torch.save(self.diffusion.state_dict(), PATH+name+'.pth')
        
        JSON = json.dumps(self.DDPM_Dict)
        f = open(PATH+name+'_DDPM_Dict.json', 'w')
        f.write(JSON)
        f.close()
    



