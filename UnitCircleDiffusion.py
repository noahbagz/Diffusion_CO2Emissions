#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:39:10 2023

@author: shannon
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import TabDiffusionTools as TDT

# Genrate Data for a unit circle
factor = 2.5
num_samples = 10000

dataset = np.zeros((num_samples,2))

theta = np.linspace(0.0,2*np.pi,num_samples)

#dataset[:,0] = np.linspace(-1.0,1.0,num_samples)*factor #np.cos(theta)
dataset[:,0] = np.cos(theta)*factor
dataset[:,1] = np.sin(theta)*factor

DDPM_Dict = {
        'xdim' : 2,
        'X_LL' : [-factor,-factor],
        'X_UL' : [factor,factor],
        'ydim': 0,
        'cdim': 0,
        'datalength': num_samples,
        'tdim': 64,
        'net': [64,64,64],
        'batch_size': 128,
        'Training_Epochs': 100000,
        'Diffusion_Timesteps': 100,
        'lr' : 0.00025,                 # learning rate
        'weight_decay': 0.0,            # weight decay
        'device_name': 'cuda:0'}        # gpu device name}
'''
DDPM_Dict = {
        'xdim' : len(X[0]),             # Dimension of parametric design vector
        'datalength': len(X),           # number of samples
        'X_LL' : X_lower_lim,           # lower limits of parametric design vector variables
        'X_UL' : X_upper_lim,
        'ydim': len(Y_set[0]),          # Number of objectives
        'cdim': len(Cons[0]),           # number of classes for classifier
        'gamma' : 0.5,                  # weight of feasibility guidence for guided sampling
        'lambdas': [1,1,1,1,1,1,1],     # dummy variable for performance guided sampling
        'tdim': 128,                    # dimension of latent variable
        'net': [1024,1024,1024,1024],   # network architecture
        'batch_size': 1024,             # batch size
        'Training_Epochs': 100000,      # number of training epochs
        'Diffusion_Timesteps': 1000,    # number of diffusion timesteps
        'lr' : 0.00025,                 # learning rate
        'weight_decay': 0.0,            # weight decay
        'device_name': 'cuda:0'}        # gpu device name
'''




T = TDT.DiffusionEnv(DDPM_Dict, 
                X=dataset) 
                #Y=Y_set,
                #Cons = Con_set,
                #dataLength=num_samples,
                #lr = 0.00025, 
                #weight_decay=0.0,  
                #device=torch.device('cuda:0'))

T.run_train_loop(batches_per_epoch=1)


X_gen, unnorm = T.gen_samples(100)
#X_gen, unnorm_logbased = T.sample_all(100)


title = 'Distribution Dataset Parameters and Generated Parameters'

fig, axs = plt.subplots(1, 1, figsize=(3.5,3.5), dpi=160)

#plt.xlim(0,2000)

axs.spines["right"].set_visible(False)

axs.spines["top"].set_visible(False)

plt.rcParams['font.size'] = '6'       

axs.set_title(title, fontsize=10)

plt.xlabel('X0', fontsize = 10)

plt.ylabel('X1', fontsize = 10)

axs.plot(dataset[:,0], dataset[:,1], '-', color = 'red', label = 'Dataset')

axs.plot(X_gen[:,0], X_gen[:,1], 'o', color = 'blue', label = 'X_gen')

      

plt.legend(fontsize = 6, loc = 'upper right')

plt.show()