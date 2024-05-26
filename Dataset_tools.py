# Dataset tools:
'''
This file contains the tools to create a parametric dataset for tabular and image data to train DGMs to assess their energy consumption.

The following class creates a dataset of images with sinusoidal waves of varying amplitude, frequency, and incedence. 

This code is written by Noah Bagazinski for the 2.834 class project

'''

# import the necessary packages
import numpy as np
import csv
import json
import matplotlib.pyplot as plt
from PIL import Image as im 
#Class Functions for dataset generation



class Wave:
    def __init__(self, Params):
        '''
        Params is an Array of the form: [[A, freq, theta, alpha, B], ...]
        
        Where 
        A is the amplitude, 
        freq is the frequency, (number of cycles per image width)
        theta is the incedence angle, 
        alpha is the phase shift 
        B is the offset
        
        This can repeat so that a Wave object is the sum of multiple waves.

        There are constriants on the values of the parameters.

        
        '''

        self.Vec = Params
        self.num_waves = len(Params)
        self.check_constraints()

    def check_constraints(self):
        '''
        This function checks that the parameters are within the constraints.
        '''
        max = np.sum(self.Vec[:,0]) + np.sum(self.Vec[:,4])
        min = np.sum(self.Vec[:,4]) - np.sum(self.Vec[:,0])
        cons = np.zeros((2,))
        if max > 255:
            cons[0] = 1
        if min < 0:
            cons[1] = 1
        return cons #Returns a vector of 1s and 0s where 1 indicates a constraint violation
    

    # Image Generation Functions
    def Gen_Image(self, img_size, path, label):
        '''
        This function generates an image of a wave with the given parameters at the given x and y locations.
        '''
        img = np.zeros((img_size, img_size))
    
        for i in range(0, self.num_waves):
            dir_vec = np.array([np.cos(self.Vec[i,2]),np.sin(self.Vec[i,2])])
            #print(dir_vec)

            omega = 2*np.pi*self.Vec[i,1]/img_size
            #print(omega)

            for j in range(0, img_size):
                for k in range(0, img_size):
                    # pixel += A*cos(freq*2*pi * [j,k] *dot* [cos(theta),sin(theta)] + alpha) + B
                    img[j,k] = img[j,k] + self.Vec[i,0]*np.cos(omega*np.vdot([j,k],dir_vec) + self.Vec[i,3]) + self.Vec[i,4]

        img = img.astype(np.uint8)
        image = im.fromarray(img)
        image.save(path+ '/' + label + '.png')
        
        return img
    
    
def norm_waves(Params):
    '''
    This function normalizes each wave so that the maximum and minimum of the set of waves is within [0,255] 
    '''
    num_samples = Params.shape[0]
    num_waves = Params.shape[1]

    for i in range(0,num_samples):
        max = np.sum(Params[i,:,0]) + np.sum(Params[i,:,4])
        #print(max)
        min = np.sum(Params[i,:,4]) - np.sum(Params[i,:,0])
        #print(min)

        if max > 255 or min < 0: #floor values to zero and ceiling values to 255
            Params[i,:,0] = 254*(Params[i,:,0])/(max-min) #make amplitude range from 0 to 127 ((max-min) = 2*sum(A))
            Params[i,:,4] = 254*(Params[i,:,4])/(max+min) #make offset range from 0 to 127 ((max+min) = 2*sum(B))

    return Params

def Gen_Rnd_Wave_Set(num_samples, num_waves):
    '''
    This function generates a random wave with the given constraints on the parameters.
    '''
    Params = np.zeros((num_samples, num_waves, 5))
    Params[:,:,0] = np.random.uniform(0,255, (num_samples, num_waves)) #Amplitude
    Params[:,:,1] = np.power(2.0,np.random.uniform(-1, 3, (num_samples, num_waves))) #Frequency
    Params[:,:,2] = np.random.uniform(-np.pi, np.pi, (num_samples, num_waves)) #Theta
    Params[:,:,3] = np.random.uniform(-np.pi, np.pi, (num_samples, num_waves)) #Alpha
    Params[:,:,4] = np.random.uniform(0, 255, (num_samples,num_waves)) #Offset

    Params = norm_waves(Params)

    return Params    