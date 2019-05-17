'''
    File name:      Noise.py
    Description:    Contains functions with Random noise and
                    Random noise with a threshold
    Author:         Sharvaree V
    Last modified:  16th May 2019
    Python Version: 3.5
'''

import numpy as np
from scipy.spatial import distance
import random

def sign():
    a= random.randint(0,1)
    if a==0:
        a= -1
    return a


def add_random_noise(data, z, max_value, min_value):
    z_indx= np.random.choice(len(data)-1, z)#pick z random points
    for index in z_indx:
        noise= np.array([sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value)])
        data[index]= data[index]+ noise
    return data, z_indx


def add_rand_noise_th(data, z, max_value, min_value):
    z_indx= np.random.choice(len(data)-1, z)#pick z random points
    #x, d = data.shape
    for index in z_indx:
        noise= np.array([sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value)])
        data[index]= data[index]+ np.array(noise)
        data= np.clip(data, 0, 255)
    return data, z_indx

def add_rand_noise_general(data, z, min_value, max_value):
    z_indx= np.random.choice(len(data)-1, z)
    for index in z_indx:
        noise= np.random.uniform(min_value, max_value, data.shape)
        data[index]= data[index] + sign()*noise[index]
    return data, z_indx

def add_rand_noise_SUSY8(data, z, max_value, min_value):
    z_indx= np.random.choice(len(data)-1, z)#pick z random points
    #x, d = data.shape
    for index in z_indx:
        noise= np.array([sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value)])
        data[index]= data[index]+ np.array(noise)
    return data, z_indx

def add_rand_noise_SUSY10(data, z, max_value, min_value):
    z_indx= np.random.choice(len(data)-1, z)#pick z random points
    #x, d = data.shape
    for index in z_indx:
        noise= np.array([sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value),  sign()*np.random.uniform(min_value,max_value)])
        data[index]= data[index]+ np.array(noise)
    return data, z_indx

def add_rand_noise_SUSY18(data, z, max_value, min_value):
    z_indx= np.random.choice(len(data)-1, z)#pick z random points
    #x, d = data.shape
    for index in z_indx:
        noise= np.array([sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value),  sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value), sign()*np.random.uniform(min_value,max_value),
                         sign()*np.random.uniform(min_value,max_value),  sign()*np.random.uniform(min_value,max_value)])
        data[index]= data[index]+ np.array(noise)
    return data, z_indx
