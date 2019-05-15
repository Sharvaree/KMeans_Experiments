#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from scipy.spatial import distance

def compute_phi_star(data, num_clusters, centers, z):
    dist_matrix= distance.cdist(data, centers)
    dist = np.amin(dist_matrix, axis = 1)
    phi_star=np.sum(dist)
    return phi_star

def add_random_noise(data, z, max_value, min_value):
    z_indx= np.random.choice(len(data)-1, z)#pick z random points
    #x, d = data.shape
    for index in z_indx:
        noise= np.random.uniform(max_value, min_value)
        data[index]= data[index]+ noise
    return data, z_indx

def cost(data, cid, centers, z):

    dist= distance.cdist(data, np.array(centers))
    dist = np.amin(dist, axis = 1)
    indx_list = np.argpartition(dist, -z)[-z:] #get index of farthest 2z points
    
    cid_pruned = cid.copy()
    cid_pruned[indx_list] = len(centers) + 1 # comment this line out if you do not want to remove points

    cost= np.zeros(len(centers))
    for i in range(len(centers)):
        cluster_indx = np.where(cid_pruned==i)
        cluster_points = data[cluster_indx]
        cost[i] = np.mean((cluster_points-centers[i])**2)
        
    return cost, indx_list


def add_rand_noise_th(data, z, max_value, min_value):
    z_indx= np.random.choice(len(data)-1, z)#pick z random points
    #x, d = data.shape
    for index in z_indx:
        noise= np.random.uniform(max_value, min_value)
        data[index]= data[index]+ noise
        data= np.clip(data, 0, 255)
    return data, z_indx

