#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.spatial import distance
import math
import random
import time


# In[7]:


def get_csv(fileName):
    return np.genfromtxt(fileName, delimiter=',')

def update_dist(dist, tempdist, cid, data, ix):
    for indx in range(len(data)):
        if dist[indx] <= tempdist[indx]:
            continue
        else:
            dist[indx] = tempdist[indx]
            cid[indx] = ix
    return dist, cid


def kmeanspp(data,num_clusters):
    
    # Random initialization
    wins = [random.randint(0, len(data)-1)]
    centers = np.array([data[wins[0]]]) 
    
    # set size of points as max numbers for dist array
    dist = np.array([1.7e308]*len(data))
    
    #index
    cid =  np.zeros(len(data))
    
    for i in range(num_clusters-1):
        #Calculate distance to new center
        tempdist = distance.cdist(data, np.array([centers[len(centers)-1]]))
        tempdist = np.array([item[0] for item in tempdist])
        
        #keep only the distance min(dist, tempdist)
        dist, cid = update_dist(dist, tempdist, cid, data, i)
        
        #Normalizing
        tempSum = sum(dist)
        distribution = dist/tempSum
        
        #Picking new center with the above distribution
        new_index = np.random.choice(len(data), 1, p=distribution)
        wins.append(new_index[0])
        
        #Adding new center
        centers = np.append(centers, data[new_index], axis = 0)

    return centers


def cost(data, cid, centers, z):

    dist= distance.cdist(data, np.array(centers))
    dist = np.amin(dist, axis = 1)
    indx_list = np.argpartition(dist, -z)[-z:] #get index of farthest z points
    
    cid_pruned = cid.copy()
    cid_pruned[indx_list] = len(centers) + 1 # comment this line out if you do not want to remove points

    cost= np.zeros(len(centers))
    for i in range(len(centers)):
        cluster_indx = np.where(cid_pruned==i)
        cluster_points = data[cluster_indx]
        cost[i] = np.mean((cluster_points-centers[i])**2)
        
    return cost, indx_list

def cost2(data, centers,z):
    dist= distance.cdist(data, np.array(centers))
    print("Shape:", dist.shape)
    dist = np.amin(dist, axis = 1)
    print("Shape:", dist.shape)
    s = np.sort(dist)
    s = s[:len(dist) - z- 1]
    print("Cost:", np.sum(s))
    return np.sum(s)


