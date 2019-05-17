#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial import distance
import math
import random
#import numba
import time
from Noise import add_random_noise, compute_phi_star, cost, add_rand_noise_th
from KMeansOut import kmeansOutliers, cost
from KmeansPPcenters import KMeanPlusPlus
import matplotlib.pyplot as plt


def get_csv(fileName):
    return np.genfromtxt(fileName, delimiter=',')

def random_centers(data, num_clusters):
    centers= data[np.random.choice(np.arange(len(data)-1), num_clusters)]
    return centers

def  KPP_centers(data, num_clusters):
    random_indx= random.randint(0, len(data)-1)
    init=data[random_indx]
    KPP=KMeanPlusPlus(num_clusters=num_clusters, init=init)
    KPP.fit(data)
    KPP_centers= KPP.centers
    return KPP_centers

def add_noise(data, z, min_value, max_value):
    data_with_outliers, z_indx = add_random_noise(data, z, min_value, max_value)
    data_inliers= np.delete(data, z_indx, axis=0)
    return data_with_outliers, z_indx, data_inliers

def LloydOut(data, centers, num_clusters,z, tol, itr, z_indx):


    dist= distance.cdist(data, centers)
    dist = np.amin(dist, axis = 1)
    X, d= data.shape
    new_centers=np.zeros((num_clusters,d))
    for i in range(itr):
        print(i)
        dist= distance.cdist(data, np.array(centers))
        cid = np.argmin(dist, axis=1)
        dist = np.amin(dist, axis = 1)

        indx_list = np.argpartition(dist, -z)[-z:]
        data_new = np.delete(data, indx_list, axis=0)

        dist_new = distance.cdist(data_new, centers)
        cid = np.argmin(dist_new, axis=1)
        dist_new = np.amin(dist_new, axis = 1)

        for j in range(num_clusters):
            if len(data_new[cid==j])>0:
                new_centers[j]= np.mean(data_new[cid==j], axis=0)

        plt.figure()
        plt.scatter(data_new[:,0], data_new[:,1], c=cid)
        plt.scatter(centers[:,0], centers[:,1], marker='o', color= 'red')
        plt.scatter(new_centers[:,0], new_centers[:,1], marker='o', color= 'black')


        old_centers= centers.copy()
        centers=new_centers.copy()

        isOPTIMAL= True

        if (len(np.setdiff1d(new_centers,old_centers)))>tol:
            isOPTIMAL= False


        if isOPTIMAL:
            break
    #Step 10: Compute precision and recall
    precision = len(np.intersect1d(z_indx, indx_list))/len(z_indx)
    recall = len(np.intersect1d(z_indx, indx_list))/len(indx_list)
    #x1= KPP.predict(data_with_outliers)
        #print(("Precision:{}, recall:{}". format(precision, recall)))
    print(("Precision:{}, recall:{}". format(precision, recall)))
    return new_centers, cid, indx_list


def LO_cost(data, cid, centers, z):
    dist= distance.cdist(data, np.array(centers))
    dist = np.amin(dist, axis = 1)
    indx_list = np.argpartition(dist, -z)[-z:] #get index of farthest z points

    cid_pruned = cid.copy()
    cid_pruned[indx_list] = len(centers) + 1 # comment this line out if you do not want to remove points

    cost= np.zeros(len(centers))
    for i in range(len(centers)):
        cluster_indx = np.where(cid_pruned==i)
        cluster_points = data[cluster_indx]
        cost[i] = np.sum((cluster_points-centers[i])**2)
    total_cost= np.sum(cost)/len(data_new)

    return total_cost, indx_list
