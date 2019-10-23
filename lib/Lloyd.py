'''
    File name:      LLoyds.py
    Description:    Classic Lloyd's algorithm (without Outliers)
    Author:         Sharvaree V
    Last modified:  20th May 2019
    Python Version: 3.5
'''

import numpy as np
from scipy.spatial import distance
import math
import random
#import numba
import time
#from Noise import add_random_noise, compute_phi_star, cost, add_rand_noise_th
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

def Lloyd(data, centers, num_clusters, tol, itr):
    X, d= data.shape
    new_centers=np.zeros((num_clusters,d))
    for i in range(itr):
        dist= distance.cdist(data, np.array(centers))
        cid = np.argmin(dist, axis=1)
        dist = np.amin(dist, axis = 1)


        for j in range(num_clusters):
            if len(data[cid==j])>0:
                new_centers[j]= np.mean(data[cid==j], axis=0)

        plt.figure()
        plt.scatter(data[:,0], data[:,1], c=cid)
        plt.scatter(centers[:,0], centers[:,1], marker='o', color= 'red')
        plt.scatter(new_centers[:,0], new_centers[:,1], marker='o', color= 'black')


        old_centers= centers
        centers=new_centers

        isOPTIMAL= True

        if (len(np.setdiff1d(new_centers,old_centers)))>tol:
            isOPTIMAL= False


        if isOPTIMAL:
            break


    return new_centers, cid



def L_cost(data, cid, centers):
    dist= distance.cdist(data, np.array(centers))
    dist = np.amin(dist, axis = 1)
    cost= np.zeros(len(centers))
    for i in range(len(centers)):
        cluster_indx = np.where(cid==i)
        cluster_points = data_new[cluster_indx]
        cost[i] = np.sum((cluster_points-centers[i])**2)

    total_cost= np.sum(cost)/len(data_new)
    return total_cost
