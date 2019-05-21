'''
    File name:      LO.py
    Description:    Classic Lloyd's algorithm which ignores z furthest cluster
                    points at every iteration, with Kmeans++ and Random
                    initialization (same as LO.py, except that this returns Lloyd convergence iteration number)
    Author:         Sharvaree V
    Last modified:  16th May 2019
    Python Version: 3.5
'''

import numpy as np
from scipy.spatial import distance
import math
import random
#import numba  #To make it run faster
import time
import matplotlib.pyplot as plt

# functions from other .py files.
#Add the path of the files if they are not in the same directory

from Noise import add_random_noise, add_rand_noise_th, add_rand_noise_SUSY18, add_rand_noise_general
from KMeansOut import kmeansOutliers, cost, compute_phi_star
from KmeansPPcenters import KMeanPlusPlus



def get_csv(fileName):
    return np.genfromtxt(fileName, delimiter=',')

#Random initialization
def random_centers(data, num_clusters):
    centers= data[np.random.choice(np.arange(len(data)-1), num_clusters)]
    return centers

#Kmeans++ initialization
def  KPP_centers(data, num_clusters):
    random_indx= random.randint(0, len(data)-1)
    init=data[random_indx]
    KPP=KMeanPlusPlus(num_clusters=num_clusters, init=init)
    KPP.fit(data)
    KPP_centers= KPP.centers
    return np.array(KPP_centers)

def KMO_centers(data, num_clusters, phi_star, z):
    ko_centers, cid, dist = kmeansOutliers(data, phi_star, z, num_clusters)
    return np.array(ko_centers)



# Add noise
def add_noise(data, z, min_value, max_value):
    data_copy=data.copy()
    data_with_outliers, z_indx = add_random_noise(data_copy, z, min_value, max_value)
    data_inliers= np.delete(data_copy, z_indx, axis=0)
    return data_with_outliers, z_indx, data_inliers

def add_noise_general(data, z, min_value, max_value):
    data_copy=data.copy()
    data_with_outliers, z_indx = add_rand_noise_general(data_copy, z, min_value, max_value)
    data_inliers= np.delete(data, z_indx, axis=0)
    return data_with_outliers, z_indx, data_inliers


def add_noise_SUSY18(data, z, min_value, max_value):
    data_copy=data.copy()
    data_with_outliers, z_indx = add_rand_noise_SUSY18(data_copy, z, min_value, max_value)
    data_inliers= np.delete(data, z_indx, axis=0)
    return data_with_outliers, z_indx, data_inliers

#Lloyd's algorithm with outliers

def LloydOut(data, centers, num_clusters,z, tol, itr, z_indx):
    '''
    data            Data as numpy array
    centers         Cluster centers as numpy array
    num_clusters    Number of clusters (k)
    z               Number of outliers
    tol             Tolerance
    itr             Number of iterations for converging
    z_indx          Index of true Outliers as numpy array
    new_centers     Cluster centers post convergence as numpy array
    cid             Closest center index for each data point as numpy array
    indx_list       Index of calculated Outliers(Z') as numpy array
    precision       (Z intersection Z_prime)/Z
    recall          (Z_prime intersection Z)/Z_prime
    '''
    data_copy= data.copy()
    dist= distance.cdist(data_copy, centers)
    dist = np.amin(dist, axis = 1)
    X, d= data.shape
    new_centers=np.zeros((num_clusters,d))
    for i in range(itr):
        #print(i)
        dist= distance.cdist(data_copy, np.array(centers))
        cid = np.argmin(dist, axis=1)
        dist = np.amin(dist, axis = 1)

        indx_list = np.argpartition(dist, -z)[-z:]
        data_new = np.delete(data_copy, indx_list, axis=0)

        dist_new = distance.cdist(data_new, centers)
        cid = np.argmin(dist_new, axis=1)
        dist_new = np.amin(dist_new, axis = 1)

        for j in range(num_clusters):
            if len(data_new[cid==j])>0:
                new_centers[j]= np.mean(data_new[cid==j], axis=0)

        #plt.figure()
        #plt.scatter(data_new[:,0], data_new[:,1], c=cid)
        #plt.scatter(centers[:,0], centers[:,1], marker='o', color= 'red')
        #plt.scatter(new_centers[:,0], new_centers[:,1], marker='o', color= 'black')


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
    #print(("Precision:{}, recall:{}, itr:{}". format(precision, recall, i)))
    return new_centers, cid, indx_list, precision, recall, data_copy, i


def LO_cost(data, cid, centers, z):
    data_copy= data.copy()
    dist= distance.cdist(data_copy, np.array(centers))
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

def LO_cost2(data, centers, z):
	dist=distance.cdist(data, np.array(centers))
	dist=np.amin(dist, axis=1)
	s=np.sort(dist)
	s=s[:len(dist)-z-1]
	return np.sum(s)

def LO_cost3(data, centers, z):
    dist=distance.cdist(data, np.array(centers))
    dist=np.amin(dist, axis=1)
    indx_list = np.argpartition(dist, -z)[-z:]
    s=np.sort(dist)
    s=s[:len(dist)-z-1]
    return np.sum(s), indx_list
