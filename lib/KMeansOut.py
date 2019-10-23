'''
    File name:      KMeansOut.py
    Description:    Kmeans algorithm as described in the paper in Section 4
    Author:         Sharvaree V
    Last modified:  16th May 2019
    Python Version: 3.5
'''


import numpy as np
from scipy.spatial import distance
import math
import random
#import numba
import time

def get_csv(fileName):
    return np.genfromtxt(fileName, delimiter=',')

#@numba.njit() #to run functions without Python interpretor
def update_dist(dist, tempdist, cid, data, ix):
    for indx in range(len(data)):
        if dist[indx] <= tempdist[indx]:
            continue
        else:
            dist[indx] = tempdist[indx]
            cid[indx] = ix
    return dist, cid

def compute_phi_star(data, num_clusters, centers, z):
    dist_matrix= distance.cdist(data, centers)
    dist = np.amin(dist_matrix, axis = 1)
    phi_star=np.sum(dist)
    return phi_star

def squareMat(ar):
	for i in range(len(ar)):
		ar[i] = ar[i]*ar[i]
	return ar

def kmeansOutliers(data, phi_star, z, num_clusters):
    '''
        data            Data as numpy array
        phi_star        Threshold value
        num_clusters    Number of clusters (k)
        z               Number of outliers
        centers         Cluster centers post convergence as numpy array
        cid             Closest center index for each data point as numpy array
        indx_list       Index of calculated Outliers(Z') as numpy array
    '''

    #np.random.seed(1)

    # Random initialization
    rand_indx= random.randint(0, len(data)-1)
    centers = np.array([data[rand_indx]])

    # set size of points as max numbers for dist array
    dist = np.array([1.7e308]*len(data))

    #index
    cid =  np.zeros(len(data))

    for i in range(num_clusters):

        #Calculate distance to new center
        tempdist = distance.cdist(data, np.array([centers[len(centers)-1]]))
        tempdist = np.array([item[0] for item in tempdist])

        #keep only the distance min(dist, tempdist)
        dist, cid = update_dist(dist, tempdist, cid, data, i)
        dist = squareMat(dist)

        #thresholded value
        th = (phi_star/z)*np.ones(len(data))

        #Distribution post thresholding
        distribution = np.minimum(dist, th)
        tempSum = sum(distribution)

        new_index = [random.randint(0, len(data)-1)]
        #Normalizing
        if(not tempSum == 0):
            distribution = distribution/tempSum

            #Picking new center with the above distribution
            new_index = np.random.choice(len(data), 1, p=distribution)


        #Adding new center
        centers = np.append(centers, data[new_index], axis = 0)
    centers=centers[:-1]

    return centers, cid, dist


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
        cost[i] = np.sum((cluster_points-centers[i])**2)
    total_cost= np.sum(cost)/len(data)

    return total_cost, indx_list
