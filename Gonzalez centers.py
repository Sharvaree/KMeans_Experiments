#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np # linear algebra
from scipy.spatial import distance
import math
import csv

def max_dist(data, centers):
    distances = np.zeros(len(data)) #cumulative distance measure for all points
    for cluster_id, center in enumerate(centers):
        for point_id, point in enumerate(data):
            if distance.euclidean(point,center) == 0.0:
                distances[point_id] = -math.inf # already in cluster
            if not math.isinf(distances[point_id]):
                # add the distance 
                distances[point_id] = distances[point_id] + distance.euclidean(point,center) 
                # return the point which is furthest away 
    return data[np.argmax(distances)]

def Gonzalez(data, num_clusters, init):
    centers = []
    centers.append(init) # initialize the first center
    while len(centers) is not num_clusters:
        centers.append(max_dist(data, centers)) 
    return np.array(centers)

#Gonzalez_centers = gonzalez(data, num_clusters, init)
#print('Cluster Centeroids:', Gonzalez_centers)

