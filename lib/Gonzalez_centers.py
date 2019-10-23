'''
    File name:      Gonzalez_centers.py
    Description:    Classic Kcenters algorithm which picks the furthest point
                    as the center
    Author:         Sharvaree V
    Last modified:  16th May 2019
    Python Version: 3.5
'''

import numpy as np
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
    '''
    data            Data as numpy array
    num_clusters    Number of clusters (k)
    init            First center to initialize the algorithm
    '''


    centers = []
    centers.append(init) # initialize the first center
    while len(centers) is not num_clusters:
        centers.append(max_dist(data, centers))
    return np.array(centers)


#dummy data_set
from sklearn import cluster, datasets, mixture
n_samples=100
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
data= noisy_circles[0]
Gonzalez_centers = Gonzalez(data, num_clusters=3, init=data[0])
#print('Cluster Centers:', Gonzalez_centers)
