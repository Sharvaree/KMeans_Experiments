#Code for Kmeans++
#default num of clusters is 3

import numpy as np
from scipy.spatial import distance

class KMeanPlusPlus:
    def __init__(self, num_clusters=3, init=None):
        self.num_clusters = num_clusters
        self.centers = np.atleast_2d(init) #in case the file is (n, ) format
        
    def fit(self, X):
        for i in range(1, self.num_clusters):
            # find the distance to the closest center for each of the points
            corresponding_centers = self.predict(X)
            distance_matrix = distance.cdist(X, self.centers)
            distances = distance_matrix[np.arange(len(X)), corresponding_centers]
            dist_prob = distances / sum(distances)
            
            # pick a new point based on the dist_prob
            new_center_index = np.random.choice(np.arange(len(X)), p=dist_prob)
            new_center = X[new_center_index]

            # add farthest point to centers
            self.centers = np.vstack([self.centers, new_center])
            
        return self
    
    def predict(self, X):
        # compute distance matrix 
        distances = distance.cdist(X, self.centers)
        
        # find the closest center
        closest_centers = np.argmin(distances, axis=1)
        
        return closest_centers
        
    def cost(self, X, mean=False):
        # compute closest centers
        closest_centers = self.predict(X)
        
        # find distance to all centers
        distances = cdist(X, self.centers)
        
        # Retain distance only to the point and its assigned center
        distances = distances[np.arange(len(distances)), closest_centers]
        
        if mean == True:
            cost = np.sqrt(np.sum(distances ** 2) / len(X))
        else:
            cost = distances.max()
        
        return cost    