#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np

class Lloyd:
    def __init__(self, num_clusters, tol, max_itr):
        self.num_clusters=num_clusters
        self.tol=tol
        self.max_itr= max_itr
        
    def fit(self, data):
        self.centers= {}
        
        for i in range(self.num_clusters):
            self.centers[i]=data[i]
            
        for i in range(self.max_itr):
            self.clusters={}
            for i in range(self.num_clusters):
                self.clusters[i]=[]
                
            for points in data:
                distances= [np.linalg.norm(points- self.centers[centers]) for centers in self.centers]
                min_distances = distances.index(min(distances))
                self.clusters[min_distances].append(points)
            old_centers=dict(self.centers)
            
        
            for min_distances in self.clusters:
                self.centers[min_distances]= np.average(self.clusters[min_distances])
            
            isOPTIMAL= True
        
            for centers in self.centers:
                original_centers=old_centers[centers]
                current_centers= self.centers[centers]
                #print(current_centers)
                if np.sum((current_centers-original_centers)/original_centers*100)> self.tol:
                    isOPTIMAL= False
            
            if isOPTIMAL:
                break
            
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centers[centers]) for centers in self.centers]
        min_distances=distances.index(min(distances))
        return min_distances
    
 
# Testing on dummy data
from sklearn import cluster, datasets, mixture
n_samples=100
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)

LLD=Lloyd(num_clusters=3, tol=.1, max_itr=10)
LLD.fit(noisy_circles[0])    
LLD.clusters        
    

