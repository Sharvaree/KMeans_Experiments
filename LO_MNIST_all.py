#!/usr/bin/env python
# coding: utf-8

# In[1]:


from KmeansPPcenters import KMeanPlusPlus
from Gonzalez_centers import Gonzalez
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
import pandas as pd
import random
import scipy.io as scio
from numpy import genfromtxt
from KMeansOut import kmeansOutliers, cost
from sklearn.metrics import average_precision_score, precision_recall_curve
from Noise import add_random_noise, compute_phi_star, cost, add_rand_noise_th
from LO import add_noise, KPP_centers, random_centers, LO_cost, LloydOut
#from LO_figure import LloydOut
from sklearn.decomposition import PCA
import sklearn.datasets
mnist_data = genfromtxt('realDataProcessed/mnist_train.csv', delimiter=',')
print("Data loaded")
processed_data= np.delete(mnist_data,0, axis=0)
labels= processed_data[:, 0:1]
data_actual= np.delete(processed_data, 0, axis=1)
n_components=40
pca = PCA(n_components=n_components)
pca.fit(data_actual.T)
a= pca.components_
print("PCA done. Considering first {} principal components".format(n_components))
init= a[np.random.choice(1, len(a)-1)]
data=a.T
#data= data[0:8000, :]
num_clusters=10
z=25
min_value= [-0.022]
max_value=[.023]
tol=.01
itr=10
#PCA 20
#min_values=[-0.019954221635605316,0]
#max_values= [5, 9.79]
data_with_outliers, z_indx, data_inliers = add_noise(data, z, min_value, max_value)
centers= KPP_centers(data_with_outliers, num_clusters)
centers, index=LloydOut(data_with_outliers, centers, num_clusters, z, tol, itr, z_indx )
data_with_outliers, z_indx, data_inliers = add_noise(data, z, min_value, max_value)
centers= random_centers(data_with_outliers, num_clusters)
centers, index= LloydOut(data_with_outliers, centers, num_clusters, z, tol, itr, z_indx)

