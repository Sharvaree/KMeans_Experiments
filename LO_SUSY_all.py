#!/usr/bin/env python
# coding: utf-8

# In[11]:


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
susy_data = pd.read_csv('realData/SUSY/SUSY.csv', header=None)
labels= susy_data.values[:,0:1]
processed_data= np.delete(susy_data.values,0, axis=1)
data_all= processed_data
data_part_8= processed_data[:,0:8]
data_part_10= processed_data[:,0:10]
data= data_part_8
num_clusters=10
z =25
min_value= -16
max_value=33
#min_values=[-16, 0, 7.693475e-08]
#max_values= [15, 20, 33]
tol= .05
itr=5
data_with_outliers, z_indx, data_inliers = add_noise(data, z, min_value, max_value)
centers= KPP_centers(data_with_outliers, num_clusters)
centers, index=LloydOut(data_with_outliers, centers, num_clusters, z, tol, itr, z_indx )
data_with_outliers, z_indx, data_inliers = add_noise(data, z, min_value, max_value)
centers= random_centers(data_with_outliers, num_clusters)
centers, index= LloydOut(data_with_outliers, centers, num_clusters, z, tol, itr, z_indx)


# In[ ]:




