#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import random
import scipy.io as scio
from numpy import genfromtxt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics import average_precision_score, precision_recall_curve


# In[4]:


from KmeansPPcenters import KMeanPlusPlus
from Gonzalez_centers import Gonzalez
from KMeansOut import kmeansOutliers, cost
from Noise import add_random_noise, compute_phi_star, cost


# In[7]:


susy_data = pd.read_csv('realData/SUSY/SUSY.csv', header=None)


# In[8]:


labels= susy_data.values[:,0:1]


# In[9]:


processed_data= np.delete(susy_data.values,0, axis=1)
data_all= processed_data
data_part_8= processed_data[:,0:8]
data_part_10= processed_data[:,0:10]


# In[10]:


data= data_part_8[0:10000, :]


# In[ ]:


num_clusters=[10, 20]
zs =[25, 50,100]
min_values=[0]
max_values= [100]

for num_cluster in num_clusters:
    for z in zs:
        for min_value in min_values:
            for max_value in max_values:
                print("Adding noise")
                data_with_outliers, z_indx = add_random_noise(data, z, max_value, min_value)
                data_inliers= np.delete(data, z_indx, axis=0)
                
                print("KPP initilization to calculate phi_star")
                init= data[np.random.choice(1, len(data)-1)]
                KPP=KMeanPlusPlus(num_clusters=num_cluster, init=init)
                KPP.fit(data_with_outliers)
                phi_star= compute_phi_star(data_inliers,num_cluster, KPP.centers, z)
                print("Phi_star: {}".format(phi_star))
                
                print("Calculating KMO")
                centers, cid, dist= kmeansOutliers(data_with_outliers, phi_star, z, num_cluster)
                costs, z_alg = cost(data_with_outliers, cid, centers, z)
                
                #print("Actual_outliers:{}, Calculated_outliers:{}". format(z_indx, z_alg))
                
                print("Calculating precision and recall")
                precision = len(z_indx)/(len(z_indx)+ len(np.setdiff1d(z_indx, z_alg)))
                recall= precision = len(z_indx)/(len(z_indx)+ len(np.setdiff1d(z_alg, z_indx)))
                
                #x1= KPP.predict(data_with_outliers)
                #x2= cid
                #precision = len(x1)/(len(x1)+ len(np.setdiff1d(x1, x2)))
                #recall= len(x1)/(len(x1)+ len(np.setdiff1d(x2, x1)))
                #print(x1)
                #print(cid)
                print(("Precision:{}, recall:{}". format(precision, recall)))
                #print("centers: {}, cid: {}, dist: {}".format(centers, cid, dist))
                #print("Next")
                
                
                
                


# In[ ]:




