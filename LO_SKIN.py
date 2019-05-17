'''
Tests on SKIN dataset
'''


from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
import pandas as pd
import random
import scipy.io as scio
from numpy import genfromtxt
from KmeansPPcenters import KMeanPlusPlus
from KMeansOut import kmeansOutliers, cost,  compute_phi_star
from sklearn.metrics import average_precision_score, precision_recall_curve
from LO import add_noise, KPP_centers, random_centers, LO_cost, LloydOut, KMO_centers



read_data = genfromtxt('realDataProcessed/skin.csv', delimiter=',')
print("data loaded")
skin_labels= read_data[:,3]
skin_data=read_data[:,0:3]

num_clusters=[10, 20]
zs =[25, 50, 100]
min_value= 210
max_value=255
#min_values=[-16, 0, 7.693475e-08]
#max_values= [15, 20, 33]
tol= .05
itr=1000

data= skin_data
#data_with_outliers=skin_data


for num_cluster in num_clusters:
    for z in zs:
        print("num_cluster:{}, z:{}".format(num_cluster, z))
        data_with_outliers, z_indx, data_inliers = add_noise(data, z, min_value, max_value)
        kpp_centers = KPP_centers(data_with_outliers, num_cluster)
        centers, cid, indx_list, precision, recall=LloydOut(data_with_outliers, kpp_centers, num_cluster, z, tol, itr, z_indx )

        rand_centers= random_centers(data_with_outliers, num_cluster)
        centers, cid, indx_list, precision, recall= LloydOut(data_with_outliers, rand_centers, num_cluster, z, tol, itr, z_indx)

        phi_star= compute_phi_star(data, num_cluster, kpp_centers, z)
        kmo_centers= KMO_centers(data_with_outliers, num_cluster, 10000*phi_star, z)
        centers, cid, indx_list, precision, recall= LloydOut(data_with_outliers, kmo_centers, num_cluster, z, tol, itr, z_indx)
