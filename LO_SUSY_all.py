'''
Tests on SUSY dataset
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
from LO import add_noise, add_noise_SUSY8,add_noise_SUSY10, add_noise_SUSY18, KPP_centers, random_centers, LO_cost, KMO_centers, LO_cost2
from LO_with_itr import LloydOut



susy_data = pd.read_csv('realDataProcessed/SUSY.csv', header=None)
labels= susy_data.values[:,0:1]
processed_data= np.delete(susy_data.values,0, axis=1)
data_all= processed_data
data_part_8= processed_data[:,0:8]
data_part_10= processed_data[:,0:10]
data= processed_data
print(data.shape)

num_clusters=[10]
zs =[25]
min_value= 40
max_value=48
#min_values=[-16, 0, 7.693475e-08]
#max_values= [15, 20, 33]
tol= .05
itr=100
iterations=10

for num_cluster in num_clusters:
    for z in zs:
        print("num_cluster:{}, z:{}".format(num_cluster, z))
        KMO_LO_prec = []
        KPP_LO_prec = []
        R_LO_prec = []
        KPP_LO_cost=[]
        KMO_LO_cost=[]
        R_LO_cost=[]
        KPP_LO_itr=[]
        KMO_LO_itr=[]
        R_LO_itr=[]
        for i in range(iterations):
            data_with_outliers, z_indx, data_inliers = add_noise_SUSY18(data, z, min_value, max_value)
            kpp_centers = KPP_centers(data_with_outliers, num_cluster)
            centers, cid, indx_list, KPP_precision, recall, data_out, KPP_itr  =LloydOut(data_with_outliers, kpp_centers, num_cluster, z, tol, itr, z_indx )
            KPP_cost= LO_cost2(data_with_outliers, centers, z)
            KPP_LO_prec.append(KPP_precision)
            KPP_LO_cost.append(KPP_cost)
            KPP_LO_itr.append(KPP_itr)

            #rand_centers= random_centers(data_with_outliers, num_cluster)
            #centers, cid, indx_list, R_precision, recall, data_out= LloydOut(data_with_outliers, rand_centers, num_cluster, z, tol, itr, z_indx)
            #R_cost= LO_cost2(data_with_outliers, centers, z)
            #R_LO_prec.append(R_precision)
            #R_LO_cost.append(R_cost)

            phi_star= compute_phi_star(data, num_cluster, kpp_centers, z)
            kmo_centers= KMO_centers(data_with_outliers, num_cluster, 1.5*phi_star, z)
            centers, cid, indx_list, KMO_precision, recall, data_out, KMO_itr= LloydOut(data_with_outliers, kmo_centers, num_cluster, z, tol, itr, z_indx)
            KMO_cost= LO_cost2(data_with_outliers, centers, z)
            KMO_LO_prec.append(KMO_precision)
            KMO_LO_cost.append(KMO_cost)
            KMO_LO_itr.append(KMO_itr)
        print("KPP:{}, cost:{}, itr: {}".format(np.mean(np.array(KPP_LO_prec)), np.mean(np.array(KPP_LO_cost)), np.mean(np.array(KPP_LO_itr))))
        #print("Random:{}, cost:{}, itr:{}".format(np.mean(np.array(R_LO_prec)),np.mean(np.array(R_LO_cost))))
        print("KMO:{},cost:{}, itr:{}".format(np.mean(np.array(KMO_LO_prec)), np.mean(np.array(KMO_LO_cost)), np.mean(np.array(KMO_LO_itr))))
