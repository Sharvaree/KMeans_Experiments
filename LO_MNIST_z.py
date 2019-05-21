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
from LO import add_noise, KPP_centers, random_centers, LO_cost, KMO_centers, LO_cost2, add_noise_general
from LO_with_itr import LloydOut
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
a= pca.components_.T
print("PCA done. Considering first {} principal components".format(n_components))
init= a[random.randint(1, len(a)-1)]
data=a
print(data.shape)
num_clusters= [10]
zs=[6000]
min_value= [-0.022]
max_value=[.023]
#min_values=[-16, 0, 7.693475e-08]
#max_values= [15, 20, 33]
tol= .05
itr=100

#data= skin_data
#data_with_outliers=skin_data
iterations=5
runs=10
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
            KMO_LO_prec_runs = []
            KPP_LO_prec_runs = []
            R_LO_prec_runs = []
            KPP_LO_cost_runs=[]
            KMO_LO_cost_runs=[]
            R_LO_cost_runs=[]
            KPP_LO_itr_runs=[]
            KMO_LO_itr_runs=[]
            R_LO_itr_runs=[]
            data_with_outliers, z_indx, data_inliers = add_noise_general(data, z, min_value, max_value)
            for j in range(runs):
                kpp_centers = KPP_centers(data_with_outliers, num_cluster)
                #print("KPP")
                centers, cid, indx_list, KPP_precision, recall, data_out, KPP_itr =LloydOut(data_with_outliers, kpp_centers, num_cluster, z, tol, itr, z_indx )
                KPP_cost= LO_cost2(data_with_outliers, centers, z)
                KPP_LO_prec_runs.append(KPP_precision)
                KPP_LO_cost_runs.append(KPP_cost)
                KPP_LO_itr_runs.append(KPP_itr)

                #rand_centers= random_centers(data_with_outliers, num_cluster)
                #centers, cid, indx_list, R_precision, recall, data_out= LloydOut(data_with_outliers, rand_centers, num_cluster, z, tol, itr, z_indx)
                #R_cost= LO_cost2(data_with_outliers, centers, z)
                #R_LO_prec.append(R_precision)
                #3R_LO_cost.append(R_cost)
                #KPP_LO_cost.append(KPP_cost)
                #print("KMO")
                phi_star= compute_phi_star(data, num_cluster, kpp_centers, z)
                kmo_centers= KMO_centers(data_with_outliers, num_cluster, .5*phi_star, z)
		#print("PHI-star:{}".format(phi_star))
                centers, cid, indx_list, KMO_precision, recall, data_out, KMO_itr= LloydOut(data_with_outliers, kmo_centers, num_cluster, z, tol, itr, z_indx)
                KMO_cost= LO_cost2(data_with_outliers, centers, z)
                KMO_LO_prec_runs.append(KMO_precision)
                KMO_LO_cost_runs.append(KMO_cost)
                KMO_LO_itr_runs.append(KMO_itr)
            print("runs:{},KPP:{}, cost:{}, itr: {}".format(j, np.mean(np.array(KPP_LO_prec_runs)), np.mean(np.array(KPP_LO_cost_runs)), np.mean(np.array(KPP_LO_itr_runs))))
            #print("Random:{}, cost:{}, itr:{}".format(np.mean(np.array(R_LO_prec)),np.mean(np.array(R_LO_cost))))
            print("runs:{}, KMO:{},cost:{}, itr:{}".format(j, np.mean(np.array(KMO_LO_prec_runs)), np.mean(np.array(KMO_LO_cost_runs)), np.mean(np.array(KMO_LO_itr_runs))))
            KPP_LO_prec.append(np.mean(np.array(KPP_LO_prec_runs)))
            KPP_LO_cost.append(np.mean(np.array(KPP_LO_cost_runs)))
            KPP_LO_itr.append(np.mean(np.array(KPP_LO_itr_runs)))
            
            KMO_LO_prec.append(np.mean(np.array(KMO_LO_prec_runs)))
            KMO_LO_cost.append(np.mean(np.array(KMO_LO_cost_runs)))
            KMO_LO_itr.append(np.mean(np.array(KMO_LO_itr_runs)))
        print("KPP:{}, cost:{}, itr: {}".format( np.mean(np.array(KPP_LO_prec)), np.mean(np.array(KPP_LO_cost)), np.mean(np.array(KPP_LO_itr))))
        #print("Random:{}, cost:{}, itr:{}".format(np.mean(np.array(R_LO_prec)),np.mean(np.array(R_LO_cost))))
        print(" KMO:{},cost:{}, itr:{}".format( np.mean(np.array(KMO_LO_prec)), np.mean(np.array(KMO_LO_cost)), np.mean(np.array(KMO_LO_itr))))

