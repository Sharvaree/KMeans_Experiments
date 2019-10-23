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
from LO import add_noise, KPP_centers, random_centers, LO_cost, KMO_centers, LO_cost3, LO_cost2
from LO_with_itr import LloydOut



read_data = genfromtxt('realDataProcessed/skin.csv', delimiter=',')
print("data loaded")
skin_labels= read_data[:,3]
skin_data=read_data[:,0:3]

num_clusters=[20]
zs =[100]
min_value=0
max_value=100000
#min_values=[-16, 0, 7.693475e-08]
#max_values= [15, 20, 33]
tol= .05
#itr=100

data= skin_data
#data_with_outliers=skin_data
iterations=10
runs=5
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
            data_with_outliers, z_indx, data_inliers = add_noise(data, z, min_value, max_value)
            for j in range(runs):
                kpp_centers = KPP_centers(data_with_outliers, num_cluster)
                KPP_cost, indx_list= LO_cost3(data_with_outliers, kpp_centers, z)
                KPP_precision = len(np.intersect1d(z_indx, indx_list))/len(z_indx)
                #print("KPP")
                #centers, cid, indx_list, KPP_precision, recall, data_out, KPP_itr =LloydOut(data_with_outliers, kpp_centers, num_cluster, z, tol, itr, z_indx )
                KPP_LO_prec_runs.append(KPP_precision)
                KPP_LO_cost_runs.append(KPP_cost)
                #KPP_LO_itr_runs.append(KPP_itr)

                #rand_centers= random_centers(data_with_outliers, num_cluster)
                #centers, cid, indx_list, R_precision, recall, data_out= LloydOut(data_with_outliers, rand_centers, num_cluster, z, tol, itr, z_indx)
                #R_cost= LO_cost2(data_with_outliers, centers, z)
                #R_LO_prec.append(R_precision)
                #3R_LO_cost.append(R_cost)
                #KPP_LO_cost.append(KPP_cost)
                #print("KMO")
                phi_star= compute_phi_star(data, num_cluster, kpp_centers, z)
                kmo_centers= KMO_centers(data_with_outliers, num_cluster, phi_star, z)
                #KMO_cost, indx_list= LO_cost2(data_with_outliers, kmo_centers, z)
                #centers, cid, indx_list, KMO_precision, recall, data_out, KMO_itr= LloydOut(data_with_outliers, kmo_centers, num_cluster, z, tol, itr, z_indx)
                KMO_cost, indx_list= LO_cost3(data_with_outliers, kmo_centers, z)
                KMO_precision = len(np.intersect1d(z_indx, indx_list))/len(z_indx)
                KMO_LO_prec_runs.append(KMO_precision)
                KMO_LO_cost_runs.append(KMO_cost)
                #KMO_LO_itr_runs.append(KMO_itr)
            print("runs:{},KPP:{}, cost:{}, itr: {}".format(j, np.mean(np.array(KPP_LO_prec_runs)), np.mean(np.array(KPP_LO_cost_runs)), np.mean(np.array(KPP_LO_itr_runs))))
            #print("Random:{}, cost:{}, itr:{}".format(np.mean(np.array(R_LO_prec)),np.mean(np.array(R_LO_cost))))
            print("runs:{}, KMO:{},cost:{}, itr:{}".format(j, np.mean(np.array(KMO_LO_prec_runs)), np.mean(np.array(KMO_LO_cost_runs)), np.mean(np.array(KMO_LO_itr_runs))))
            KPP_LO_prec.append(np.mean(np.array(KPP_LO_prec_runs)))
            KPP_LO_cost.append(np.mean(np.array(KPP_LO_cost_runs)))
            #KPP_LO_itr.append(np.mean(np.array(KPP_LO_itr_runs)))
            
            KMO_LO_prec.append(np.mean(np.array(KMO_LO_prec_runs)))
            KMO_LO_cost.append(np.mean(np.array(KMO_LO_cost_runs)))
            #KMO_LO_itr.append(np.mean(np.array(KMO_LO_itr_runs)))
        print("KPP:{}, cost:{}, itr: {}".format( np.mean(np.array(KPP_LO_prec)), np.mean(np.array(KPP_LO_cost)), np.mean(np.array(KPP_LO_itr))))
        #print("Random:{}, cost:{}, itr:{}".format(np.mean(np.array(R_LO_prec)),np.mean(np.array(R_LO_cost))))
        print(" KMO:{},cost:{}, itr:{}".format( np.mean(np.array(KMO_LO_prec)), np.mean(np.array(KMO_LO_cost)), np.mean(np.array(KMO_LO_itr))))


