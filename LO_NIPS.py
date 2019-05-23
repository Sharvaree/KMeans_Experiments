'''
Tests on NIPS dataset
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

import realAux as real
import localSearch as ls


#######
from sklearn.decomposition import TruncatedSVD

rng = np.random.RandomState(8)

path = 'realDataProcessed/NIPS.csv'
data = np.array(pd.read_csv(path, header=0, index_col=0))

d = 50
svd = TruncatedSVD(n_components=d, n_iter=20, random_state=rng)
data = svd.fit_transform(data)
#######
'''
nips_data = pd.read_csv('realDataProcessed/NIPS.csv', header=None)
print("Data loaded")
header= nips_data.values[0:1:,]
preprocessed_data= np.delete(nips_data.values,0, axis=0)
labels= nips_data.values[:,0:1]
processed_data= np.delete(preprocessed_data,0, axis=1)
print(processed_data.shape)
n_components=50
pca = PCA(n_components=n_components)
pca.fit(processed_data.T)
a= pca.components_.T
print("PCA done. Considering first {} principal components".format(n_components))
data=a
'''
print(data.shape)   

factor = 2
min_value= np.min(np.amin(data, axis=1), axis=0)*factor
max_value= np.max(np.amax(data, axis=1), axis=0)*factor

print(min_value, max_value, factor)

num_clusters=[10, 20, 30]
betas= [.125,.25, .5, .75, 1, 2, 5]

#Approx 5% of te dataset
z=int(500/2)
tol= .05
#Number of Lloyds iterations
itr=100

#Number of experiments
iterations=5

#number of runs for a fixed dataset
runs=5

#LS iterations
lsit = 1

#####
statsK = []
#####
for num_cluster in num_clusters:
	#####
	statsKPP = []
	statsLS = []
	statsKMO = []
	for j in range(len(betas)):
		statsKMO.append([])
	#####
	print("num_cluster:{}".format(num_cluster))
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
		#####
		KMOruns = [[],[],[],[]]
		KMO_LO_prec_runs = KMOruns[0]
		KMO_LO_rec_runs = KMOruns[1]
		KMO_LO_cost_runs= KMOruns[2]
		KMO_LO_itr_runs= KMOruns[3]
		for j in range(len(betas)):
			KMO_LO_prec_runs.append([])
			KMO_LO_rec_runs.append([])
			KMO_LO_cost_runs.append([])
			KMO_LO_itr_runs.append([])
		
		KPPruns = [[],[],[],[]]
		KPP_LO_prec_runs = KPPruns[0]
		KPP_LO_rec_runs = KPPruns[1]
		KPP_LO_cost_runs= KPPruns[2]
		KPP_LO_itr_runs= KPPruns[3]

		LSruns = [[],[],[],[]]
		LS_LO_prec_runs = LSruns[0]
		LS_LO_rec_runs = LSruns[1]
		LS_LO_cost_runs= LSruns[2]
		LS_LO_itr_runs= LSruns[3]
		#####

		data_with_outliers, z_indx, data_inliers = add_noise_general(data, z, min_value, max_value)
		for j in range(runs):
			print("-", end='')
			#----------
			#kpp
			#----------
			kpp_centers = KPP_centers(data_with_outliers, num_cluster)
			#print("KPP")
			centers, cid, indx_list, KPP_precision, recall, data_out, KPP_itr =LloydOut(data_with_outliers, kpp_centers, num_cluster, z, tol, itr, z_indx )
			KPP_cost= LO_cost2(data_with_outliers, centers, 0)
			KPP_LO_prec_runs.append(KPP_precision)
			KPP_LO_rec_runs.append(KPP_precision)
			KPP_LO_cost_runs.append(KPP_cost)
			KPP_LO_itr_runs.append(KPP_itr)
			'''
			KPP_LO_prec_runs.append(0)
			KPP_LO_rec_runs.append(0)
			KPP_LO_cost_runs.append(0)
			KPP_LO_itr_runs.append(0)
			'''
			#----------
			#kmo
			#----------
			for k in range(len(betas)):
				
				beta = betas[i]
				phi_star= compute_phi_star(data, num_cluster, kpp_centers, z)
				kmo_centers= KMO_centers(data_with_outliers, num_cluster, beta*phi_star, z)
				#print("PHI-star:{}".format(beta*phi_star))
				centers, cid, indx_list, KMO_precision, recall, data_out, KMO_itr= LloydOut(data_with_outliers, kmo_centers, num_cluster, z, tol, itr, z_indx)
				KMO_cost= LO_cost2(data_with_outliers, centers, 0)
				KMO_LO_prec_runs[k].append(KMO_precision)
				KMO_LO_rec_runs[k].append(KMO_precision)
				KMO_LO_cost_runs[k].append(KMO_cost)
				KMO_LO_itr_runs[k].append(KMO_itr)
				'''
				KMO_LO_prec_runs.append(0)
				KMO_LO_rec_runs.append(0)
				KMO_LO_cost_runs.append(0)
				KMO_LO_itr_runs.append(0)
				'''

			#----------
			#ls
			#----------
			if(j < lsit):
				
				LS_centers, empz = ls.lsOutCor(data_with_outliers, num_cluster,z , 0.1, int(1.5*(z+num_cluster)), debug = False)
				#print("KPP")
				centers, cid, indx_list, LS_precision, LS_recall, data_out, LS_itr =LloydOut(data_with_outliers, LS_centers, num_cluster, empz, tol, itr, z_indx )
				LS_cost= LO_cost2(data_with_outliers, centers, 0)
				LS_LO_prec_runs.append(LS_recall)
				LS_LO_rec_runs.append(LS_precision)
				LS_LO_cost_runs.append(LS_cost)
				LS_LO_itr_runs.append(LS_itr)
				'''
				LS_LO_prec_runs.append(0)
				LS_LO_rec_runs.append(0)
				LS_LO_cost_runs.append(0)
				LS_LO_itr_runs.append(0)
				'''

		print("PHI-star:{}".format(beta*phi_star))
		print("runs:{},KPP: prec:{}, rec:{}, cost:{}, itr: {}".format(j+1, np.mean(np.array(KPP_LO_prec_runs)),np.mean(np.array(KPP_LO_rec_runs)), np.mean(np.array(KPP_LO_cost_runs)), np.mean(np.array(KPP_LO_itr_runs))))
		#print("Random:{}, cost:{}, itr:{}".format(np.mean(np.array(R_LO_prec)),np.mean(np.array(R_LO_cost))))
		for k in range(len(KMO_LO_cost_runs)):
			print("runs:{}, KMO: prec:{}, rec:{},cost:{}, itr: {}, beta: {}".format(j+1, np.mean(np.array(KMO_LO_prec_runs[k])), np.mean(np.array(KMO_LO_rec_runs[k])),np.mean(np.array(KMO_LO_cost_runs[k])), np.mean(np.array(KMO_LO_itr_runs[k])), betas[k]))
		print("runs:{},LS: prec:{}, rec:{}, cost:{}, itr: {}".format(j+1, np.mean(np.array(LS_LO_prec_runs)), np.mean(np.array(LS_LO_rec_runs)), np.mean(np.array(LS_LO_cost_runs)), np.mean(np.array(LS_LO_itr_runs))))

		#####
		LSruns = real.avOverRows(LSruns)
		KPPruns = real.avOverRows(KPPruns)

		for k in range(len(KMOruns)):
			KMOruns[k] = real.avOverRows(KMOruns[k])

		KMOruns = np.transpose(KMOruns)
		
		statsKPP.append(KPPruns)
		statsLS.append(LSruns)
		for k in range(len(betas)):
			statsKMO[k].append(list(KMOruns[k]))
		#####	
		
		KPP_LO_prec.append(np.mean(np.array(KPP_LO_prec_runs)))
		KPP_LO_cost.append(np.mean(np.array(KPP_LO_cost_runs)))
		KPP_LO_itr.append(np.mean(np.array(KPP_LO_itr_runs)))

		KMO_LO_prec.append(np.mean(np.array(KMO_LO_prec_runs)))
		KMO_LO_cost.append(np.mean(np.array(KMO_LO_cost_runs)))
		KMO_LO_itr.append(np.mean(np.array(KMO_LO_itr_runs)))

	#####
	statsKPP = np.mean(statsKPP, axis = 0)
	statsLS = np.mean(statsLS, axis = 0)
	for k in range(len(betas)):
		statsKMO[k] = np.mean(statsKMO[k], axis = 0)
	statsK.append([statsKPP, statsLS, statsKMO])
	#####

print(statsK)

real.writeRealStats("real_NIPS", statsK, num_clusters, betas)
		
