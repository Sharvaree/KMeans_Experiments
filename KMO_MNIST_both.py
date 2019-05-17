'''
Test on MNIST data
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
from LO import add_noise,add_noise_general, add_noise_SUSY8, add_noise_SUSY10, add_noise_SUSY18, KPP_centers, random_centers, LO_cost, LloydOut, KMO_centers
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
print("Size of A:{}".format(a.shape))
print("PCA done. Considering first {} principal components".format(n_components))
init= a[random.randint(1, len(a)-1)]
data=a[0:1000,:]
print("Using whole of the data")

num_clusters=[10, 20]
zs =[25, 50, 100]
tol= .05
itr=1000
min_value= 10
max_value=20

for num_cluster in num_clusters:
    for z in zs:
        print("num_cluster:{}, z:{}".format(num_cluster, z))
        data_with_outliers, z_indx, data_inliers = add_noise_general(data, z, min_value, max_value)
        kpp_centers = KPP_centers(data_with_outliers, num_cluster)
        centers, cid, indx_list, precision, recall=LloydOut(data_with_outliers, kpp_centers, num_cluster, z, tol, itr, z_indx )

        rand_centers= random_centers(data_with_outliers, num_cluster)
        centers, cid, indx_list, precision, recall= LloydOut(data_with_outliers, rand_centers, num_cluster, z, tol, itr, z_indx)

        phi_star= compute_phi_star(data, num_cluster, kpp_centers, z)
        kmo_centers= KMO_centers(data_with_outliers, num_cluster, phi_star, z)
        centers, cid, indx_list, precision, recall= LloydOut(data_with_outliers, kmo_centers, num_cluster, z, tol, itr, z_indx)






#centers= KPP_centers(data_with_outliers, num_clusters)
#print("KPP init")
#phi_star= compute_phi_star(data_inliers,num_clusters, centers, z)
#print("Phi_star: {}".format(phi_star))
#centers, cid, dist= kmeansOutliers(data_with_outliers, phi_star, z, num_clusters)
#costs, z_alg = cost(data_with_outliers, cid, centers, z)
#precision = len(np.intersect1d(z_indx, z_alg))/len(z_indx)
#recall = len(np.intersect1d(z_indx, z_alg))/len(z_alg)
#print(("Precision:{}, recall:{}". format(precision, recall)))
#print("Random init")
#data_with_outliers, z_indx, data_inliers = add_noise(data, z, min_value, max_value)
#centers= random_centers(data_with_outliers, num_clusters)
#phi_star= compute_phi_star(data_inliers,num_clusters, centers, z)
#rint("Phi_star: {}".format(phi_star))
#centers, cid, dist= kmeansOutliers(data_with_outliers, phi_star, z, num_clusters)
#costs, z_alg = cost(data_with_outliers, cid, centers, z)
#precision = len(np.intersect1d(z_indx, z_alg))/len(z_indx)
#recall = len(np.intersect1d(z_indx, z_alg))/len(z_alg)
#print(("Precision:{}, recall:{}". format(precision, recall)))
