#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.spatial import distance
import math
import random
import time

import kMeanspp as kmpp


def update_dist(dist, tempdist, cid, data, ix):
	for indx in range(len(data)):
		if dist[indx] <= tempdist[indx]:
			continue
		else:
			dist[indx] = tempdist[indx]
			cid[indx] = ix
	return dist, cid

def kmeansOutliers(data, phi_star, z, num_clusters):
	# Random initialization
	wins = [random.randint(0, len(data)-1)]
	centers = np.array([data[wins[0]]]) 
	
	# set size of points as max numbers for dist array
	dist = np.array([1.7e308]*len(data))
    
	#index
	cid =  np.zeros(len(data))
    
	for i in range(num_clusters):
        
		#Calculate distance to new center
		tempdist = distance.cdist(data, np.array([centers[len(centers)-1]]))
		tempdist = np.array([item[0] for item in tempdist])
        
		#keep only the distance min(dist, tempdist)
		dist, cid = update_dist(dist, tempdist, cid, data, i)
        
		#thresholded value
		th = (phi_star/z)*np.ones(len(data))
        
		#Distribution post thresholding
		distribution = np.minimum(dist, th)
		tempSum = sum(distribution)
        
        
		#Normalizing
		distribution = distribution/tempSum
        
		#Picking new center with the above distribution
		new_index = np.random.choice(len(data), 1, p=distribution)
		wins.append(new_index[0])
        
		#Adding new center
		centers = np.append(centers, data[new_index], axis = 0)
	centers=centers[:-1]

	return centers, cid, dist, np.array(wins)


def cost(data, cid, centers, z):

	dist= distance.cdist(data, np.array(centers))
	dist = np.amin(dist, axis = 1)
	indx_list = np.argpartition(dist, -z)[-z:] #get index of farthest z points
    
	cid_pruned = cid.copy()
	cid_pruned[indx_list] = len(centers) + 1 # comment this line out if you do not want to remove points

	cost= np.zeros(len(centers))
	for i in range(len(centers)):
		cluster_indx = np.where(cid_pruned==i)
		cluster_points = data[cluster_indx]
		cost[i] = np.mean((cluster_points-centers[i])**2)
        
	return cost, indx_list

def cost2(data, centers,z):
    dist= distance.cdist(data, np.array(centers))
    dist = np.amin(dist, axis = 1)
    s = np.sort(dist)
    s = s[:len(dist) - z- 1]
    return np.sum(s)

def cost_per_pt(data, centers):
	dist= distance.cdist(data, np.array(centers))
	dist = np.amin(dist, axis = 1)
	return dist

def cost_per_pt_w_centers_eliminated(data, centers):
	costs=[]
	for i in range(len(centers)):
		centerswithoutkth = np.delete(np.array(centers), [i], axis=0)
		costs.append(cost_per_pt(data, centerswithoutkth))
	return costs

def ls(u, c, k, eps):
	alpha = float("inf")
	
	curcost = cost2(u, c,0)
	count=0
	while(alpha*(1-(eps/k)) > curcost):
		print("			LS:it",count)
		alpha = curcost
		
		improvedC = c.copy()
		bestCost = cost2(u,improvedC,0) #current best cost
		cpps = cost_per_pt_w_centers_eliminated(u, improvedC) #cost per point with each center eliminated(k x n)

		for i in range(len(u)):
			tempu = u[i]
			cpppp = (cost_per_pt(u, [tempu]))#cost per point of picking only tempu as center(n)
			
			for j in range(len(improvedC)):
				#amin finds the elementwise minimum over [cost per point with jth center eliminated] and [cost of picking point]
				#sum adds these points
				newCost = sum(np.amin([cpps[j], cpppp], axis=0))

				if(newCost< bestCost):
					print("			BestCost", bestCost, "newCost", newCost)
					bestCost = newCost
					improvedC[j] = tempu
					cpps = cost_per_pt_w_centers_eliminated(u, improvedC)

		c = improvedC
		curcost = cost2(u, c,0)
		count+=1
	
	return c

def outliers(data, centers,z):
    dist= distance.cdist(data, np.array(centers))
    dist = np.amin(dist, axis = 1)
    s = np.argsort(dist)
    return s[len(dist) - z:]

def randomInit(u, k):
	centers = []

	for i in range(k):
		centers.append(u[random.randint(0, len(u)-1)])

	return np.array(centers)

def zero_out_outliers(uvcosts, z):
	ind = np.argsort(uvcosts)
	badInd = ind[len(ind) - z:]
	for i in badInd:
		uvcosts[i] = 0
	return uvcosts, badInd

def zero_out_ind(uvcosts, ind):
	for i in ind:
		uvcosts[i] = 0
	return uvcosts

def lsOut(u,k,z, eps):
	c = kmpp.kmeanspp(u,k)
	#c = randomInit(u,k)
	zsInd = outliers(u, c, z)
	uNoOut = np.delete(u,zsInd, axis=0)
	alpha = float("inf")

	curcost = cost2(uNoOut, c,0)
	count = 0
	while(alpha*(1-(eps/k)) > curcost):
		print("		LSOut:it",count)
		alpha = curcost

		#(i)
		uNoOut = np.delete(u,zsInd, axis=0)
		c = ls(uNoOut, c, k,eps)
		curcost = cost2(uNoOut, c,0)

		improvedC = c.copy()
		improvedZ = zsInd.copy()

		#(ii)
		zsInd2 = outliers(uNoOut, c, z)
		uNoOut2 = np.delete(uNoOut,zsInd2, axis=0)
		if(curcost*(1-(eps/k)) > cost2(uNoOut2, c,0)):
			improvedZ = np.append(improvedZ, zsInd2)

		#(iii)
		cpps = cost_per_pt_w_centers_eliminated(u, improvedC) #cost per point with each center eliminated(k x n)
		uNoOut3 = np.delete(u,improvedZ, axis=0)
		bestCost = cost2(uNoOut3,improvedC,0)
		
		for i in range(len(u)):
			tempu = u[i]
			cpppp = (cost_per_pt(u, [tempu]))#cost per point of picking only tempu as center(n)
			for j in range(len(improvedC)):
				uvcosts = np.amin([cpps[j], cpppp], axis=0)
				uvcosts, new_out_ind = zero_out_outliers(uvcosts,z)
				uvcosts = zero_out_ind(uvcosts, zsInd)

				newCost = sum(uvcosts)
				
				if(newCost < bestCost):
					print("		BestCost", bestCost, "newCost", newCost)
					improvedC[j] = tempu
					improvedZ = list(set(zsInd).union(set(new_out_ind)))
					bestCost = newCost
					cpps = cost_per_pt_w_centers_eliminated(u, improvedC) 
			

		#Last step
		if(curcost*(1-(eps/k)) > bestCost):
			c = improvedC
			zsInd = improvedZ
			uNoOut = np.delete(u,zsInd, axis=0)
			curcost = cost2(uNoOut, c,0)
		count+=1

	return c, len(zsInd)

########################################################
#Improved LS: With coreset implementation
########################################################

def rowMult(m, v):
	for i in range(len(m)):
		m[i] = m[i]*v[i]
	return m

def cost2im(data, centers,z,w):
	dist= distance.cdist(data, np.array(centers))
	dist = rowMult(dist,w)
	dist = np.amin(dist, axis = 1)
	s = np.sort(dist)
	s = s[:len(dist) - z- 1]
	return np.sum(s)

def cost_per_ptim(data, centers,w):
	dist= distance.cdist(data, np.array(centers))
	dist = rowMult(dist,w)
	dist = np.amin(dist, axis = 1)
	return dist

def cost_per_pt_w_centers_eliminatedim(data, centers,w):
	costs=[]
	for i in range(len(centers)):
		centerswithoutkth = np.delete(np.array(centers), [i], axis=0)
		costs.append(cost_per_ptim(data, centerswithoutkth,w))
	return costs

def outliersim(data, centers,z,w):
	dist= distance.cdist(data, np.array(centers))
	dist = rowMult(dist,w)
	dist = np.amin(dist, axis = 1)
	s = np.argsort(dist)
	return s[len(dist) - z:]

def kmeanspp(data,num_clusters):
    
    # Random initialization
    wins = [random.randint(0, len(data)-1)]
    centers = np.array([data[wins[0]]]) 
    
    # set size of points as max numbers for dist array
    dist = np.array([1.7e308]*len(data))
    
    #index
    cid =  np.zeros(len(data))
    
    for i in range(num_clusters-1):
        #Calculate distance to new center
        tempdist = distance.cdist(data, np.array([centers[len(centers)-1]]))
        tempdist = np.array([item[0] for item in tempdist])
        
        #keep only the distance min(dist, tempdist)
        dist, cid = update_dist(dist, tempdist, cid, data, i)
        
        #Normalizing
        tempSum = sum(dist)
        distribution = dist/tempSum
        
        #Picking new center with the above distribution
        new_index = np.random.choice(len(data), 1, p=distribution)
        wins.append(new_index[0])
        
        #Adding new center
        centers = np.append(centers, data[new_index], axis = 0)

    dist= distance.cdist(data, np.array(centers))
    dist = np.argmin(dist, axis = 1)
	
    weights = [0.0]*len(centers)

    for d in dist:
        weights[d] += 1

    wsum = sum(weights)
    weights = np.array(weights)

    return centers, weights

def lsImproved(u,w, c, k, eps):
	alpha = float("inf")
	
	curcost = cost2im(u, c,0,w)
	count=0
	while(alpha*(1-(eps/k)) > curcost):
		print("			LS:it",count)
		alpha = curcost
		
		improvedC = c.copy()
		bestCost = cost2im(u,improvedC,0,w) #current best cost
		cpps = cost_per_pt_w_centers_eliminatedim(u, improvedC,w) #cost per point with each center eliminated(k x n)

		for i in range(len(u)):
			tempu = u[i]
			cpppp = (cost_per_ptim(u, [tempu],w))#cost per point of picking only tempu as center(n)
			
			for j in range(len(improvedC)):
				#amin finds the elementwise minimum over [cost per point with jth center eliminated] and [cost of picking point]
				#sum adds these points
				uvcosts = np.amin([cpps[j], cpppp], axis=0)
				#uvcosts = rowMult(uvcosts, w)
				newCost = sum(uvcosts)

				if(newCost< bestCost):
					print("			BestCost", bestCost, "newCost", newCost)
					bestCost = newCost
					improvedC[j] = tempu
					cpps = cost_per_pt_w_centers_eliminatedim(u, improvedC,w)

		c = improvedC
		curcost = cost2im(u, c,0,w)
		count+=1
	
	return c

def lsOutImproved(u,k,z, eps, coresetSize):
	u, w = kmeanspp(u,coresetSize)
	c = kmpp.kmeanspp(u,k)
	#c = randomInit(u,k)
	zsInd = outliersim(u, c, z,w)
	uNoOut = np.delete(u,zsInd, axis=0)
	alpha = float("inf")

	curcost = cost2im(uNoOut, c,0,w)
	count = 0
	while(alpha*(1-(eps/k)) > curcost):
		print("		LSOut:it",count)
		alpha = curcost

		#(i)
		uNoOut = np.delete(u,zsInd, axis=0)
		c = lsImproved(uNoOut, w, c, k,eps)
		curcost = cost2im(uNoOut, c,0,w)

		improvedC = c.copy()
		improvedZ = zsInd.copy()

		#(ii)
		zsInd2 = outliersim(uNoOut, c, z,w)
		uNoOut2 = np.delete(uNoOut,zsInd2, axis=0)
		if(curcost*(1-(eps/k)) > cost2im(uNoOut2, c,0,w)):
			improvedZ = np.append(improvedZ, zsInd2)

		
		#(iii)
		cpps = cost_per_pt_w_centers_eliminatedim(u, improvedC,w) #cost per point with each center eliminated(k x n)
		uNoOut3 = np.delete(u,improvedZ, axis=0)
		bestCost = cost2im(uNoOut3,improvedC,0,w)
		
		for i in range(len(u)):
			tempu = u[i]
			cpppp = (cost_per_ptim(u, [tempu],w))#cost per point of picking only tempu as center(n)
			for j in range(len(improvedC)):
				uvcosts = np.amin([cpps[j], cpppp], axis=0)
				uvcosts, new_out_ind = zero_out_outliers(uvcosts,z)
				uvcosts = zero_out_ind(uvcosts, zsInd)
				#uvcosts = rowMult(uvcosts, w)

				newCost = sum(uvcosts)
				
				if(newCost < bestCost):
					print("		BestCost", bestCost, "newCost", newCost)
					improvedC[j] = tempu
					improvedZ = list(set(zsInd).union(set(new_out_ind)))
					bestCost = newCost
					cpps = cost_per_pt_w_centers_eliminatedim(u, improvedC,w)
			

		#Last step
		if(curcost*(1-(eps/k)) > bestCost):
			c = improvedC
			zsInd = improvedZ
			uNoOut = np.delete(u,zsInd, axis=0)
			curcost = cost2im(uNoOut, c,0,w)
		count+=1

	return c, len(zsInd)

	


