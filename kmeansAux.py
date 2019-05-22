import numpy as np
from scipy.spatial import distance
import math
import random
	
def outliers(data, centers,z):
    dist= distance.cdist(data, np.array(centers))
    dist = np.amin(dist, axis = 1)
    s = np.argsort(dist)
    return s[len(dist) - z:]
	
def kMPrecRecall(sd, wins):
	tp = 0

	outInd = outliers(sd.data, wins, sd.z)
	
	for i in range(sd.k, sd.k+sd.z):
		for ind in outInd:
			if(ind == i):
				tp += 1
	
	prec = tp/sd.z
	recall = tp/sd.z

	print("Precrec:",prec, recall)

	return prec, recall

def kMPrecRecallVar(sd, wins, num):
	tp = 0

	outInd = outliers(sd.data, wins, num)
	print(outInd)
	
	for i in range(sd.k, sd.k+sd.z):
		for ind in outInd:
			if(ind == i):
				tp += 1
	
	prec = tp/num
	recall = tp/sd.z

	print("Precrec:",prec, recall)

	return prec, recall

def kMPrecRecallVar2(sd, wins, num):
	tp = 0

	indx_list = outliers(sd.data, wins, num)
	print(indx_list)
	z_indx = []

	for i in range(sd.k, sd.k+sd.z):
		z_indx.append(i)
	print(z_indx)
	
	prec = len(np.intersect1d(z_indx, indx_list))/len(indx_list)
	recall = len(np.intersect1d(z_indx, indx_list))/len(z_indx)

	print("Precrec:",prec, recall)

	return prec, recall
			
def kMPrecRecall2(sd, wins):
	hit = np.zeros(sd.k)
	tp = 0
	fp = 0
	fn = 0
	
	for i in range(len(wins)):
		win = int(wins[i])
		if(win >= sd.k+sd.z):
			c = int((win-sd.k-sd.z)/(sd.n/sd.k))
			if(hit[c] == 0):
				tp += 1
				hit[c] = 1
			else:
				fp += 1
		else:
			if(win < sd.k):
				if(hit[win] == 0):
					tp += 1
					hit[win] = 1
				else:
					fp += 1
			else:
				fp += 1
	
	fn = sd.k - np.sum(hit)

	prec = tp/(tp+fp)
	recall = tp/(tp+fn)
	
	#print("tp, fp, fn:", tp, fp, fn, len(wins))
	print("Precrec2:",prec, recall)

	return pre

def inds(data, wins):
	newwins = []
	for i in range(len(wins)):
		for j in range(len(data)):
			eq = True
			for k in range(len(data[0])):
				if(not data[j][k] == wins[i][k]):
					eq = False
					break
			if(eq):
				newwins.append(j)
				break
	return newwins

def closest(data, centers):
    dist= distance.cdist(data, np.array(centers))
    dist = np.amin(dist, axis = 1)
    s = np.argsort(dist)
    return s[0]

def inds2(data, wins):
	newwins = []
	for i in range(len(wins)):
		newwins.append(closest(data, [wins[i]]))
	return newwins

def cr(sd, wins):
	cs = wins.copy()
	wins = inds2(sd.data, cs)
	assert(len(wins) == len(cs))
	hit = np.zeros(sd.k)
	tp = 0
	fp = 0
	fn = 0
	
	for i in range(len(wins)):
		win = int(wins[i])
		if(win >= sd.k+sd.z):
			c = int((win-sd.k-sd.z)/(sd.n/sd.k))
			if(hit[c] == 0):
				tp += 1
				hit[c] = 1
			else:
				fp += 1
		else:
			if(win < sd.k):
				if(hit[win] == 0):
					tp += 1
					hit[win] = 1
				else:
					fp += 1
			else:
				fp += 1
	
	fn = sd.k - np.sum(hit)

	prec = tp/(tp+fp)
	recall = tp/(tp+fn)
	
	#print("tp, fp, fn:", tp, fp, fn, len(wins))
	print("cr:",recall)

	return recall


