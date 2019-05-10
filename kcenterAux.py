import numpy as np
from scipy.spatial import distance
import math
import random

def kCCost(data, centers, r):
	#Calculates minimum distances
	dist = distance.cdist(data, np.array(centers))
	dist = np.amin(dist, axis = 1)
	#print(dist[0:100])
	#Counting how many > 2r
	r2 = 2*r

	#count = (dist > r2).sum()
	count = np.count_nonzero(dist > r2)
	print(count)
	return count
	
	
def kCPrecRecall(sd, wins):
	hit = np.zeros(sd.k)
	tp = 0
	fp = 0
	fn = 0
	centers = sd.data[:sd.k]
	r2 = 2*sd.s
		
	dists = distance.cdist(wins, centers)
	
	for i in range(len(dists)):
		ard = dists[i]
		h = False
		for j in range(len(ard)):
			if(ard[j] <= r2):
				if(hit[j] == 0):
					hit[j] = 1
					tp += 1
					h = True
		if(not h):
			fp += 1
	
	fn = sd.k - np.sum(hit)

	prec = tp/(tp+fp)
	recall = tp/(tp+fn)

	print(prec, recall)

	return prec, recall
			
