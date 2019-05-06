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
	
	
