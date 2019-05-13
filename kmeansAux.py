import numpy as np
from scipy.spatial import distance
import math
import random
	
	
def kMPrecRecall(sd, wins):
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
	print(prec, recall)

	return prec, recall
			
