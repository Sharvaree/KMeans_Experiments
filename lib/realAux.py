import numpy as np
from scipy.spatial import distance
import math
import random

def avOverRows(a):
	for i in range(len(a)):
		a[i] = np.mean(a[i])
	return a

def writeRealStats(filename, stats, ks, rps):

	output = [["k", "Algorithm", "Beta", "Precision", "Recall", "Cost", "Iterations"]]

	for i in range(len(ks)):
		k = ks[i]
		substats = stats[i]

		kpp = [k, "km++", 1]
		kpp.extend(substats[0])
		output.append(kpp)

		ls = [k, "ls", 1]
		ls.extend(substats[1])
		output.append(ls)

		tks = substats[2]	
		for j in range(len(rps)):
			rp = rps[j]
			
			temp = [k, "Tkm++", rp]
			temp.extend(tks[j])

			output.append(temp)

	for i in range(len(output)):
		for j in range(len(output[0])):
			output[i][j] = str(output[i][j])

	np.savetxt("outputs/" + filename + ".csv", np.array(output), fmt = '%s', delimiter = ",")
	
			
			
 
	
