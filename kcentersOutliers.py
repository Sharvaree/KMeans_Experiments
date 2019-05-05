import numpy as np
from scipy.spatial import distance
import math
import random

class kcentersOut:
    def get_csv(self,fileName):
        return np.genfromtxt(fileName, delimiter=',')

    def __init__(self,data, it, r2):
        self.it = it
        self.data = data
        self.r2 = r2
        
    def kcentersOut(self):
        size = len(self.data)

        #Random initialization
        centers = np.array([self.data[random.randint(0, size-1)]])

        #Populating dist array
        dist = distance.cdist(self.data, np.array([centers[len(centers)-1]]))
        dist = np.array([item for sublist in dist for item in sublist])
        
        for i in range(self.it):
            print("-----------------------------\n", centers.shape)
            #Get distance to new center
            tempdist = distance.cdist(self.data, np.array([centers[len(centers)-1]]))
            tempdist = np.array([item for sublist in tempdist for item in sublist])
            #For each entry, if leq, replace
            dist = np.array([dist[i] if dist[i] <= tempdist[i] else tempdist[i] for i in range(size)])
            #Computinf distribution
            distribution = np.where(dist > self.r2, 1, [0]*size)
            tempSum = sum(distribution)
            distribution = distribution/float(tempSum)
            #Picking center
            winnerInd = np.random.choice(size,1, p = distribution)
            #Adding center
            centers = np.append(centers, self.data[winnerInd], axis = 0)
            
        return np.array(centers)



#kcent = kcentersOutliers("syntheticData/data.txt",10,10.0)
#kcent.kcentersOutliers()

