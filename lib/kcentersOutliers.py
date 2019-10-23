import numpy as np
from scipy.spatial import distance
import math
import random

DEBUG = False
UNIFORM = True

class kcentersOut:
    def get_csv(self,fileName):
        return np.genfromtxt(fileName, delimiter=',')

    def __init__(self,data, it, r):
        self.it = it
        self.data = data
        self.r2 = r * 2
        
    def kcentersOut(self):
        size = len(self.data)

        #Random initialization
        winners = [[random.randint(0, size-1)]]
        centers = np.array([self.data[winners[0][0]]])

        #Populating dist array
        dist = distance.cdist(self.data, np.array([centers[len(centers)-1]]))
        dist = np.array([item for sublist in dist for item in sublist])
        if(DEBUG):
        	print(dist[0:100])
        
        
        for i in range(self.it-1):
            if(DEBUG):
                print("-----------------------------\n", centers.shape)
            #Get distance to new center
            if(DEBUG):
                print(dist[0:100])
            tempdist = distance.cdist(self.data, np.array([centers[len(centers)-1]]))
            tempdist = np.array([item for sublist in tempdist for item in sublist])
            if(DEBUG):
                print(tempdist[0:100])
            #For each entry, if leq, replace
            dist = np.array([dist[i] if dist[i] <= tempdist[i] else tempdist[i] for i in range(size)])
            if(DEBUG):
                print(dist[0:100])
            #Computinf distribution
            if(UNIFORM):
                distribution = np.where(dist > self.r2, 1, [0]*size)
            else:
                distribution = np.where(dist > self.r2, dist, [0]*size)
            if(DEBUG):
                print(dist[0:100])
            tempSum = sum(distribution)
            distribution = distribution/float(tempSum)
            if(DEBUG):
                print(distribution[0:100])
            #Picking center
            winnerInd = [0]
            if(not np.isnan(distribution).any()):
                winnerInd = np.random.choice(size,1, p = distribution)
            else:
                if(tempSum >= 0):			
                    pick = random.randint(0,tempSum)
                    distribution = np.where(dist > self.r2, 1, [0]*size)
                    ind = 0
                    while(pick > 0):
                        if(distribution[ind] == 1):
                            pick -= 1
                        ind += 1
                    winnerInd == [ind]
            winners.append(winnerInd)
            if(DEBUG):
                print(winnerInd, "\n", self.data[winnerInd])
            #Adding center
            centers = np.append(centers, self.data[winnerInd], axis = 0)

        if(DEBUG):
            print(winners)
            
        return np.array(centers)



#kcent = kcentersOutliers("syntheticData/data.txt",10,10.0)
#kcent.kcentersOutliers()

