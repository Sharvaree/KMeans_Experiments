import numpy as np
import math
import random
from scipy.spatial import cKDTree
import sys


#Debug global
DEBUG = False

def find_k_closest(centroids, data, k=1, distance_norm=2):
    kdtree = cKDTree(data, leafsize=100)
    distances, indices = kdtree.query(centroids, k, p=distance_norm)
    if k > 1:
        indices = indices[:,-1]
    #values = data[indices]
    return distances, indices

def phi(P, C):
    distances, indices = find_k_closest(P, C)

    return sum(distances)

def ourAlg():
    print("Preparing...")
    print("Reading and parsing data...")

    #Parameters
    fileName = "syntheticData/data.txt"
    
    #Constants
    z = 2
    phistar = 100
    k = 4
    delta = 0.1 #Desired probability of failure
    times = int(math.log(1/delta))

    #Getting csv
    data = np.genfromtxt(fileName, delimiter=',')

    #Initializations
    t = int(23*(k + math.sqrt(k)))
    C = []

    print("Ready...")
    print("Running...")

    #Algorithm 1
    rn = random.randint(0, len(data)-1)
    C.append(data[rn])

    wlists = []
    winnerlist = []
    costs = []
    iteration = 0

    minCost = float("inf")

    bestCentroids = []

    for j in range(times):
    
        winnerlist = []
        rn = random.randint(0, len(data)-1)
        C = [data[rn]]
        
        for i in range(t):
            if(DEBUG):
                print("----------------------------------------------------\nRunning i:", i+1, "of t:", t , ". Iteration",iteration+1, "of" ,times,  "\n----------------------------------------------------")
                print("  Calculating distances", i+1)

            #Calculate d^2 distances
            d2snonthresh, indices = find_k_closest(data, C)
            d2s = np.where(d2snonthresh > (phistar/z), (phistar/z), d2snonthresh)
            countThresh = len(d2snonthresh[np.where(d2snonthresh > phistar/z)])
            sumdist = sum(d2snonthresh)
            if(DEBUG):
                print("  Done")

            #Case >= 2z are thresholded
            if(countThresh < 2*z):
                if(DEBUG):
                    print("  Case 1", i)
                    print("    Picking centroid...")
                normd2s = d2s/sum(d2s)
                winner =  np.random.choice(len(normd2s), 1, p=normd2s)
                C.append(data[winner][0])
                winnerlist.append(winner[0])
                if(DEBUG):
                    print("  Done")
            #Case < 2z are thresholded
            else:
                if(DEBUG):
                    print("  Case 2", i)
                    print("    Removing 2z max...")
                a = np.array(d2snonthresh)
                ind = np.argpartition(a, -2*z)[-2*z:]
                index = []
                
                for j in range(len(d2snonthresh)):
                    index.append(j)
                
                
                datamod = np.delete(data, (ind), axis=0)
                indexmod = np.setdiff1d(index, ind)

                if(DEBUG):
                    print("    Recalculating distances...")

                d2snonthresh, indices = find_k_closest(datamod, C)
                d2s = np.where(d2snonthresh > (phistar/z), (phistar/z), d2snonthresh)
                countThresh = len(d2snonthresh[np.where(d2snonthresh > phistar/z)])
                sumdist = sum(d2snonthresh)
                
                if(DEBUG):
                    print("  Done")
                    print("    Picking centroid...")
                
                normd2s = d2s/sum(d2s)
                winner =  indexmod[np.random.choice(len(normd2s), 1, p=normd2s)[0]]
                C.append(data[winner])
                winnerlist.append(winner)
                
                if(DEBUG):
                    print("  Done")


        wlists.append(winnerlist)
        print(winnerlist)
        currCost = phi(data,C)
        costs.append(currCost)
        if(currCost < minCost):
            minCost = currCost
            bestCentroids = winnerlist
        iteration = iteration+1
        

    print(costs)

ourAlg()
