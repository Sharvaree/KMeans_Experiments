import numpy as np
import math
import random

def phi(P, C):
    phi = 0
    for j in range(len(P)):
        minim = float("inf")
        for k in range(len(C)):
            d2 = np.linalg.norm(P[j]-C[k])
            if(d2 < minim):
                minim = d2
        phi += minim

    return phi

def main():
    
    print("Preparing...")
    print("Reading and parsing data...")
    
    #Constants
    z = 2
    phistar = 100
    k = 4
    delta = 0.1 #Desired probability of failure
    times = int(math.log(1/delta))

    #Read from file
    content = ''
    with open("syntheticData/data.txt") as f:
            content = f.readlines()
    content = [x.strip() for x in content]

    #Parse data to numpy
    sep = []
    data = []

    for line in content:
        line.replace(" ", "")
        sep = str.split(line, ',')
        for i in range(len(sep)):
            sep[i] = float(sep[i])
        data.append(np.array(sep))

    
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

    for j in range(times):
    
        winnerlist = []
        C = []
        
        for i in range(t):
            print("----------------\ni:", i, "of t:", t , ". Iteration",iteration, "of" ,times,  "\n----------------")
            print("  Calculating distances", i)
            #Calculate d^2 distances
            countThresh = 0
            d2snonthresh = []
            d2s = []
            sumdist = 0
            ranges = []
            for j in range(len(data)):
                minim = float("inf")
                for k in range(len(C)):
                    d2 = np.linalg.norm(data[j]-C[k])
                    if(d2 < minim):
                        minim = d2
                d2snonthresh.append(minim)
                d2s.append(min(minim, phistar/z))
                ranges.append(sumdist)
                sumdist += min(minim, phistar/z)
                if(minim > phistar/z):
                    countThresh+=1
            ranges.append(sumdist)
            print("  Done")
            
            #Case >= 2z are thresholded
            if(countThresh >= 2*z):
                print("  Case 1", i)
                print("    Picking centroid...")
                ran = random.uniform(0, sumdist)
                winner = 0
                for j in range(len(ranges) -1):
                    if(ran >= ranges[j] and ran < ranges[j+1]):
                        winner = j
                winnerlist.append(winner)
                C.append(data[winner])
                print("  Done")
            #Case < 2z are thresholded
            else:
                print("  Case 2", i)
                print("    Removing 2z max...")
                a = np.array(d2snonthresh)
                ind = np.argpartition(a, -2*z)[-2*z:]
                ranges = []
                index = []
                
                for j in range(len(d2snonthresh)):
                    index.append(j)
                
                npd2s = np.array(d2s)
                npindex = np.array(index)
                npd2smod = np.delete(npd2s, ind)
                npindexmod = np.delete(npindex, ind)

                print("    Recalculating distances...")
                
                sumdist2 = 0
                for j in range(len(npd2smod)):
                    ranges.append(sumdist2)
                    sumdist2 += npd2smod[j]
                ranges.append(sumdist2)

                print("    Picking centroid...")
                
                ran = random.uniform(0, sumdist2)
                winner = 0
                for j in range(len(ranges) -1):
                    if(ran >= ranges[j] and ran < ranges[j+1]):
                        winner = npindexmod[j]
                winnerlist.append(winner)
                C.append(data[winner])
                print("  Done")
                
        wlists.append(winnerlist)
        print(winnerlist)
        costs.append(phi(data,C))
        iteration = iteration+1

    print(costs)

    '''
    infile = open("out.txt", 'w')
    for i in range(len(n4)):
            infile.write(n4[i] + ", " + n6[i] + ", " + n8[i] + "\n")
    '''
main()
