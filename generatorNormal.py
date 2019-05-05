import numpy as np
import math
import random

#Parameters
#n: if integer, then n is numeber of points, and each cluster will have n/k. 
#	if list, n is a list of size k with number of points in each cluster.
#		example: n = [1000] * k will yield k clusters of 1000 points each
#				 n = 5000 will yield k clusters of 5000/k points each
#d: number of dimensions
#k: number of clusters
#rang: range of points, each dimension will range from -rang to rang (centered at origin)
#sds: if number, sds is a numeer with the standard deviation of the distribution
#	  if list, sds is a list of size d with the standard deviations on each dimension
#		example: sds = [1] * d will define all standard deviations to be 1
#		example: sds = 1 will have the same effect
#z: number of outliers
def generatorNorm(n,d,k,rang,sds,z,num):
    #Constants
    if(isinstance(sds,int) or isinstance(sds,float)):
        sds = [sds]*d
    if(isinstance(n,list)):    
        s = n
        n = sum(n)
    else:
        s = [int(n/k)]*k
    zrang = rang
    
    #Generate centers
    centers = []
    #sds = []
    for i in range(k):
        center = []
        #sd = []
        for j in range(d):
            center.append(random.uniform(-rang, rang))
            #sd.append(random.uniform(0, csd))
        centers.append(center)
        #sds.append(sd)

    #Generate outliers
    for i in range(z):
        center = []
        #sd = []
        for j in range(d):
            center.append(random.uniform(-zrang, zrang))
            #sd.append(random.uniform(0, csd))
        centers.append(center)
        #sds.append(sd)

    #Draw samples from normal
    pointset = centers.copy()

    for i in range(k):
        for j in range(s[i]):
            point = []
            for l in range(d):
                point.append(np.random.normal(centers[i][l],sds[l]))
            pointset.append(point)
    
    print(len(pointset))

    fn = "syntheticData/n" + str(n) + "d" + str(d) + "k" + str(k) + "rang" + str(int(rang)) + "z" + str(z) +"c"+ str(num) + ".csv"

    infile = open(fn , 'w')
    for i in range(len(pointset)):
        for j in range(len(pointset[0])):
            if(j == d - 1):
                infile.write(str(pointset[i][j]))
            else:
                infile.write(str(pointset[i][j]) + ",")
        infile.write("\n")
    return fn
