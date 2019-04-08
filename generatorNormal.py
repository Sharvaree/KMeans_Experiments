import numpy as np
import math
import random

def main():

    #Constants
    d = 15
    k = 5
    s = [1000] * k
    rang = 50.0
    #csd = 10000.0
    sds = [[1] * d] * k
    z = 1000
    zrang = 100.0
    
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
                point.append(np.random.normal(centers[i][l],sds[i][l]))
            pointset.append(point)
    
    print(len(pointset))

    infile = open("syntheticData/data.txt", 'w')
    for i in range(len(pointset)):
        for j in range(len(pointset[0])):
            if(j == d - 1):
                infile.write(str(pointset[i][j]))
            else:
                infile.write(str(pointset[i][j]) + ",")
        infile.write("\n")
main()
