import numpy as np
import math
import random
import sys
import kcentersOutliers as kco
import generatorNormal as gn

#Reads csv file to numpy array.
def get_csv(fileName):
        return np.genfromtxt(fileName, delimiter=',')

#Main tasks
def main():
	dataSynth = get_csv("syntheticData/data.txt")
	kcent = kco.kcentersOut(dataSynth,10,10.0)
	ans = kcent.kcentersOut()
	print(ans)

	k = 5
	d = 15
	s = [1000] * k 
	sds = [1] * d
	filename = gn.generatorNorm(5000,15 ,5,50.0,sds,1000)
	print(filename)
  
main()
