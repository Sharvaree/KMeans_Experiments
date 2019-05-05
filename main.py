import numpy as np
import math
import random
import sys
import os

#Custom imports
import kcentersOutliers as kco
import generatorNormal as gn

#Class that contains info and 
#data of a single synthetic file
class synthD:
	filename = ""
	n = 0
	d = 0
	k = 0
	rang = 0.0
	z = 0
	c = 0
	data = 0
	
#Reads csv file to numpy array.
def get_csv(fileName):
        return np.genfromtxt(fileName, delimiter=',')

#This function creates the synthetic data described by
#Gupta, Kumar and Lu in their paper "Local Search Methods 
#for k-Means with Outliers" 
def createSynthDataGKL():
	files = []
	ks = [10, 20]
	zs = [25,50,100]
	ds = [2,15]
	for k in ks:
		for z in zs:
			for d in ds:
				for num in range(10):
					files.append(gn.generatorNorm(10000-k-z,d ,k,50.0,1,z,num))
	print("----------------\nFiles generated\n----------------")
	print("Generated files from of synthetic data \n as done by Gupta, Kumar and Lu.\n Listed below")
	print(files)

#Gets the names of all synthetic data files
def getAllSynthNames():
	fns = os.listdir("syntheticData/")
	for i in range(len(fns)):
		fns[i] = "syntheticData/" + fns[i]
	return fns

#Separate strings by alphabetical letters
def splitalpha(s):
	pos = 0
	last = 0
	l = []
	while pos < len(s):
		while pos < len(s) and not s[pos].isalpha():
			pos+=1
		l.append(s[last:pos])
		pos+=1
		last = pos
	return l

#Reads in a synthetic data file along with attributes
def readSynthetic(fileName):
	sd = synthD()
	parse = splitalpha(fileName.lstrip("syntheticData/"))
	sd.filename = fileName
	sd.n = int(parse[0])
	sd.d = int(parse[1])
	sd.k =int(parse[2])
	sd.rang = int(parse[6])
	sd.z =int(parse[7])
	sd.c =int(parse[8].rstrip('.'))
	sd.data = get_csv(fileName)
	return sd


############################################################################################
#Main tasks
############################################################################################
def main():
	#Creates synthetic data as described in paper 10 copies each
	#createSynthDataGKL()

	#Gets the names of the synthetic data files
	synthData = getAllSynthNames()

	#reads data and parses first file in folder
	sd = readSynthetic(synthData[0])

	#Running kcenterOut on the data
	kcent = kco.kcentersOut(sd.data,sd.k,2)
	ans = kcent.kcentersOut()
	print(ans)
############################################################################################

#Sample functions
def sampleDataGen():
	k = 5
	d = 15
	s = [1000] * k 
	sds = [1] * d
	filename = gn.generatorNorm(5000,15 ,5,50.0,1,1000,0)
	print(filename)

def sampleReadData():
	dataSynth = get_csv(synthData[0])
	kcent = kco.kcentersOut(dataSynth,10,10.0)
	ans = kcent.kcentersOut()
	print(ans)
  
main()
