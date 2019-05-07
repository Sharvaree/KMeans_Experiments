import numpy as np
import math
import random
import sys
import os
import statistics

#Custom imports
import kcentersOutliers as kco
import generatorNormal as gn
import kcenterAux as kc
import generatorNormalCenters as gnc

#Constants
extraInfo = ["example1", "example2", "example3"] # add header names to this list, e.g. ["cluster1cost", "cluster2cost"]. make sure values are numers, since they will be averaged over runs.

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
	s = 0
	data = 0
	centers = []
	costs = []
	extrastats = [] #Please make sure len(extrastats) == len(extraInfo) if possible
	
#Reads csv file to numpy array.
def get_csv(fileName):
        return np.genfromtxt(fileName, delimiter=',')

#This function creates the synthetic data described by
#Gupta, Kumar and Lu in their paper "Local Search Methods 
#for k-Means with Outliers" 
def createSynthDataGKL():
	files = []
	sds = [1, 5, 10]
	ks = [10, 20]
	zs = [25,50,100]
	ds = [2,15]
	for sd in sds:
		for k in ks:
			for z in zs:
				for d in ds:
					for num in range(10):
						files.append(gn.generatorNorm(10000-k-z,d ,k,50.0,sd,z,num))
	print("----------------\nFiles generated\n----------------")
	print("Generated files from of synthetic data \n as done by Gupta, Kumar and Lu.\n Listed below")
	print(files)

#This function creates the synthetic data described by
#Gupta, Kumar and Lu in their paper "Local Search Methods 
#for k-Means with Outliers" 
def createSynthDataGKLCenters():
	files = []
	sds = [1, 5, 10]
	ks = [10, 20]
	zs = [25,50,100]
	ds = [2,15]
	for sd in sds:
		for k in ks:
			for z in zs:
				for d in ds:
					for num in range(10):
						files.append(gnc.generatorNormCenters(10000-k-z,d ,k,50.0,sd,z,num))
	print("----------------\nFiles generated\n----------------")
	print("Generated files from of synthetic data \n as done by Gupta, Kumar and Lu.\n Listed below")
	print(files)

#Gets the names of all synthetic data files
def getAllSynthNames():
	fns = os.listdir("syntheticData/")
	for i in range(len(fns)):
		fns[i] = "syntheticData/" + fns[i]
	return fns

#Gets the names of all synthetic data files for centers
def getAllSynthNamesCenters():
	fns = os.listdir("syntheticDataCenters/")
	for i in range(len(fns)):
		fns[i] = "syntheticDataCenters/" + fns[i]
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
	parse = splitalpha(fileName.lstrip("syntheticData/").lstrip("syntheticDataCenters/"))
	sd.filename = fileName
	sd.n = int(parse[0])
	sd.d = int(parse[1])
	sd.k =int(parse[2])
	sd.rang = int(parse[6])
	sd.z =int(parse[7])
	sd.c =int(parse[8].rstrip('.'))
	sd.s =int(parse[9].rstrip('.'))
	sd.data = get_csv(fileName)
	sd.costs = []
	return sd

#Prints a synthetic data structure
def printSD(sd):
	print("syntheticData: [n: " + str(sd.n) + ", d: " + str(sd.d) + ", k: " + str(sd.k) + ", rang: " + str(int(sd.rang)) + ", z: " + str(sd.z) +", num: "+ str(sd.c) + ", sigma: " + str(sd.s) + ", costs", str(sd.costs),"]")

#Adds the calculated answer of one k centers w/ outliers instance
def addAnswer(stats, sd):
	temp = [sd.n, sd.d,sd.k,sd.rang,sd.z,sd.s]
	found = False
	for i in range(len(stats)):
		matching = True
		for j in range(6):
			if(not stats[i][j] == temp[j]):
				matching = False
		if(matching):
			stats[i][6].append(sd.costs)
			stats[i][7].append(sd.extrastats)
			found = True
	if(not found):
		temp.append([sd.costs])
		temp.append([sd.extrastats])
		stats.append(temp)
	return stats
		

#Compute k centers w/ outliers
def computeKCoutliers(synthD):
	num = 0
	stats = []
	for f in synthD:
		#reads data and parses first file in folder
		sd = readSynthetic(f)
		print("Iteration:",num)
		num+=1

		for i in range(5):
			#Running kcenterOut on the data
			kcent = kco.kcentersOut(sd.data,sd.k,sd.s)
			ans = kcent.kcentersOut()

			#Computing cost
			sd.costs.append(kc.kCCost(sd.data, ans, sd.s))

		#example for adding extra stats, i.e. time. For headers, go to top
		sd.extrastats = [0,num, num*num]

		printSD(sd)
			
		stats = addAnswer(stats, sd)
	
	return stats

#Writes k centers statistics to a csv file
def writeKCOStats(stats):
	header = ["n","d","k","rang","z","sigma"]
	newStats = []
	for i in range(len(stats[0][6])):
		header.append("Run " + str(i+1) + " cost")
	header.append("Average")
	header.extend(extraInfo)
	
	newStats.append(np.array(header))
	for i in range(len(stats)):
		temp = stats[i][0:6]
		for j in range(len(stats[i][6])):
			temp.append(statistics.mean(stats[i][6][j]))
		temp.append(statistics.mean(temp[6:]))

		s = len(stats[i][7][0])
		aver = [0] * s
		for j in range(len(stats[i][7])):
			print(stats[i][7][j])
			for k in range(s):
				aver[k] += stats[i][7][j][k]
		for j in range(s):
			aver[j] = float(aver[j])/float(len(stats[i][7]))
		temp.extend(aver)

		for j in range(len(temp)):
			temp[j] = str(temp[j])
		newStats.append(np.array(temp))

	newStats = np.array(newStats)
	np.savetxt("outputs/kcenterOutStats.csv", newStats, fmt = '%s', delimiter = ",")
	

############################################################################################
#Main tasks
############################################################################################
def main():
	#Creates synthetic data as described in paper 10 copies each
	#createSynthDataGKL()
	#createSynthDataGKLCenters()
	
	#Gets the names of the synthetic data files
	#synthData = getAllSynthNames()
	synthData = getAllSynthNamesCenters()

	#Computing sds Note: can run stats = computeKCoutliers(synthData[:nums]) so it only runs the first nums files. Good for testing
	stats = computeKCoutliers(synthData)

	#Writing stats
	writeKCOStats(stats)
	
############################################################################################

#Sample functions
def runOnSynth():
	#reads data and parses first file in folder
	sd = readSynthetic(synthData[0])

	#Running kcenterOut on the data
	kcent = kco.kcentersOut(sd.data,sd.k,sd.s)
	ans = kcent.kcentersOut()

	printSD(sd)
	kc.kCCost(sd.data, ans, sd.s)

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
