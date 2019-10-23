import numpy as np
import math
import random
import sys
import os
import statistics
import matplotlib.pyplot as plt
import operator

#Custom imports
sys.path.insert(1, 'lib/')
import kcentersOutliers as kco
import generatorNormal as gn
import kcenterAux as kc
import generatorNormalCenters as gnc
import gonzalez as gon
import KMeansOut2 as kmo
import localSearch as ls

def create_dir(path):
	try:
		os.mkdir(path)
	except OSError:
		print ("Creation of the directory %s failed" % path)
	else:
		print ("Successfully created the directory %s " % path)

path1 = 'syntheticData'
create_dir(path1)
path1 = 'realData'
create_dir(path1)
path1 = 'realDataProcessed'
create_dir(path1)
path1 = 'syntheticDataCenters'
create_dir(path1)
path1 = 'Tests'
create_dir(path1)
path1 = 'visualizations/Final'
create_dir(path1)
path1 = 'outputs/NIPS'
create_dir(path1)

plt.rcParams['pdf.fonttype'] = 42

#Constants
extraInfo = ["av prec", "max prec", "av recall", "max recall", "prec sd", "recall sd"] # add header names to this list, e.g. ["cluster1cost", "cluster2cost"]. make sure values are numers, since they will be averaged over runs.

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
	runk = 0
	centers = []
	costs = []
	extrastats = [] #Please make sure len(extrastats) == len(extraInfo) if possible
	precs = []
	recs = []
	

def mean(ar):
	return sum(ar)/len(ar)

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
						files.append(gn.generatorNorm(10000,d ,k,50.0,sd,z,num))
	print("----------------\nFiles generated\n----------------")
	print("Generated files from of synthetic data \n as done by Gupta, Kumar and Lu.\n Listed below")
	print(files)

#This function creates the synthetic data described by
#Gupta, Kumar and Lu in their paper "Local Search Methods 
#for k-Means with Outliers" 
def createSynthDataGKLCenters():
	files = []
	sds = [5]
	ks = [20]
	zs = [100]
	ds = [15]
	for sd in sds:
		for k in ks:
			for z in zs:
				for d in ds:
					for num in range(10):
						files.append(gnc.generatorNormCenters(10000,d ,k,50.0,sd,z,num))
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
	print("syntheticData: [n: " + str(sd.n) + ", d: " + str(sd.d) + ", k: " + str(sd.k) + ", rang: " + str(int(sd.rang)) + ", z: " + str(sd.z) +", num: "+ str(sd.c) + ", sigma: " + str(sd.s) + ", runk", str(sd.runk) ,", costs", str(sd.costs),"]")

#Adds the calculated answer of one k centers w/ outliers instance
def addAnswer(stats, sd):
	temp = [sd.n, sd.d,sd.k,sd.rang,sd.z,sd.s,sd.runk]
	found = False
	for i in range(len(stats)):
		matching = True
		for j in range(7):
			if(not stats[i][j] == temp[j]):
				matching = False
		if(matching):
			stats[i][7].append(sd.costs)
			stats[i][8].append(sd.extrastats)
			stats[i][9].append(sd.precs)
			stats[i][10].append(sd.recs)
			found = True
	if(not found):
		temp.append([sd.costs])
		temp.append([sd.extrastats])
		temp.append([sd.precs])
		temp.append([sd.recs])
		stats.append(temp)
	return stats

#Compute k centers w/ gonzalez
def computeKC(synthD):
	num = 0
	stats = []
	for f in synthD:
		#reads data and parses first file in folder
		sd = readSynthetic(f)
		print("Iteration:",num)
		num+=1

		for j in range(int(sd.k)):
			sd.runk = int(sd.k/2) + j
			precs = []
			recs = []
			for i in range(1):
				#Running kcenterOut on the data
				random.seed(12345)
				kcent = gon.gonzalez(sd.data,sd.runk,sd.s)
				ans= kcent.gonzalez()

				#Computing cost
				sd.costs.append(kc.kCCost(sd.data, ans, sd.s))

				prec, rec = kc.kCPrecRecall(sd,ans)
				precs.append(prec)
				recs.append(rec)

			sd.precs = precs
			sd.recs = recs

			#example for adding extra stats, i.e. time. For headers, go to top
			sd.extrastats = [mean(np.array(precs)), max(precs), mean(np.array(recs)), max(recs)]

			printSD(sd)
			
			stats = addAnswer(stats, sd)
			sd.costs = []
	
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

		for j in range(int(sd.k)):
			sd.runk = int(sd.k/2)  + j
			precs = []
			recs = []
			for i in range(5):
				#Running kcenterOut on the data
				random.seed(12345)
				kcent = kco.kcentersOut(sd.data,sd.runk,sd.s)
				ans= kcent.kcentersOut()

				#Computing cost
				sd.costs.append(kc.kCCost(sd.data, ans, sd.s))

				prec, rec = kc.kCPrecRecall(sd,ans)
				precs.append(prec)
				recs.append(rec)

			sd.precs = precs
			sd.recs = recs

			#example for adding extra stats, i.e. time. For headers, go to top
			sd.extrastats = [mean(np.array(precs)), max(precs), mean(np.array(recs)), max(recs)]

			printSD(sd)
			
			stats = addAnswer(stats, sd)
			sd.costs = []
	
	return stats

#Compute k centers w/ random centers
def computeKCRandom(synthD):
	num = 0
	stats = []
	for f in synthD:
		#reads data and parses first file in folder
		sd = readSynthetic(f)
		print("Iteration:",num)
		num+=1

		for j in range(int(sd.k)):
			sd.runk = int(sd.k/2) + j
			precs = []
			recs = []
			for i in range(5):
				#Running kcenterOut on the data
				ans= ls.randomInit(sd.data, sd.runk)

				#Computing cost
				sd.costs.append(kc.kCCost(sd.data, ans, sd.s))

				prec, rec = kc.kCPrecRecall(sd,ans)
				precs.append(prec)
				recs.append(rec)

			sd.precs = precs
			sd.recs = recs

			#example for adding extra stats, i.e. time. For headers, go to top
			sd.extrastats = [mean(np.array(precs)), max(precs), mean(np.array(recs)), max(recs)]

			printSD(sd)
			
			stats = addAnswer(stats, sd)
			sd.costs = []
	
	return stats

#Writes k centers statistics to a csv file
def writeKCOStats(stats, filename):
	header = ["n","d","k","rang","z","sigma","runk"]
	newStats = []
	for i in range(len(stats[0][7])):
		header.append("Run " + str(i+1) + " cost")
	header.append("Average")
	header.extend(extraInfo)
	
	newStats.append(np.array(header))
	for i in range(len(stats)):
		temp = stats[i][0:7]
		for j in range(len(stats[i][7])):
			temp.append(mean(stats[i][7][j]))
		temp.append(mean(temp[7:]))

		s = len(stats[i][8][0])
		aver = [0] * s
		for j in range(len(stats[i][8])):
			for k in range(s):
				aver[k] += stats[i][8][j][k]
		for j in range(s):
			aver[j] = float(aver[j])/float(len(stats[i][8]))
		temp.extend(aver)

		#Sds
		sds = [0]*2
		precs = np.array(stats[i][8])[:,0]
		recs = np.array(stats[i][8])[:,2]
		
		'''
		for i in range(len(precs)):
			precs[i] = np.std(precs[i])
			recs[i] = np.std(recs[i])
		'''

		sds[0] = np.std(precs)
		sds[1] = np.std(recs)
		temp.extend(sds)

		for j in range(len(temp)):
			temp[j] = str(temp[j])
		newStats.append(np.array(temp))

	newStats = np.array(newStats)
	np.savetxt("outputs/" + filename, newStats, fmt = '%s', delimiter = ",")
	
def processStats(stats):
	newStats = []

	for i in range(len(stats)):
		temp = stats[i][0:7]
		for j in range(len(stats[i][7])):
			temp.append(mean(stats[i][7][j]))
		temp.append(mean(temp[7:]))

		s = len(stats[i][8][0])
		aver = [0] * s
		for j in range(len(stats[i][8])):
			for k in range(s):
				aver[k] += stats[i][8][j][k]
		for j in range(s):
			aver[j] = float(aver[j])/float(len(stats[i][8]))
		temp.extend(aver)

		#sds
		sds = [0.0]*2
		precs = np.array(stats[i][8])[:,0]
		recs = np.array(stats[i][8])[:,2]

		sds[0] = np.std(precs)
		sds[1] = np.std(recs)
		temp.extend(sds)

		newStats.append(np.array(temp))

	newStats = np.array(newStats)
	return newStats

def plotNewStats(stats):
	ks = stats[:,6]
	avprec = stats[:, 18]
	maxprec = stats[:, 19]
	avrec = stats[:,20]
	maxrec = stats[:,21]

	plotpts = np.array([avprec, maxprec, avrec, maxrec])
	
	#adding title
	plt.title("Stats of [n: " + str(stats[0][0]) + ", d: " + str(stats[0][1]) + ", k: " + str(stats[0][2]) + ", rang: " + str(int(float(stats[0][3]))) + ", z: " + str(stats[0][4]) + ", sigma: " + str(stats[0][6]) + "]")
	name = "n" + str(stats[0][0]) + "d" + str(stats[0][1]) + "k" + str(stats[0][2]) + "r" + str(int(float(stats[0][3]))) + "z" + str(stats[0][4]) + "s" + str(stats[0][5]) + ".pdf"

	print(plotpts)
	plt.xlabel("k")
	plt.ylabel("Precision/Recall")
	plt.scatter(ks, avprec)
	plt.plot(ks, avprec, label = "Average Precision")
	plt.scatter(ks, maxprec)
	plt.plot(ks, maxprec, label = "Max Precision")
	plt.scatter(ks, avrec)
	plt.plot(ks, avrec, label = "Average Recall")
	plt.scatter(ks, maxrec)
	plt.plot(ks, maxrec, label = "Max Recall")
	plt.legend(loc='best')
	plt.savefig("visualizations/" + name)
	plt.clf()

def plotGonOut(stats,stats2):
	ks = stats[:, 6]
	avprec = stats[:, 18]
	maxprec = stats[:, 19]
	avrec = stats[:,20]
	maxrec = stats[:,21]
	sdprec = stats[:,22]
	sdrec = stats[:,23]

	ks = stats2[:,6]
	avprec2 = stats2[:, 18]
	maxprec2 = stats2[:, 19]
	avrec2 = stats2[:,20]
	maxrec2 = stats2[:,21]
	sdprec2 = stats2[:,22]
	sdrec2 = stats2[:,23]

	plotpts = np.array([avprec, maxprec, avrec, maxrec])

	#Start with center outliers
	
	plt.figure(figsize=(6.4,9.6))
	plt.suptitle("Figure   n=" + str(int(stats[0][0])) + ", d=" + str(int(stats[0][1])) + ", k=" + str(int(stats[0][2])) + ", rang=" + str(int(float(stats[0][3]))) + ", z=" + str(int(stats[0][4])) + ", sigma=" + str(int(stats[0][5])) + ".")

	plt.subplot(211)
	plt.title("Mean Precision")

	plt.xlabel("k")
	plt.ylabel("Precision")
	ax = plt.gca()
	ax.set_ylim([0,1])
	plt.errorbar(ks, avprec, yerr=sdprec, fmt='o',capsize=2, capthick=2, color = "red")
	plt.plot(ks, avprec, linestyle = '-', label = "k-center-adaptive-sampling", color = "red")
	#plt.scatter(ks, maxprec)
	#plt.plot(ks, maxprec,linestyle = '--', label = "Max Precision Algorithm 1")
	plt.scatter(ks, avprec2, color = "blue")
	plt.plot(ks, avprec2,linestyle = '-.', label = "Gonzalez", color = "blue")
	#plt.scatter(ks, maxprec2)
	#plt.plot(ks, maxprec2,linestyle = ':', label = "Max Precision Gonzalez")
	plt.legend(loc='best')
	plt.tight_layout(rect = [0,0.03,1,0.95])

	plt.subplot(212)
	plt.title("Mean Recall")

	plt.xlabel("k")
	plt.ylabel("Recall")
	ax = plt.gca()
	ax.set_ylim([0,1])
	plt.errorbar(ks, avrec, yerr = sdrec, fmt='o',capsize=2, capthick=2, color = "red")
	plt.plot(ks, avrec,linestyle = '-', label = "k-center-adaptive-sampling", color = "red")
	#plt.scatter(ks, maxrec)
	#plt.plot(ks, maxrec,linestyle = '--', label = "Max Recall Algorithm 1")
	plt.scatter(ks, avrec2, color = "blue")
	plt.plot(ks, avrec2,linestyle = '-.', label = "Gonzalez", color = "blue")
	#plt.scatter(ks, maxrec2)
	#plt.plot(ks, maxrec2,linestyle = ':', label = "Max Recall Gonzalez")
	plt.legend(loc='best')
	plt.tight_layout(rect = [0,0.03,1,0.95])
	
	name = "kc_n" + str(stats[0][0]) + "d" + str(stats[0][1]) + "k" + str(stats[0][2]) + "r" + str(int(float(stats[0][3]))) + "z" + str(stats[0][4]) + "s" + str(stats[0][5]) + ".pdf"
	plt.savefig("visualizations/" + name)
	plt.clf()

def plotGonOutRand(stats,stats2,stats3):
	ks = stats[:, 6]
	avprec = stats[:, 18]
	maxprec = stats[:, 19]
	avrec = stats[:,20]
	maxrec = stats[:,21]
	sdprec = stats[:,22]
	sdrec = stats[:,23]

	ks = stats2[:,6]
	avprec2 = stats2[:, 18]
	maxprec2 = stats2[:, 19]
	avrec2 = stats2[:,20]
	maxrec2 = stats2[:,21]
	sdprec2 = stats2[:,22]
	sdrec2 = stats2[:,23]

	ks = stats3[:,6]
	avprec3 = stats3[:, 18]
	maxprec3 = stats3[:, 19]
	avrec3 = stats3[:,20]
	maxrec3 = stats3[:,21]
	sdprec3 = stats3[:,22]
	sdrec3 = stats3[:,23]


	plotpts = np.array([avprec, maxprec, avrec, maxrec])

	#Start with center outliers
	
	plt.figure(figsize=(6.4,4.8))
	plt.title("Mean Recall")

	plt.xlabel("k")
	plt.ylabel("Center Recall")
	ax = plt.gca()
	ax.set_ylim([0,1])
	plt.errorbar(ks, avrec, yerr = sdrec, fmt='o',capsize=2, capthick=1, color = "red")
	plt.plot(ks, avrec,linestyle = '-', label = "k-center-adaptive-sampling", color = "red")
	#plt.scatter(ks, maxrec)
	#plt.plot(ks, maxrec,linestyle = '--', label = "Max Recall Algorithm 1")
	plt.scatter(ks, avrec2, color = "blue")
	plt.plot(ks, avrec2,linestyle = '-.', label = "Gonzalez", color = "blue")
	#plt.scatter(ks, maxrec2)
	#plt.plot(ks, maxrec2,linestyle = ':', label = "Max Recall Gonzalez")
	plt.errorbar(ks, avrec3, yerr = sdrec, fmt='o',capsize=2, capthick=1, color = "g")
	plt.plot(ks, avrec3,linestyle = '-', label = "random-sampling", color = "g")
	plt.legend(loc='best')
	plt.tight_layout(rect = [0,0.03,1,0.95])
	
	name = "kc_n" + str(stats[0][0]) + "d" + str(stats[0][1]) + "k" + str(stats[0][2]) + "r" + str(int(float(stats[0][3]))) + "z" + str(stats[0][4]) + "s" + str(stats[0][5]) + ".pdf"
	plt.savefig("visualizations/" + name)
	plt.clf()

def combine(y):
	for i in range(len(y)):
		y[i] = np.mean(y[i], axis=0)
	return y

def flatten(y):
	for i in range(len(y)):
		y[i] = [item for sublist in y[i] for item in sublist]
	return y

def boxPlot(stats):
	stats = sorted(stats, key=operator.itemgetter(6))
	stats = np.array(stats)
	x = stats[:,6]
	y = flatten(stats[:,10])
	
	plt.figure(figsize=(6.4,4.8))
	plt.title("Figure   n=" + str(int(stats[0][0])) + ", d=" + str(int(stats[0][1])) + ", k=" + str(int(stats[0][2])) + ", rang=" + str(int(float(stats[0][3]))) + ", z=" + str(int(stats[0][4])) + ", sigma=" + str(int(stats[0][5])) + ".")
	plt.xlabel("k")
	plt.ylabel("Recall")
	ax = plt.gca()
	ax.set_ylim([0,1])
		
	plt.boxplot(y, labels = x)

	name = "kc_boxPlot_n" + str(stats[0][0]) + "d" + str(stats[0][1]) + "k" + str(stats[0][2]) + "r" + str(int(float(stats[0][3]))) + "z" + str(stats[0][4]) + "s" + str(stats[0][5]) + ".pdf"
	plt.savefig("visualizations/" + name)
	plt.clf()

############################################################################################
#Main tasks
############################################################################################
def main():
	#Creates synthetic data as described in paper 10 copies each
	#createSynthDataGKL()
	createSynthDataGKLCenters()
	
	#Gets the names of the synthetic data files
	#synthData = getAllSynthNames()
	synthData = getAllSynthNamesCenters()

	#Computing sds Note: can run stats = computeKCoutliers(synthData[:nums]) so it only runs the first nums files. Good for testing
	writeStats = []
	writeStatsGon = []
	writeStatsRand = []
	for i in range(1):#int(len(synthData)/10) - 1):
		statsRand = computeKCRandom(synthData[i*10:(i+1)*10])
		statsGon = computeKC(synthData[i*10:(i+1)*10])
		stats = computeKCoutliers(synthData[i*10:(i+1)*10])
		writeStats.extend(stats)
		writeStatsGon.extend(statsGon)
		writeStatsRand.extend(statsRand)

		#Ploting and writing plots
		newStatsRand = processStats(statsRand)
		newStats = processStats(stats)
		newStatsGon = processStats(statsGon)
		plotGonOutRand(newStats,newStatsGon, newStatsRand)

	#Writing stats kcout
	writeKCOStats(writeStats,"center.csv")

	#Writing stats gonzalez
	writeKCOStats(writeStatsGon, "gonzalez.csv")

	#Writing stats random
	writeKCOStats(writeStatsRand, "random.csv")
	
############################################################################################

  
main()
