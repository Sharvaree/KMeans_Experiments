import numpy as np
import math
import random
import sys
import os
import statistics
import matplotlib.pyplot as plt
import operator
from scipy.spatial import distance

#Custom imports
import kcentersOutliers as kco
import generatorNormal as gn
import kcenterAux as kc
import generatorNormalCenters as gnc
import gonzalez as gon
import KMeansOut2 as kmo
import kmeansAux as km
import localSearch as ls

#Constants
extraInfo = ["optimal cost","av prec", "max prec", "av recall", "max recall", "prec sd", "recall sd"] # add header names to this list, e.g. ["cluster1cost", "cluster2cost"]. make sure values are numers, since they will be averaged over runs.

zprop = [0.5, 1, 2]
phistarprop = [0.5, 1, 2]
eps = 0.1

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
	phistar = 0
	runz = 1
	runphi = 0
	

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
	sds = [1, 5, 10]
	ks = [10, 20]
	zs = [25,50,100]
	ds = [2,15]
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
	print("syntheticData: [n: " + str(sd.n) + ", d: " + str(sd.d) + ", k: " + str(sd.k) + ", rang: " + str(int(sd.rang)) + ", z: " + str(sd.z) +", num: "+ str(sd.c) + ", sigma: " + str(sd.s) + ", runk", str(sd.runk) , ",runz", str(sd.runz), ",runphi", str(sd.runphi),", costs", str(sd.costs),"]")

#Adds the calculated answer of one k centers w/ outliers instance
def addAnswer(stats, sd):
	temp = [sd.n, sd.d,sd.k,sd.rang,sd.z,sd.s,sd.runk,sd.runphi]
	found = False
	for i in range(len(stats)):
		matching = True
		for j in range(8):
			if(not stats[i][j] == temp[j]):
				matching = False
		if(matching):
			stats[i][8].append(sd.costs)
			stats[i][9].append(sd.extrastats)
			stats[i][10].append(sd.precs)
			stats[i][11].append(sd.recs)
			found = True
	if(not found):
		temp.append([sd.costs])
		temp.append([sd.extrastats])
		temp.append([sd.precs])
		temp.append([sd.recs])
		stats.append(temp)
	return stats
		
#computes phi_star
def compute_phi_star(sd):
	return kmo.cost2(sd.data[sd.k + sd.z:], sd.data[:sd.k], int(sd.z))

#Compute k means w/ lsout
def computeKMLS(synthD):
	num = 0
	stats = []
	for f in synthD:
		#reads data and parses first file in folder
		sd = readSynthetic(f)
		sd.phistar = compute_phi_star(sd)
		print("Iteration:",num)
		num+=1

		for rp in phistarprop:
			for j in range(int(sd.k/2)):
				print("TrueCost:", sd.phistar)
				sd.runphi = rp
				sd.runk = sd.k + j
				precs = []
				recs = []
				for i in range(1):
					#Running kcenterOut on the data
					ans, empz = ls.lsOut(sd.data,sd.runk,sd.z, eps)

					cost2 = kmo.cost2(sd.data, ans, int(sd.z))

					#Computing cost
					sd.costs.append(cost2)

					prec, rec = kc.kCPrecRecall(sd,ans)
					precs.append(prec)
					recs.append(rec)

				sd.precs = precs
				sd.recs = recs

				#example for adding extra stats, i.e. time. For headers, go to top
				sd.extrastats = [0,mean(np.array(precs)), max(precs), mean(np.array(recs)), max(recs)]

				printSD(sd)
		
				stats = addAnswer(stats, sd)
				sd.costs = []
	
	return stats

#Compute thresholded k means w/ outliers
def computeKMoutliers(synthD):
	num = 0
	stats = []
	for f in synthD:
		#reads data and parses first file in folder
		sd = readSynthetic(f)
		sd.phistar = compute_phi_star(sd)
		print("-------------------------\nIteration:",num,"\n-------------------------")
		num+=1

		for rp in phistarprop:
			for j in range(int(sd.k/2)):
				sd.runphi = rp
				sd.runk = sd.k + j
				precs = []
				recs = []
				for i in range(3):
					#Running kMeansOut on the data
					ans, cid, dist, wins = kmo.kmeansOutliers(sd.data,sd.phistar*sd.runphi,sd.z, sd.runk)
					#kmo_cost, index_list = kmo.cost(sd.data, cid, ans, int(sd.z))
					#average_cost= np.sum(kmo_cost)
					cost2 = kmo.cost2(sd.data, ans, int(sd.z))
					#print("Sharvaree_cost:", average_cost)
					#assert(cost2 == average_cost)

					#Computing cost
					sd.costs.append(cost2)
					prec, rec = km.kMPrecRecall(sd,wins)
					precs.append(prec)
					recs.append(rec)

				sd.precs = precs
				sd.recs = recs

				#example for adding extra stats, i.e. time. For headers, go to top
				sd.extrastats = [sd.phistar,mean(np.array(precs)), max(precs), mean(np.array(recs)), max(recs)]

				printSD(sd)
		
				stats = addAnswer(stats, sd)
				sd.costs = []
	
	return stats

#Writes k means statistics to a csv file
def writeKMStats(stats, filename):
	header = ["n","d","k","rang","z","sigma","runk", "runphi"]
	newStats = []
	for i in range(len(stats[0][9])):
		header.append("Run " + str(i+1) + " cost")
	header.append("Average")
	header.extend(extraInfo)
	
	newStats.append(np.array(header))
	for i in range(len(stats)):
		temp = stats[i][0:8]
		for j in range(len(stats[i][8])):
			temp.append(mean(stats[i][8][j]))
		temp.append(mean(temp[8:]))

		s = len(stats[i][9][0])
		aver = [0] * s
		for j in range(len(stats[i][9])):
			for k in range(s):
				aver[k] += stats[i][9][j][k]
		for j in range(s):
			aver[j] = float(aver[j])/float(len(stats[i][9]))
		temp.extend(aver)

		#sds
		sds = [0.0]*2
		precs = np.array(stats[i][9])[:,1]
		recs = np.array(stats[i][9])[:,3]

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
		temp = stats[i][0:8]
		for j in range(len(stats[i][8])):
			temp.append(mean(stats[i][8][j]))
		temp.append(mean(temp[8:]))

		s = len(stats[i][9][0])
		aver = [0] * s
		for j in range(len(stats[i][9])):
			for k in range(s):
				aver[k] += stats[i][9][j][k]
		for j in range(s):
			aver[j] = float(aver[j])/float(len(stats[i][9]))
		temp.extend(aver)

		#sds
		sds = [0.0]*2
		precs = np.array(stats[i][9])[:,1]
		recs = np.array(stats[i][9])[:,3]

		sds[0] = np.std(precs)
		sds[1] = np.std(recs)
		temp.extend(sds)

		newStats.append(np.array(temp))

	newStats = np.array(newStats)
	return newStats

def plotNewStats(stats):
	ks = stats[:,6]
	avprec = stats[:, 17]
	maxprec = stats[:, 18]
	avrec = stats[:,19]
	maxrec = stats[:,20]

	plotpts = np.array([avprec, maxprec, avrec, maxrec])
	
	#adding title
	plt.title("Stats of [n: " + str(stats[0][0]) + ", d: " + str(stats[0][1]) + ", k: " + str(stats[0][2]) + ", rang: " + str(int(float(stats[0][3]))) + ", z: " + str(stats[0][4]) + ", sigma: " + str(stats[0][6]) + "]")
	name = "km_n" + str(stats[0][0]) + "d" + str(stats[0][1]) + "k" + str(stats[0][2]) + "r" + str(int(float(stats[0][3]))) + "z" + str(stats[0][4]) + "s" + str(stats[0][6]) + "z" + str(int(stats[0][7])) + "p" + str(int(stats[0][8])) + ".png"

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

def plot2(stats,stats2):

	ks = stats[:, 6]
	avprec = stats[:, 20]
	maxprec = stats[:, 21]
	avrec = stats[:,22]
	maxrec = stats[:,23]
	sdprec = stats[:,24]
	sdrec = stats[:,25]

	plotpts = np.array([avprec, maxprec, avrec, maxrec])	

	print(plotpts)
	
	ks2 = stats2[:,6]
	avprec2 = stats2[:, 20]
	maxprec2 = stats2[:, 21]
	avrec2 = stats2[:,22]
	maxrec2 = stats2[:,23]
	sdprec2 = stats2[:,24]
	sdrec2 = stats2[:,25]

	plotpts2 = np.array([avprec2, maxprec2, avrec2, maxrec2])
	print(plotpts2)

	#Start with center outliers
	
	plt.figure(figsize=(6.4,9.6))
	plt.suptitle("Figure   n=" + str(int(stats[0][0])) + ", d=" + str(int(stats[0][1])) + ", k=" + str(int(stats[0][2])) + ", rang=" + str(int(float(stats[0][3]))) + ", z=" + str(int(stats[0][4])) + ", sigma=" + str(int(stats[0][6]))+", phifactor=" + str(int(stats[0][7])) + ".")

	plt.subplot(211)
	plt.title("Precision")

	plt.xlabel("k")
	plt.ylabel("Precision")
	ax = plt.gca()
	ax.set_ylim([0,1])
	plt.errorbar(ks, avprec, yerr=sdprec)
	plt.plot(ks, avprec, linestyle = '-', label = "Mean Precision Algorithm 1")
	#plt.scatter(ks, maxprec)
	#plt.plot(ks, maxprec,linestyle = '--', label = "Max Precision Algorithm 1")
	plt.scatter(ks, avprec2)
	plt.plot(ks, avprec2,linestyle = '-.', label = "Mean Precision Gonzalez")
	#plt.scatter(ks, maxprec2)
	#plt.plot(ks, maxprec2,linestyle = ':', label = "Max Precision Gonzalez")
	plt.legend(loc='best')
	plt.tight_layout(rect = [0,0.03,1,0.95])

	plt.subplot(212)
	plt.title("Recall")

	plt.xlabel("k")
	plt.ylabel("Recall")
	ax = plt.gca()
	ax.set_ylim([0,1])
	plt.errorbar(ks, avrec, yerr = sdrec)
	plt.plot(ks, avrec,linestyle = '-', label = "Mean Recall Algorithm 1")
	#plt.scatter(ks, maxrec)
	#plt.plot(ks, maxrec,linestyle = '--', label = "Max Recall Algorithm 1")
	plt.scatter(ks, avrec2)
	plt.plot(ks, avrec2,linestyle = '-.', label = "Mean Recall Gonzalez")
	#plt.scatter(ks, maxrec2)
	#plt.plot(ks, maxrec2,linestyle = ':', label = "Max Recall Gonzalez")
	plt.legend(loc='best')
	plt.tight_layout(rect = [0,0.03,1,0.95])
	
	name = "km_n" + str(stats[0][0]) + "d" + str(stats[0][1]) + "k" + str(stats[0][2]) + "r" + str(int(float(stats[0][3]))) + "z" + str(stats[0][4]) + "s" + str(stats[0][6])+ "p" + str(int(stats[0][7])) + ".png"
	plt.savefig("visualizations/" + name)
	plt.clf()

def collapseAndAverageOver(k, precrec):
	uniquek = list(set(k))
	grouped = [0.0]*len(uniquek)
	num = len(k)/len(uniquek)
	for i in range(len(k)):
		for j in range(len(uniquek)):
			if(k[i] == uniquek[j]):
				grouped[j] += precrec[i]/num
	
	return np.array(grouped)

def plot2averagedK(stats,stats2):
	print("---------------------------\n",len(stats))
	for line in stats:
		print("	", len(line))
	print("---------------------------\n",len(stats2))
	for line in stats2:
		print("	", len(line))

	ks = stats[:, 6]
	avprec = collapseAndAverageOver(ks, stats[:, 20])
	maxprec = collapseAndAverageOver(ks, stats[:, 21])
	avrec = collapseAndAverageOver(ks, stats[:,22])
	maxrec = collapseAndAverageOver(ks, stats[:,23])
	sdprec = collapseAndAverageOver(ks, stats[:,24])
	sdrec = collapseAndAverageOver(ks, stats[:,25])
	ks = list(set(ks))

	plotpts = np.array([ks, avprec, maxprec, avrec, maxrec])	

	print(plotpts)
	
	ks2 = stats2[:,6]
	avprec2 = collapseAndAverageOver(ks2, stats2[:, 20])
	maxprec2 = collapseAndAverageOver(ks2, stats2[:, 21])
	avrec2 = collapseAndAverageOver(ks2, stats2[:,22])
	maxrec2 = collapseAndAverageOver(ks2, stats2[:,23])
	sdprec2 = collapseAndAverageOver(ks2, stats2[:,24])
	sdrec2 = collapseAndAverageOver(ks2, stats2[:,25])
	ks2 = list(set(ks2))

	plotpts2 = np.array([ks2, avprec2, maxprec2, avrec2, maxrec2])

	print(plotpts2)

	#Start with center outliers
	
	plt.figure(figsize=(6.4,9.6))
	plt.suptitle("Figure n=" + str(int(stats[0][0])) + ", d=" + str(int(stats[0][1])) + ", k=" + str(int(stats[0][2])) + ", rang=" + str(int(float(stats[0][3]))) + ", z=" + str(int(stats[0][4])) + ", sigma=" + str(int(stats[0][6]))+", phifactor=" + str(int(stats[0][7])) + ".")

	plt.subplot(211)
	plt.title("Precision")

	plt.xlabel("k")
	plt.ylabel("Precision")
	ax = plt.gca()
	ax.set_ylim([0,1])
	plt.errorbar(ks, avprec, yerr=sdprec, fmt='o',capsize=2, capthick=2)
	plt.plot(ks, avprec, linestyle = '-', label = "Mean Precision Algorithm 1")
	#plt.scatter(ks, maxprec)
	#plt.plot(ks, maxprec,linestyle = '--', label = "Max Precision Algorithm 1")
	plt.scatter(ks, avprec2)
	plt.plot(ks, avprec2,linestyle = '-.', label = "Mean Precision Gonzalez")
	#plt.scatter(ks, maxprec2)
	#plt.plot(ks, maxprec2,linestyle = ':', label = "Max Precision Gonzalez")
	plt.legend(loc='best')
	plt.tight_layout(rect = [0,0.03,1,0.95])

	plt.subplot(212)
	plt.title("Recall")

	plt.xlabel("k")
	plt.ylabel("Recall")
	ax = plt.gca()
	ax.set_ylim([0,1])
	plt.errorbar(ks, avrec, yerr = sdrec, fmt='o',capsize=2, capthick=2)
	plt.plot(ks, avrec,linestyle = '-', label = "Mean Recall Algorithm 1")
	#plt.scatter(ks, maxrec)
	#plt.plot(ks, maxrec,linestyle = '--', label = "Max Recall Algorithm 1")
	plt.scatter(ks, avrec2)
	plt.plot(ks, avrec2,linestyle = '-.', label = "Mean Recall Gonzalez")
	#plt.scatter(ks, maxrec2)
	#plt.plot(ks, maxrec2,linestyle = ':', label = "Max Recall Gonzalez")
	plt.legend(loc='best')
	plt.tight_layout(rect = [0,0.03,1,0.95])
	
	name = "km_avK_n" + str(stats[0][0]) + "d" + str(stats[0][1]) + "k" + str(stats[0][2]) + "r" + str(int(float(stats[0][3]))) + "z" + str(stats[0][4]) + "s" + str(stats[0][6])+ "p" + str(int(stats[0][7])) + ".png"
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
	y = flatten(stats[:,11])
	
	plt.figure(figsize=(12.8,9.6))
	plt.title("Figure   n=" + str(int(stats[0][0])) + ", d=" + str(int(stats[0][1])) + ", k=" + str(int(stats[0][2])) + ", rang=" + str(int(float(stats[0][3]))) + ", z=" + str(int(stats[0][4])) + ", sigma=" + str(int(stats[0][6]))+", phifactor=" + str(int(stats[0][7])) + ".")
	plt.xlabel("k")
	plt.ylabel("Recall")
	ax = plt.gca()
	ax.set_ylim([0,1])
		
	plt.boxplot(y, labels = x)

	name = "km_boxPlot_n" + str(stats[0][0]) + "d" + str(stats[0][1]) + "k" + str(stats[0][2]) + "r" + str(int(float(stats[0][3]))) + "z" + str(stats[0][4]) + "s" + str(stats[0][6])+ "p" + str(int(stats[0][7])) + ".png"
	plt.savefig("visualizations/" + name)
	plt.clf()

############################################################################################
#Main tasks
############################################################################################
def main():
	#Creates synthetic data as described in paper 10 copies each
	#createSynthDataGKL()
	#createSynthDataGKLCenters()
	
	#Gets the names of the synthetic data files
	synthData = getAllSynthNames()
	#synthData = getAllSynthNamesCenters()

	#Computing sds Note: can run stats = computeKCoutliers(synthData[:nums]) so it only runs the first nums files. Good for testing
	writeStats = []
	writeStatsGon = []
	for i in range(2):#int(len(synthData)/10) - 1):
		statsLS = computeKMLS(synthData[i*10:(i+1)*10])
		stats = computeKMoutliers(synthData[i*10:(i+1)*10])
		writeStats.extend(stats)
		writeStatsGon.extend(statsGon)

		boxPlot(stats)

		#Ploting and writing plots
		newStats = processStats(stats)
		newStatsLS = processStats(statsLS)		
		plot2averagedK(newStats,newStatsLS)

	#Writing stats kcout
	writeKMStats(writeStats,"threshmeans.csv")

	#Writing stats gonzalez
	writeKMStats(writeStatsLS, "LS.csv")
	
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
