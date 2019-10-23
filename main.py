import os

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

import mainCenter
import mainMeans
import LO_MNIST
import LO_NIPS
import LO_SKIN
