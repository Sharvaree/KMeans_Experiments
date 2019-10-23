'''
Put SUSY.csv, Skin_NonSkin.txt, shuttle.trn, and covtype.data in realData and run.
'''

import numpy as np
import os
'''
from sh import gunzip
import wget
'''


def preprocess():
    '''
    destination = 'realData'
    exists = os.path.isfile('realData/SUSY.csv')
    if(not exists):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz'
        wget.download(url, destination + '/susy.csv.gz')
        gunzip('realData/SUSY.csv.gz')

    exists = os.path.isfile('realData/covtype.data')
    if(not exists):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
        wget.download(url, destination + '/covtype.data')
        gunzip('realData/covtype.data.gz')

    exists = os.path.isfile('realData/shuttle.trn')
    if(not exists):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/shuttle.trn.Z'
        wget.download(url, destination + '/shuttle.trn')

    exists = os.path.isfile('realData/Skin_NonSkin.txt')
    if(not exists):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
        wget.download(url, destination + '/Skin_NonSkin.txt')
    '''
    dirNam = "realData/"
    proDir = "realDataProcessed/"

    #Processing skin
    fileName = "Skin_NonSkin.txt"

    skin = np.genfromtxt(dirNam + fileName, delimiter='')

    np.savetxt(proDir + "skin.csv", skin, delimiter=",",  fmt='%f')

    
    #Processing shuttle
    shuttleName = "shuttle.trn"

    shuttle = np.genfromtxt(dirNam + shuttleName, delimiter='')

    np.savetxt(proDir + "shuttle.csv", shuttle, delimiter=",",  fmt='%f')

    
    #Processing CoverType
    covName = "covtype.data"

    cov = np.genfromtxt(dirNam + covName, delimiter=',')

    np.savetxt(proDir + "covertype.csv", cov, delimiter=",",  fmt='%f')

    
    #Processing SUSY
    susyName = "SUSY.csv"

    susy = np.genfromtxt(dirNam + covName, delimiter=',')

    np.savetxt(proDir + "SUSY.csv", susy, delimiter=",",  fmt='%f')
    

preprocess()
