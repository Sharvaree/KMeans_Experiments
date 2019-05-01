'''
Put SUSY.csv, Skin_NonSkin.txt, shuttle.trn, and covtype.data in realData and run.
'''

import numpy as np
import os

def preprocess():
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
