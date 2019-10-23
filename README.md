# Adaptive Sampling k-Means++ in the presence outliers
---

This repository contains the experimental
code for our paper. The goal is to test 
KC-Outlier and T-kmeans++ against known algorithms. 

---
## Python and Numpy Versions:
	Python 3.6.5
	Numpy 1.14.3
	Scipy 1.2.1

---
## Run

To run all experiments, run 

> python3 main.py

To create synthetic datasets and run KC-Outlier experiments, run

> python3 mainCenter.py

To create synthetic datasets and run T-kmeans++ experiments, run

> python3 mainMeans.py

To run on real datasets, run

> python3 LO_<dataset>.py

where <dataset> is either NIPS, MNIST, or SKIN depending on the dataset one wants to run on.

Note: Make sure you have the necessary python libraries intalled.

---
## Results

Spreadsheet results are outputted into the outputs directory.

Visual results are outputted into the visualizations directory.
