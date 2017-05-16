# Ryan Berndt
# James Bocinsky
# Chris Brown
# Jesus Pintados

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from spectral import *
from sklearn.metrics.cluster import adjusted_rand_score
import math

#Loads the dictionaries
groundtruth = scipy.io.loadmat('Indian_pines_gt.mat')
datain = scipy.io.loadmat('Indian_pines_corrected.mat')

#Loads the arrays from the dictionaries
data = datain['indian_pines_corrected']
truth = groundtruth['indian_pines_gt']

#Shows the ground truth
# plt.imshow(truth, cmap='hot',  interpolation='nearest')
# plt.show()

#Shows each layer of the hyperspectral data 
# for k in range(0,199):
#   printthis = data[:,:,k]
#   plt.imshow(printthis, cmap='nipy_spectral',  interpolation='nearest')
#   plt.title(k)
#   plt.pause(.2)

numClusters = 16
numIterations = 4
(m, c) = kmeans(data,numClusters,numIterations)
# print(np.shape(m))
# answer = np.ravel(truth)
# dataflat = np.ravel(m)
# score = adjusted_rand_score(answer, dataflat)
# print("Score before removing zeros: ", score)

#Removes the points we aren't checking by creating a 1-D array 
#(and replaces points with 0's in 'm' because aesthetics)
# sol = []
# dat = []
# for j in range(0,len(truth)):
#     for k in range(0, len(truth[j])):
#         if(truth[j,k] == 0):
#             m[j,k] = 0
#         else:
#             sol.append(truth[j,k])
#             dat.append(m[j,k])
# #Calculates Rand Index/Score with only the points we're supposed to check
# score = adjusted_rand_score(sol, dat)
# print("Score before smoothing: ", score)

#smoothing
smoothM = m
amounts = []
for r in range(0,9):
    amounts.append(0)
for i in range(0,len(m)):
    for j in range(0, len(m[i])):
        if(i == 0 or i == (len(m)-1) or j == 0 or j == (len(m[j])-1)):
            smoothM[i][j] = m[i][j]
        else:
            amounts[0] = m[i-1][j-1]
            amounts[1] = m[i-1][j]
            amounts[2] = m[i-1][j+1]
            amounts[3] = m[i][j-1]
            amounts[4] = m[i][j]
            amounts[5] = m[i][j+1]
            amounts[6] = m[i+1][j-1]
            amounts[7] = m[i+1][j]
            amounts[8] = m[i+1][j+1]
            mostOccuring = max(set(amounts), key=amounts.count)
            smoothM[i][j] = mostOccuring

#smoothing more
smoothM2 = smoothM
amounts = []
for r in range(0,9):
    amounts.append(0)
for i in range(0,len(m)):
    for j in range(0, len(m[i])):
        if(i == 0 or i == (len(m)-1) or j == 0 or j == (len(m[j])-1)):
            smoothM2[i][j] = m[i][j]
        else:
            amounts[0] = m[i-1][j-1]
            amounts[1] = m[i-1][j]
            amounts[2] = m[i-1][j+1]
            amounts[3] = m[i][j-1]
            amounts[4] = m[i][j]
            amounts[5] = m[i][j+1]
            amounts[6] = m[i+1][j-1]
            amounts[7] = m[i+1][j]
            amounts[8] = m[i+1][j+1]
            mostOccuring = max(set(amounts), key=amounts.count)
            smoothM2[i][j] = mostOccuring

smoothSol = []
smoothDat = []
answer = np.ravel(truth)
dat = np.ravel(smoothM2)
for j in range(0,len(answer)):
    if(answer[j] != 0):
        smoothSol.append(answer[j])
        smoothDat.append(dat[j])

#Calculates Rand Index/Score with only the points we're supposed to check
score = adjusted_rand_score(smoothSol, smoothDat)
print("Score after smoothing: ", score)

plt.imshow(m, cmap='nipy_spectral',  interpolation='nearest')
plt.show()
plt.imshow(smoothM2, cmap='nipy_spectral',  interpolation='nearest')
plt.show()
plt.imshow(truth, cmap='nipy_spectral',  interpolation='nearest')
plt.show()