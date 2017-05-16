# Ryan Berndt
# James Bocinsky
# Chris Brown
# Jesus Pintados

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from spectral import *
from sklearn.metrics.cluster import adjusted_rand_score


np.set_printoptions(threshold=np.nan)

#Loads the dictionaries
groundtruth = scipy.io.loadmat('Indian_pines_gt.mat')
datain = scipy.io.loadmat('Indian_pines_corrected.mat')

#Loads the arrays from the dictionaries
data = datain['indian_pines_corrected']
truth = groundtruth['indian_pines_gt']

#__________________________________functions________________________________________
#(m, c) = kmeans(final, numclusters, max_iterations) - performs k means

#showImages(time = 0.5, colors = 'nipy_spectral') - shows the hyperspectral images

#showGT(this = truth) - shows the ground truth image

#score = getRand(sol, dat) - Calculates Rand Index/Score with only the points we're supposed to check

#sol, dat, new_m = removeZeros(m) - #Removes the points we aren't checking by creating a 1-D array

#plotting(m, this = truth) - Plots the input m and the ground truth (or whatever is plugged in at each spot)

#new_m = Window3x3(m) - Edge smoothing based on majority of neighbors

#new_m = Window5x5(m) - Edge smoothing based on majority of neighbors

#_______________________________end function list_____________________________________


#Shows each layer of the hyperspectral data 
def showImages(time = 0.5, colors = 'nipy_spectral'):
	for k in range(0,200):
		showthis = data[:,:,k]
		plt.imshow(showthis, cmap= colors,  interpolation='nearest')
		plt.title(k)
		plt.pause(time)

#Shows the ground truth
def showGT(this = truth):
	plt.imshow(this, cmap='hot',  interpolation='nearest')
	plt.show()

#Calculates Rand Index/Score with only the points we're supposed to check
def getRand(sol, dat):
	return adjusted_rand_score(sol, dat)



#Removes the points we aren't checking by creating a 1-D array 
#(and replaces points with 0's in 'm' because aesthetics)
def removeZeros(m):
	new_m = m
	sol = []
	dat = []
	for j in range(0,len(truth)):
		for k in range(0, len(truth)):
			if(truth[j,k] == 0):
				new_m[j,k] = 0
			else:
				sol.append(truth[j,k])
				dat.append(m[j,k])
	return sol, dat, new_m

#Plots the input m and the ground truth (or whatever is plugged in at each spot)
def plotting(m, this = truth):
	plt.subplot(2,1,1)
	plt.imshow(m, cmap='nipy_spectral',  interpolation='nearest')
	plt.title("First Input (m)")
	plt.subplot(2,1,2)
	plt.imshow(truth, cmap='nipy_spectral',  interpolation='nearest')
	plt.title("Second Input(groundtruth)")
	plt.show()

#this calculates the number of same/different pixels around a point
		#also keeps track of what pixels are around it
def Window5x5(m):
	new_m = m
	for i in range(2, np.shape(m)[0]-2):
	#this calculates the number of same/different pixels around a point
	#also keeps track of what pixels are around it
	#0.288883046215 rand score (with 15 classes) after this
		for j in range(2, np.shape(m)[1]-2):
			diff = 0
			same = 0 
			clas = []

			#_____________________________TOP TOP LEVEL_______________________________
			#	top top left left
			if(m[i-2,j-2] != m[i,j]):
				diff += 1
				clas.append(m[i-2,j-2])
			else:
				same += 1
				clas.append(m[i-2,j-2])

			# top top left
			if(m[i-1,j-2] != m[i,j]):
				diff += 1
				clas.append(m[i-1,j-2])
			else:
				same += 1
				clas.append(m[i-1,j-2])
				
			# top top middle
			if(m[i,j-2] != m[i,j]):
				diff += 1
				clas.append(m[i,j-2])
			else:
				same += 1
				clas.append(m[i,j-2])
			# top top right
			if(m[i+1,j-2] != m[i,j]):
				diff += 1
				clas.append(m[i+1,j-2])
			else:
				same += 1
				clas.append(m[i+1,j-2])
			# top top right right
			if(m[i+2,j-2] != m[i,j]):
				diff += 1
				clas.append(m[i+2,j-2])
			else:
				same += 1
				clas.append(m[i+2,j-2])
			#___________________________END TOP TOP LEVEL_____________________________

			#_________________________________TOP LEVEL_______________________________
			#top left left
			if(m[i-2,j-1] != m[i,j]):
				diff += 1
				clas.append(m[i-2,j-1])
			else:
				same += 1
				clas.append(m[i-2,j-1])
			#top left
			if(m[i-1,j-1] != m[i,j]):
				diff += 1
				clas.append(m[i-1,j-1])
			else:
				same += 1
				clas.append(m[i-1,j-1])

			# #top middle
			if(m[i,j-1] != m[i,j]):
				diff += 1
				clas.append(m[i,j-1])
			else:
				same += 1
				clas.append(m[i,j-1])

			#top right
			if(m[i+1,j-1] != m[i,j]):
				diff += 1
				clas.append(m[i+1,j-1])
			else:
				same += 1
				clas.append(m[i+1,j-1])

			#top right right
			if(m[i+2,j-1] != m[i,j]):
				diff += 1
				clas.append(m[i+2,j-1])
			else:
				same += 1
				clas.append(m[i+2,j-1])
			#_____________________________END TOP LEVEL_______________________________

			#_______________________________MIDDLE LEVEL______________________________
			#middle left left
			if(m[i-2,j] != m[i,j]):
				diff += 1
				clas.append(m[i-2,j])
			else:
				same += 1
				clas.append(m[i-2,j])
			#middle left
			if(m[i-1,j] != m[i,j]):
				diff += 1
				clas.append(m[i-1,j])
			else:
				same += 1
				clas.append(m[i-1,j])
			#middle right
			if(m[i+1,j] != m[i,j]):
				diff += 1
				clas.append(m[i+1,j])
			else:
				same += 1
				clas.append(m[i+1,j])
			#middle right right
			if(m[i+2,j] != m[i,j]):
				diff += 1
				clas.append(m[i+2,j])
			else:
				same += 1
				clas.append(m[i+2,j])
			#_____________________________END MIDDLE LEVEL____________________________

			#______________________________BOTTOM LEVEL_______________________________
			#bottom left left
			if(m[i-2,j+1] != m[i,j]):
				diff += 1
				clas.append(m[i-2,j+1])
			else:
				same += 1
				clas.append(m[i-2,j+1])
			#bottom left
			if(m[i-1,j+1] != m[i,j]):
				diff += 1
				clas.append(m[i-1,j+1])
			else:
				same += 1
				clas.append(m[i-1,j+1])
			#bottom middle
			if(m[i,j+1] != m[i,j]):
				diff += 1
				clas.append(m[i,j+1])
			else:
				same += 1
				clas.append(m[i,j+1])
			#bottom right
			if(m[i+1,j+1] != m[i,j]):
				diff += 1
				clas.append(m[i+1,j+1])
			else:
				same += 1
				clas.append(m[i+1,j+1])
			#bottom right right
			if(m[i+2,j+1] != m[i,j]):
				diff += 1
				clas.append(m[i+2,j+1])
			else:
				same += 1
				clas.append(m[i+2,j+1])
			#____________________________END BOTTOM LEVEL_____________________________

			#_________________________BOTTOM BOTTOM LEVEL_____________________________
			#bottom bottom left left
			if(m[i-2,j+2] != m[i,j]):
				diff += 1
				clas.append(m[i-2,j+2])
			else:
				same += 1
				clas.append(m[i-2,j+2])
			#bottom bottom left
			if(m[i-1,j+2] != m[i,j]):
				diff += 1
				clas.append(m[i-1,j+2])
			else:
				same += 1
				clas.append(m[i-1,j+2])
			#bottom bottom middle
			if(m[i,j+2] != m[i,j]):
				diff += 1
				clas.append(m[i,j+2])
			else:
				same += 1
				clas.append(m[i,j+2])
			#bottom bottom right
			if(m[i+1,j+2] != m[i,j]):
				diff += 1
				clas.append(m[i+1,j+2])
			else:
				same += 1
				clas.append(m[i+1,j+2])
			#bottom bottom right right
			if(m[i+2,j+2] != m[i,j]):
				diff += 1
				clas.append(m[i+2,j+2])
			else:
				same += 1
				clas.append(m[i+2,j+2])
			#________________________END BOTTOM BOTTOM LEVEL___________________________

			if(diff > same+1):
				mode = max(set(clas), key=clas.count)
				new_m[i,j] = mode 
	return new_m


def Window3x3(m):
	new_m = m
	for i in range(1, np.shape(m)[0]-1):
		for j in range(1, np.shape(m)[1]-1):
			diff = 0
			same = 0 
			clas = []
			#top left
			if(m[i-1,j-1] != m[i,j]):
				diff += 1
				clas.append(m[i-1,j-1])
			else:
				same += 1
				# clas.append(m[i-1,j-1])
			# #top
			if(m[i,j-1] != m[i,j]):
				diff += 1
				clas.append(m[i,j-1])
			else:
				same += 1
				# clas.append(m[i,j-1])
			#top right
			if(m[i+1,j-1] != m[i,j]):
				diff += 1
				clas.append(m[i+1,j-1])
			else:
				same += 1
				# clas.append(m[i+1,j-1])

			#left
			if(m[i-1,j] != m[i,j]):
				diff += 1
				clas.append(m[i,j])
			else:
				same += 1
				clas.append(m[i,j])
			#right
			if(m[i+1,j] != m[i,j]):
				diff += 1
				clas.append(m[i+1,j])
			else:
				same += 1
				clas.append(m[i+1,j])

			#bottom left
			if(m[i-1,j+1] != m[i,j]):
				diff += 1
				clas.append(m[i-1,j+1])
			else:
				same += 1
				clas.append(m[i-1,j+1])
			#bottom
			if(m[i,j+1] != m[i,j]):
				diff += 1
				clas.append(m[i,j+1])
			else:
				same += 1
				clas.append(m[i,j+1])
			#bottom right
			if(m[i+1,j+1] != m[i,j]):
				diff += 1
				clas.append(m[i+1,j+1])
			else:
				same += 1
				clas.append(m[i+1,j+1])
			# print(same, diff)
			if(diff > same+2):
				mode = max(set(clas), key=clas.count)
				new_m[i,j] = mode
				# print(mode)
	return new_m


#Translated from James' c++ code 
#as a generic window function

# helpme = m
# for i in range(0+x, 144-x):
# 	for j in range(0+x, 144-x):
		
# 		datapoint = m[i,j];
# 		diff = 0;
# 		same = 0;
# 		clas = []
# 		for k in range(i-1,i+1):
# 			for l in range(j-1,j+1):
# 				clas.append(m[k,l]);
# 				if(k != i and l != j):
# 					if(m[k,l] != datapoint):
# 						diff += 1;
# 					else:
# 						same += 1;
# 		if(diff > same+1):
# 			mode = max(set(clas), key=clas.count)
# 			helpme[i,j] = mode
# # 			# print(mode)
# m = helpme
