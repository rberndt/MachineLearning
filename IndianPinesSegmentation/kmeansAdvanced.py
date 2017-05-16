# Ryan Berndt
# James Bocinsky
# Chris Brown
# Jesus Pintados

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from spectral import *
from sklearn.metrics.cluster import adjusted_rand_score
from header import *
from mpl_toolkits.mplot3d import Axes3D

# a=np.random.random(16)
# cs=cm.Set1(np.arange(100)/27)

np.set_printoptions(threshold=np.nan)

#Loads the dictionaries
groundtruth = scipy.io.loadmat('Indian_pines_gt.mat')
datain = scipy.io.loadmat('Indian_pines_corrected.mat')

#Loads the arrays from the dictionaries
data = datain['indian_pines_corrected']
truth = groundtruth['indian_pines_gt']

# surface = np.array([[18,3,0.336563558228434760],
# 					[16,4,0.329653266537430820],
# 					[10,5,0.317528945328291400],
# 					[10,6,0.316291240901283850],
# 					[8,7,0.3067989817847499600],
# 					[5,8,0.3042600760806400400],
# 					[5,9,0.3184881966095800500],
# 					[5,10,0.313722327145220750],
# 					[5,11,0.304908883515208280]])

# #Creating pie charts
# density = [20, 233, 144, 746, 193, 2133, 849, 317, 747, 270, 1300, 2261, 233, 482, 235, 86]
# plt.pie(density, colors = cs)
# plt.title("Class Density", fontsize=40)
# plt.show()

# densitytrue = [46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 86, 93]
# plt.pie(densitytrue, colors = cs)
# plt.title("True Class Density", fontsize=40)
# plt.show()



# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(surface[:,0],surface[:,1],surface[:,2], cmap=cm.coolwarm)
# plt.show()


#############################Make these into a 3-D surface and talk about it in report####################################
#maxes: [rand index, numClusters, max_iterations]| 	- [EDGE SMOOTHED VERSION]
#[0.336563558228434760, 18, 3] (OHHH YEAHHH PT1) |	- [0.331211613206667590, 18, 3]
#[0.329653266537430820, 16, 4]					 |	- [0.339391270265507920, 16, 4] (OHHHHHHHH YEAHHHHHHHHH PT2)<==============================
#[0.317528945328291400, 10, 5] 					 |	- [0.308646676256147500, 10, 5]
#[0.316291240901283850, 10, 6] 					 |	- [0.3224122404475334200, 8, 6]
#[0.3067989817847499600, 8, 7] 					 |	- [0.3116604296286402700, 8, 7]
#[0.3042600760806400400, 5, 8] 					 |	- [0.3085669668707380900, 5, 8]
#[0.3184881966095800500, 5, 9] 				 	 |	- [0.3160071269394456000, 5, 9]
#[0.313722327145220750, 5, 10] 					 |	- [0.317778410218010730, 5, 10]
#[0.304908883515208280, 5, 11] 					 |	- [0.31109572588881174, 17, 11]
#[0.298215939889472430, 5, 12] 					 |	- [0.300136172788122660, 5, 12]
#[0.264337627643726420, 5, 13] 					 |	- [0.25806814399429612, 16, 13]
#[0.226352928813732780, 5, 14] 					 |	- [0.26554969366569403, 18, 14]
#[0.22498492589795288, 14, 15] 					 |	- [0.26760922551469402, 18, 15]
#[0.222877252055868560, 5, 16] 					 |	- [0.32179051854954283, 13, 16]
# 												 |	- [0.31410753718006040, 13, 17]
#												 |	- [0.35651232010458889, 13, 18] (OOOOOOOOOOOOOHHHHHHHHHHHHHHHHHHHHHH YEEEEEEAAAAAAAAAHHHHHHHHHHHH PT 3)
#												 |	- [0.32119143783625986, 13, 19]
#												 |	- [0.28203864541455703, 15, 20]


# WITH BOTH Window5x5
# [0.34172155424699879, 18, 3]
# [0.33893612824115305, 16, 4]
# [0.33581496137108879, 14, 5]
# [0.31956888719099452, 8, 6]
# [0.3294414761632381, 8, 7]



#what if we ran k-means with these numbers(above) and created another set of data 
#which we ran k-means on, AGAIN?...

final2 = np.ndarray(shape=(145,145,32))
perc = []
for i in range(3, 19):
	#These can be played with in a for loop 
	#(but playing with both at the same time screws up the program for some reason...)

	if (i == 3 or i == 14 or i == 15):
		numclusters = 18
	elif(i == 4 or i == 14):
		numclusters = 16
	elif(i==5):
		numclusters = 10
	elif(i==6 or i == 7):
		numclusters = 8
	elif(i==8 or i==9 or i==10 or i==12):
		numclusters = 5
	elif(i==11):
		numclusters = 17
	else:
		numclusters = 13

	max_iterations = i
	

	#we can also try with more data splits
	new1 = np.ndarray(shape=(145,145,49))
	new2 = np.ndarray(shape=(145,145,49))
	new3 = np.ndarray(shape=(145,145,49))
	new4 = np.ndarray(shape=(145,145,49))
	datanp = np.array(data)

	#make a new array that only has the valuable matrices
	new1[:,:,:] = datanp[:,:,0:49]
	new2[:,:,:] = datanp[:,:,50:99]
	new3[:,:,:] = datanp[:,:,100:149]
	new4[:,:,:] = datanp[:,:,150:199]

	#perform k-means on each datasplit
	(m1, c1) = kmeans(new1, numclusters, max_iterations)
	(m2, c2) = kmeans(new2, numclusters, max_iterations)
	(m3, c3) = kmeans(new3, numclusters, max_iterations)
	(m4, c4) = kmeans(new4, numclusters, max_iterations)


	#NO WINDOW SMOOTHING

	#create a final data array
	final = np.ndarray(shape=(145,145,4))

	#import/append/concatenate the already-k-meaned data
	final[:,:,0] = m1
	final[:,:,1] = m2
	final[:,:,2] = m3
	final[:,:,3] = m4

	#perform a final k-means
	(m, c) = kmeans(final, numclusters, max_iterations)

	#Edge Smoothing and majority decision of datapoints that are surrounded
	m = Window5x5(m)

	final2[:,:,i-3] = m


for i in range(3, 12):

	if (i == 3):
		numclusters = 18
	elif(i == 4):
		numclusters = 16
	elif(i==5 or i == 6):
		numclusters = 10
	elif(i == 7):
		numclusters = 8
	else:
		numclusters = 5
	max_iterations = i
	

	#we can also try with more data splits
	new1 = np.ndarray(shape=(145,145,50))
	new2 = np.ndarray(shape=(145,145,50))
	new3 = np.ndarray(shape=(145,145,50))
	new4 = np.ndarray(shape=(145,145,50))
	datanp = np.array(data)

	#make a new array that only has the valuable matrices
	new1[:,:,:] = datanp[:,:,0::4]
	new2[:,:,:] = datanp[:,:,1::4]
	new3[:,:,:] = datanp[:,:,2::4]
	new4[:,:,:] = datanp[:,:,3::4]

	#perform k-means on each datasplit
	(m1, c1) = kmeans(new1, numclusters, max_iterations)
	(m2, c2) = kmeans(new2, numclusters, max_iterations)
	(m3, c3) = kmeans(new3, numclusters, max_iterations)
	(m4, c4) = kmeans(new4, numclusters, max_iterations)


	#NO WINDOW SMOOTHING 
	final = np.ndarray(shape=(145,145,4))

	#import/append/concatenate the already-k-meaned data
	final[:,:,0] = m1
	final[:,:,1] = m2
	final[:,:,2] = m3
	final[:,:,3] = m4

	#perform a final k-means
	(m, c) = kmeans(final, numclusters, max_iterations)


	final2[:,:,i+15] = m


for i in range(3, 7):
	#These can be played with in a for loop 
	#(but playing with both at the same time screws up the program for some reason...)

	if (i == 3):
		numclusters = 18
	elif(i == 4):
		numclusters = 16
	elif(i==5 or i == 6):
		numclusters = 14
	else:
		numclusters = 8

	max_iterations = i
	

	#we can also try with more data splits
	#we can also try with more data splits
	new1 = np.ndarray(shape=(145,145,50))
	new2 = np.ndarray(shape=(145,145,50))
	new3 = np.ndarray(shape=(145,145,50))
	new4 = np.ndarray(shape=(145,145,50))
	datanp = np.array(data)

	#make a new array that only has the valuable matrices
	new1[:,:,:] = datanp[:,:,0::4]
	new2[:,:,:] = datanp[:,:,1::4]
	new3[:,:,:] = datanp[:,:,2::4]
	new4[:,:,:] = datanp[:,:,3::4]

	#perform k-means on each datasplit
	(m1, c1) = kmeans(new1, numclusters, max_iterations)
	(m2, c2) = kmeans(new2, numclusters, max_iterations)
	(m3, c3) = kmeans(new3, numclusters, max_iterations)
	(m4, c4) = kmeans(new4, numclusters, max_iterations)


	#This does edge smoothing around with 3x3 (havent completed testing with this yet)
	m1 = Window5x5(m1)
	m2 = Window5x5(m2)
	m3 = Window5x5(m3)
	m4 = Window5x5(m4)

	#create a final data array
	final = np.ndarray(shape=(145,145,4))

	#import/append/concatenate the already-k-meaned data
	final[:,:,0] = m1
	final[:,:,1] = m2
	final[:,:,2] = m3
	final[:,:,3] = m4

	#perform a final k-means
	(m, c) = kmeans(final, numclusters, max_iterations)

	#Edge Smoothing and majority decision of datapoints that are surrounded
	m = Window5x5(m)


	#Removes the points we aren't checking by creating a 1-D array 
	#(and replaces points with 0's in 'm' because aesthetics)
	# sol, dat, m = removeZeros(m)


	#Calculates Rand Index/Score with only the points we're supposed to check
	# score = getRand(sol,dat)
	# print(score)
# 	perc.append([score, numclusters, max_iterations])
	final2[:,:,i+25] = m


#This is for testing all combinations
#######################################################
# max_iterations = 3
# while(max_iterations < 20):
# 	for i in range(0,100):
# 		print(max_iterations)

# 	for i in range(3,20):
# 		numclusters = i
# 		(m, c) = kmeans(final2, numclusters, max_iterations)

# 			#Edge Smoothing and majority decision of datapoints that are surrounded
# 		m = Window5x5(m)
# 		m = Window5x5(m)
# 		m = Window5x5(m)
# 		m = Window5x5(m)
# 		m = Window5x5(m)
# 		m = Window3x3(m)


# 		#Removes the points we aren't checking by creating a 1-D array 
# 		#(and replaces points with 0's in 'm' because aesthetics)
# 		sol, dat, m = removeZeros(m)


# 		#Calculates Rand Index/Score with only the points we're supposed to check
# 		score = getRand(sol,dat)
# 		print(score)
# 		perc.append([score, numclusters, max_iterations, "MULTIPLE SMOOTHING])

# 	for i in range(3,20):
# 		numclusters = i
# 		(m, c) = kmeans(final2, numclusters, max_iterations)

# 			#Edge Smoothing and majority decision of datapoints that are surrounded
# 		# m = Window5x5(m)


# 		#Removes the points we aren't checking by creating a 1-D array 
# 		#(and replaces points with 0's in 'm' because aesthetics)
# 		sol, dat, m = removeZeros(m)


# 		#Calculates Rand Index/Score with only the points we're supposed to check
# 		score = getRand(sol,dat)
# 		print(score)
# 		perc.append([score, numclusters, max_iterations, "NO SMOOTHING"])

# 	for i in range(3,20):
# 		numclusters = i
# 		(m, c) = kmeans(final2, numclusters, max_iterations)

# 			#Edge Smoothing and majority decision of datapoints that are surrounded
# 		m = Window3x3(m)


# 		#Removes the points we aren't checking by creating a 1-D array 
# 		#(and replaces points with 0's in 'm' because aesthetics)
# 		sol, dat, m = removeZeros(m)


# 		#Calculates Rand Index/Score with only the points we're supposed to check
# 		score = getRand(sol,dat)
# 		print(score)
# 		perc.append([score, numclusters, max_iterations, "NORMAL SMOOTHING"])
# 	max_iterations +=1

# print(perc)
# print("max: ", max(perc))
######################################################
numclusters = 19
max_iterations = 10
(m, c) = kmeans(final2, numclusters, max_iterations)

#Edge Smoothing and majority decision of datapoints that are surrounded

m = Window5x5(m)
m = Window5x5(m)
m = Window5x5(m)
m = Window5x5(m)
m = Window5x5(m)
m = Window3x3(m)

fig1 = plt.figure()
plt.imshow(m, cmap='nipy_spectral',  interpolation='nearest')
plt.title("Classification Map (m)", fontsize=40)
plt.xlabel('X Pixels', fontsize=25)
plt.ylabel('Y Pixels', fontsize=25)
plt.show()

#Removes the points we aren't checking by creating a 1-D array 
#(and replaces points with 0's in 'm' because aesthetics)
sol, dat, newm = removeZeros(m)


#Calculates Rand Index/Score with only the points we're supposed to check
score = getRand(sol,dat)
print(score)
fig2 = plt.figure()
plt.imshow(m, cmap='nipy_spectral',  interpolation='nearest')
plt.title("Zero'd (m)", fontsize=40)
plt.xlabel('X Pixels', fontsize=25)
plt.ylabel('Y Pixels', fontsize=25)

fig3 = plt.figure()
plt.imshow(truth, cmap='nipy_spectral',  interpolation='nearest')
plt.title("GroundTruth", fontsize=40)
plt.xlabel('X Pixels', fontsize=25)
plt.ylabel('Y Pixels', fontsize=25)
plt.show()

#imshow of both the final dataset (m) and the ground truth 
# plotting(newm, truth)

# #top Score
# max:  [0.39972278130658589, 19, 10, '5x5']


# table_iterations = 10
# tableClusters = 19
# 	(m, c) = kmeans(final2, tableClusters, table_iterations)

# 	sol, dat, m = removeZeros(m)


# 	#Calculates Rand Index/Score with only the points we're supposed to check
# 	score = getRand(sol,dat)
# 	print(score)
# 	perc.append([score, tableClusters])
# print(perc)
