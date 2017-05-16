import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from resizeimage import resizeimage
import random
import tensorflow as tf
import math
import train
import test
import sklearn.metrics

#load the data
TrainX = np.load('TrainImages.npy')
TrainY = np.load('TrainY.npy')
Xs = TrainX[0:1654]
Ys = TrainY[0:1654]

#create the model
net = train.Train(.5, Xs, Ys, 'hard')
#train the model
net.trainHard(Xs, Ys)
