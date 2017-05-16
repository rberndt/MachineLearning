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
Xs = TrainX[0:200]
Ys = TrainY[0:200]

#create the model
net = train.Train(.5, Xs, Ys, 'easy')
#train the model
net.train(Xs, Ys)
