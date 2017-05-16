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
TestX = np.load('Easy_Test_Images.npy').ravel()
TestY = np.load('Easy_Test_Y.npy').ravel()

#load the trained model
netest = test.Test('easy')
#test the trained model
output = netest.test(TestX)

#show accuracy metrics
correct = 0
print("Output...Actual")
for i in range(0,len(output)):
    if(output[i] == int(TestY[i])):
        correct += 1
    print('  ',output[i], '      ', int(TestY[i]))
print(str(correct))
print(str(correct/len(output)))
print(sklearn.metrics.confusion_matrix(TestY, output))
