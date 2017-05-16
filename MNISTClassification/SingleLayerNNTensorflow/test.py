import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from resizeimage import resizeimage
import random
import tensorflow as tf
import math

class Test():

    def __init__(self, mode, eta =.5, numPixels = 784):
        self.picSize = numPixels

        if (mode == 'easy'):
            self.size = 2
        elif(mode == 'hard'):
            self.size = 9

        # Create the model
        self.x = tf.placeholder(tf.float32, [None, self.picSize]) #placeholder is a value that is given when computing, None means dimension of any length
        self.W = tf.Variable(tf.zeros([self.picSize, self.size])) #model paramaters are generally variables
        self.b = tf.Variable(tf.zeros([self.size]))

        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, self.size])

        # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
        # outputs of 'y', and then average across the batch.
        self.y = tf.matmul(self.x, self.W) + self.b
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        self.train_step = tf.train.GradientDescentOptimizer(eta).minimize(self.cross_entropy)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.prediction = tf.argmax(self.y,1)

    def test(self, X):
        self.saver.restore(self.sess, "/tmp/model.ckpt")
        # Test trained model
        Xs = X
        for i in range(0,Xs.size,1):
            x = Xs[i]
            im = Image.fromarray(x)
            im = im.resize((28,28))
            x = np.asarray(im)
            Xs[i] = x

        predictionVector = []
        for i in range(0, X.size, 1):
            test_xs = Xs[i]
            test_xs = np.reshape(test_xs, (-1, self.picSize))
            predictionVector.append(int(self.prediction.eval(feed_dict={self.x: test_xs}, session=self.sess))+1)
            # print(self.prediction.eval(feed_dict={self.x: test_xs}, session=self.sess))

        return predictionVector

    def testHard(self, X):
        self.saver.restore(self.sess, "/tmp/modelHard.ckpt")
        # Test trained model
        Xs = X
        for i in range(0,Xs.size,1):
            x = Xs[i]
            im = Image.fromarray(x)
            im = im.resize((28,28))
            x = np.asarray(im)
            Xs[i] = x

        predictionVector = []
        for i in range(0, X.size, 1):
            test_xs = Xs[i]
            test_xs = np.reshape(test_xs, (-1, self.picSize))
            predictionVector.append(int(self.prediction.eval(feed_dict={self.x: test_xs}, session=self.sess))+1)

        return predictionVector
