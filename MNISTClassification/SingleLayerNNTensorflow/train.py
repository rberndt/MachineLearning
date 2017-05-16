import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from resizeimage import resizeimage
import random
import tensorflow as tf
import math

class Train():

    def __init__(self, eta, inputs, targets, mode = 'easy', numPixels = 784):
        if (mode == 'easy'):
            self.size = 2
        elif(mode == 'hard'):
            self.size = 9
        self.eta = eta
        self.Xsize = np.shape(inputs)[0]
        self.Ysize = np.shape(targets)[0]
        self.picSize = numPixels


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
        self.train_step = tf.train.GradientDescentOptimizer(self.eta).minimize(self.cross_entropy)

        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()

        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.prediction = tf.argmax(self.y,1)


    def train(self, X, Y):
        Xs = X
        Ys = Y

        #Code to resize images
        for i in range(0,Xs.size,1):
            x = Xs[i]
            im = Image.fromarray(x)
            im = im.resize((28,28))
            x = np.asarray(im)
            Xs[i] = x

        #holds the random numbers generated (in a random order)
        xarr = random.sample(range(0,Xs.size), Xs.size)

        setOne = math.floor(Xs.size*0.25)
        setTwo = math.floor(Xs.size*0.50)
        setThree = math.floor(Xs.size*0.75)

        for i in range(0,4):
            if(i == 1):
                trainArr = xarr[0:setThree]
                testArr = xarr[setThree:Xs.size]
            elif(i == 2):
                trainArr = xarr[0:setTwo]
                trainArr.extend(xarr[setThree:Xs.size])
                testArr = xarr[setTwo:setThree]
            elif(i == 3):
                trainArr = xarr[0:setOne]
                trainArr.extend(xarr[setTwo:Xs.size])
                testArr = xarr[setOne:setTwo]
            else:
                trainArr = xarr[setOne:Xs.size]
                testArr = xarr[0:setOne]

            # Train
            for i in range(0, len(trainArr), 1):
                batch_xs = Xs[trainArr[i]]
                batch_xs = np.reshape(batch_xs, (-1, self.picSize))
                if(str(Ys[trainArr[i]]) == "[1]"):
                    #add a
                    batch_ys = np.array([[1,0]])
                if(str(Ys[trainArr[i]]) == "[2]"):
                    #add b
                    batch_ys = np.array([[0,1]])
                self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

            # Validate trained model
            correct = 0
            insessCorrect = 0
            for i in range(0, len(testArr), 1):
                test_xs = Xs[testArr[i]]
                test_xs = np.reshape(test_xs, (-1, self.picSize))
                if(str(Ys[testArr[i]]) == "[1]"):
                    test_ys = np.array([[1,0]])
                if(str(Ys[testArr[i]]) == "[2]"):
                    test_ys = np.array([[0,1]])
                insessCorrect = (self.sess.run(self.accuracy, feed_dict={self.x: test_xs, self.y_: test_ys}))
                correct += insessCorrect
            print("Validation: %s" % str(correct/len(testArr)))

        save_path = self.saver.save(self.sess, "/tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)


    def trainHard(self, X, Y):
        Xs = X
        Ys = Y

        #Code to resize images
        for i in range(0,Xs.size,1):
            x = Xs[i]
            im = Image.fromarray(x)
            im = im.resize((28,28))
            x = np.asarray(im)
            Xs[i] = x

        #holds the random numbers generated (in a random order)
        xarr = random.sample(range(0,Xs.size), Xs.size)

        setOne = math.floor(Xs.size*0.25)
        setTwo = math.floor(Xs.size*0.50)
        setThree = math.floor(Xs.size*0.75)

        for i in range(0,4):
            if(i == 1):
                trainArr = xarr[0:setThree]
                testArr = xarr[setThree:Xs.size]
            elif(i == 2):
                trainArr = xarr[0:setTwo]
                trainArr.extend(xarr[setThree:Xs.size])
                testArr = xarr[setTwo:setThree]
            elif(i == 3):
                trainArr = xarr[0:setOne]
                trainArr.extend(xarr[setTwo:Xs.size])
                testArr = xarr[setOne:setTwo]
            else:
                trainArr = xarr[setOne:Xs.size]
                testArr = xarr[0:setOne]

            # Train
            for i in range(0, len(trainArr), 1):
                batch_xs = Xs[trainArr[i]]
                batch_xs = np.reshape(batch_xs, (-1, self.picSize))
                if(str(Ys[trainArr[i]]) == "[1]"):
                    batch_ys = np.array([[1,0,0,0,0,0,0,0,0]])
                elif(str(Ys[trainArr[i]]) == "[2]"):
                    batch_ys = np.array([[0,1,0,0,0,0,0,0,0]])
                elif(str(Ys[trainArr[i]]) == "[3]"):
                    batch_ys = np.array([[0,0,1,0,0,0,0,0,0]])
                elif(str(Ys[trainArr[i]]) == "[4]"):
                    batch_ys = np.array([[0,0,0,1,0,0,0,0,0]])
                elif(str(Ys[trainArr[i]]) == "[5]"):
                    batch_ys = np.array([[0,0,0,0,1,0,0,0,0]])
                elif(str(Ys[trainArr[i]]) == "[6]"):
                    batch_ys = np.array([[0,0,0,0,0,1,0,0,0]])
                elif(str(Ys[trainArr[i]]) == "[7]"):
                    batch_ys = np.array([[0,0,0,0,0,0,1,0,0]])
                elif(str(Ys[trainArr[i]]) == "[8]"):
                    batch_ys = np.array([[0,0,0,0,0,0,0,1,0]])
                else:
                    batch_ys = np.array([[0,0,0,0,0,0,1,0,1]])
                self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

            # Validate trained model
            correct = 0
            insessCorrect = 0
            for i in range(0, len(testArr), 1):
                test_xs = Xs[testArr[i]]
                test_xs = np.reshape(test_xs, (-1, self.picSize))
                if(str(Ys[testArr[i]]) == "[1]"):
                    test_ys = np.array([[1,0,0,0,0,0,0,0,0]])
                elif(str(Ys[testArr[i]]) == "[2]"):
                    test_ys = np.array([[0,1,0,0,0,0,0,0,0]])
                elif(str(Ys[testArr[i]]) == "[3]"):
                    test_ys = np.array([[0,0,1,0,0,0,0,0,0]])
                elif(str(Ys[testArr[i]]) == "[4]"):
                    test_ys = np.array([[0,0,0,1,0,0,0,0,0]])
                elif(str(Ys[testArr[i]]) == "[5]"):
                    test_ys = np.array([[0,0,0,0,1,0,0,0,0]])
                elif(str(Ys[testArr[i]]) == "[6]"):
                    test_ys = np.array([[0,0,0,0,0,1,0,0,0]])
                elif(str(Ys[testArr[i]]) == "[7]"):
                    test_ys = np.array([[0,0,0,0,0,0,1,0,0]])
                elif(str(Ys[testArr[i]]) == "[8]"):
                    test_ys = np.array([[0,0,0,0,0,0,0,1,0]])
                else:
                    test_ys = np.array([[0,0,0,0,0,0,1,0,1]])
                insessCorrect = (self.sess.run(self.accuracy, feed_dict={self.x: test_xs, self.y_: test_ys}))
                correct += insessCorrect
            print("Validation: %s" % str(correct/len(testArr)))

        save_path = self.saver.save(self.sess, "/tmp/modelHard.ckpt")
        print("Model saved in file: %s" % save_path)
