# -*- coding: utf-8 -*-

## verify.py -- check the accuracy of a neural network
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##';
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
#from setup_inception import ImageNet, InceptionModel

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from pylab import *   #support chinese

BATCH_SIZE = 1

with tf.Session() as sess:
    data, model = MNIST(), MNISTModel("models/mnist", sess)
    #data, model = MNIST(), MNISTModel("models/mnist-distilled-100", sess)
    #data, model = CIFAR(), CIFARModel("models/cifar", sess)
    #data, model = CIFAR(), CIFARModel("models/cifar-distilled-100", sess)
    # data, model = ImageNet(), InceptionModel(sess)

    x = tf.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
    y = model.predict(x)

    r = []
    s = []
    p = [] 
    for i in range(0,len(data.test_data),BATCH_SIZE):
        pred = sess.run(y, {x: data.test_data[i:i+BATCH_SIZE]})
        print(pred)
        print(data.test_labels[i:i+1])
        
        
        #print('real',data.test_labels[i],'pred',np.argmax(pred))
        r.append(np.argmax(pred,1) == np.argmax(data.test_labels[i:i+BATCH_SIZE],1))
        print('Test accuracy on legitimate examples：',np.mean(r))
        
        #difference array s
        print(np.sort(pred))
        s.append(np.sort(pred)[0,-1]-np.sort(pred)[0,-2])
            
        #the probability that the second largest value is correct classification
        if np.argmax(pred,1) != np.argmax(data.test_labels[i:i+BATCH_SIZE],1):
           index=np.where(pred==np.sort(pred)[0,-2])
           p.append(index[1][0] == np.argmax(data.test_labels[i:i+BATCH_SIZE],1))
        print("the probability that the second largest value is correct classification:",np.mean(p))    
    
    #draw Scatter plot
    x=range(0,len(data.test_data))
    y=s
        
    plt.xlabel("x")
    plt.ylabel("difference between max and the second")
    plt.scatter(x,y,marker='o')
    plt.show()
