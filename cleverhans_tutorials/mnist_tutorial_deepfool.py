"""
This tutorial shows how to generate some simple adversarial examples
and train a model using adversarial training using nothing but pure
TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import DeepFool
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level

import os

import random
import matplotlib.pyplot as plt



FLAGS = flags.FLAGS


def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    dp_params = {
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([2017, 8, 30])

    if clean_train:
        model = make_basic_cnn(nb_filters=nb_filters)
        preds = model.get_probs(x)
        
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
     
        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)
        
        
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, rng=rng)
        
       
        s = []
        for i in range(0,len(X_test),1):
            pred = sess.run(preds, {x: X_test[i:i+1]})
            print(pred)
            print(Y_test[i:i+1])
            s.append(np.sort(pred)[0,-1]-np.sort(pred)[0,-2])
        #Draw a histogram
        def draw_hist(myList,Title,Xlabel,Ylabel):
            plt.hist(myList,np.arange(0,1,0.01),normed=True,stacked=True,facecolor='blue')
            plt.xlabel(Xlabel)       
            plt.ylabel(Ylabel)
            plt.title(Title)
            plt.show()
        draw_hist(myList=s,Title='legitimate',Xlabel='difference between max and second largest',
               Ylabel='Probability')
       

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the deepfool attack object and
        # graph
        
        deepfool = DeepFool(model,back='tf',sess=sess)
        adv_x = deepfool.generate(x, **dp_params)
        preds_adv = model.get_probs(adv_x)
        
    
        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        
        s = []
        for i in range(0,len(X_test),1):
            pred = sess.run(preds_adv, {x: X_test[i:i+1]})
            print(pred)
            print(Y_test[i:i+1])
            s.append(np.sort(pred)[0,-1]-np.sort(pred)[0,-2])
        
        #Draw a histogram
        def draw_hist(myList,Title,Xlabel,Ylabel):
            plt.hist(myList,np.arange(0,1,0.01),normed=True,stacked=True,facecolor='red')
            plt.xlabel(Xlabel)       
            plt.ylabel(Ylabel)
            plt.title(Title)
            plt.show()
        draw_hist(myList=s,Title='adversarial',Xlabel='difference between max and second largest',
               Ylabel='Probability')
        
    
        report.clean_train_adv_eval = acc

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc
        return report
    
def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs , batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)

   


if __name__ == '__main__':
        
        flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')

        flags.DEFINE_integer('nb_epochs', 6, 'Number of epochs to train model')

        flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
        
        flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

        flags.DEFINE_bool('clean_train', True, 'Train on clean examples')

        flags.DEFINE_bool('backprop_through_attack', False,

                         ('If True, backprop through adversarial example '

                         'construction process during adversarial training'))
        
        tf.app.run()
        
        
        
 
     
        
        
        



