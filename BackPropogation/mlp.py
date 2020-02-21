#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mlp.py
# Author: Qian Ge <qge2@ncsu.edu>

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv

sys.path.append('../')
import src.network2 as network2
import src.mnist_loader as loader
import src.activation as act

DATA_PATH = '../../data/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', action='store_true',
                        help='Check data loading.')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Check implementation of sigmoid.')
    parser.add_argument('--gradient', action='store_true',
                        help='Gradient check')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')


    return parser.parse_args()

def load_data():
    train_data, valid_data, test_data = loader.load_data_wrapper(DATA_PATH)
    print('Number of training: {}'.format(len(train_data[0])))
    print('Number of validation: {}'.format(len(valid_data[0])))
    print('Number of testing: {}'.format(len(test_data[0])))
    return train_data, valid_data, test_data

def test_sigmoid():
    z = np.arange(-10, 10, 0.1)
    y = act.sigmoid(z)
    y_p = act.sigmoid_prime(z)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(z, y)
    plt.title('sigmoid')

    plt.subplot(1, 2, 2)
    plt.plot(z, y_p)
    plt.title('derivative sigmoid')
    plt.show()

def gradient_check():
    train_data, valid_data, test_data = load_data()
    model = network2.Network([784, 40, 10])
    model.gradient_check(training_data=train_data, layer_id=1, unit_id=5, weight_id=3)

def main():
    # load train_data, valid_data, test_data
    train_data, valid_data, test_data = load_data()
    # construct the network
    model = network2.Network([784, 40, 10])
    # train the network using SGD
    valid_cost,valid_accuracy, train_cost, train_accuracy=model.SGD(
        training_data=train_data,
        epochs=100,
        mini_batch_size=128,
        eta=1e-3,
        lmbda = 0.0,
        evaluation_data=valid_data,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

    # Plots of the training and validation dataset
    
    #validation_Cost
    plt.plot(list(range(0,len(valid_cost))),valid_cost)
    plt.title("Validation Cost Plot")
    plt.savefig('Validation Cost')
    plt.clf()
    #validation_Accuracy
    plt.plot(list(range(0,len(valid_accuracy))),valid_accuracy)
    plt.title("Validation Accuracy Plot")
    plt.savefig('Validation Accuracy')
    plt.clf()
    #training_Cost
    plt.plot(list(range(0,len(train_cost))),train_cost)
    plt.title("Training Cost Plot")
    plt.savefig('Training Cost')
    plt.clf()
    #training_Accuracy
    plt.plot(list(range(0,len(train_accuracy))),train_accuracy)
    plt.title("Training Accuracy Plot")
    plt.savefig('Training Accuracy')
    plt.clf()  
    
    # Test Predictions
    test_accuracy = model.accuracy(test_data, convert=False)
    print(test_accuracy)
    predictions =[]
    testX=test_data[0]
    for i in range(0,len(testX)):
        pValue=model.feedforward(testX[i])
        predictions.append(pValue)
    
   
    # one- hot encoding
    for i in predictions :  
        m = (np.argmax(i))
        for j in range(0,len(i)) :  
            if j==m:
                i[j]=1
            else:
                i[j]=0
     #CSV file export           
    with open("PredictionEncoding.csv","w+") as file:
        csvFile =csv.writer(file, delimiter =',')
        csvFile.writerows(predictions)        
            

           
   

if __name__ == '__main__':
    FLAGS = get_args()
    if FLAGS.input:
        load_data()
    if FLAGS.sigmoid:
        test_sigmoid()
    if FLAGS.train:
        main()
    if FLAGS.gradient:
        gradient_check()
   
