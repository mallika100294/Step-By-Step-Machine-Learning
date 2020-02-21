#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as 
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]
    

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    activations =[]
    for k in range(0,num_layers-1):
        if k==0:
            a= sigmoid(np.dot(weightsT[0], x )+biases[0])
            activations.append(a)
        else:
            a = sigmoid(np.dot(weightsT[k],activations[k-1])+biases[k])
            activations.append(a)
    
    ###
    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).df_wrt_a(activations[-1], y)
    
    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    
    for k in reversed (range(num_layers-1)):
        if k==0:
            nabla_b[k] = delta
            nabla_wT[k] = np.dot(delta, np.transpose(x)) 
            
        else:
        #dfwrta = np.dot(delta,sigmoid_prime(activations[k]))
            nabla_b[k] = delta
            nabla_wT[k] = np.dot(delta, np.transpose(activations[k-1])) 
            delta = np.multiply(np.dot(np.transpose(weightsT[k]),delta), sigmoid_prime(activations[k-1]))
        
    

    return (nabla_b, nabla_wT)
