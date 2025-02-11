# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:28:24 2024

@author: cekna
"""
import numpy as np
import random

'''
functions:
    player - plays the game by taking the input of the game and choosing a move
    create - creates the weight matrices for the NN that player takes as inputs
    sigmoid - used in the function player to transform the feature matrices
    '''

def sigmoid(z):
    '''Function that takes an input z and returns the sigmoid transformation.
    If z is a vector or a matrix, it should perform the sigmoid transformation on every element.
    From Machine Learning for Chemcial Engineers Notebook 8
    Input:
    z [=] scalar or array
    Return:
    g [=] scalar or array
    '''

    # Calculate sigmoid transform of z
    g = 1/(1+np.exp(-z))

    return g


def create(DNA, loud = False):
    ''' This function takes the raw DNA as the input. Unravels it into
    chromosomes. Then based on Chromosome 0 cuts and arranges Theta weight
    matrices to fit the shape of the new neural network.
    Input:
        DNA: the list of chromosomes taken from a GA or Rand_CRs
        loud: boolean for optional print output
    Output:
        brain: a list of weight matrices for the NN'''
    
    # Unravel DNA
    for i in range(len(DNA)):
        globals()['CR'+str(i)] = DNA[i]
        
    # Initialize weight matrices
    Theta1 = 0
    Theta2 = 0
    Theta3 = 0
    Theta4 = 0
    
    # Create weight Matrices    
    
    # Create Theta1 Matrix
    
    Row1 = int(CR0[1])              # Measure row dimensions based on number of neurons in first layer
    Theta1 = CR1[0:Row1,:]          # Cut CR1 (raw matrix) to fit size
    
    
    # Create Theta2 Matrix if layer 2 or 3 active
    
    if CR0[2] == 1:                 # if layer 2 is active
        Row2 = int(CR0[3])              # number of neurons in layer 2
        Col2 = Row1+1                   # number of neurons in layer 1 + bias
        Theta2 = CR2[0:Row2,0:Col2]     # Cut CR2 (raw matrix) to fit size
    elif CR0[4] == 1:               # if layer 3 is active and layer 2 is not 
        Row2 = int(CR0[5])              # number of neurons in layer 3
        Col2 = Row1+1                   # number of neurons in layer 1 + bias
        Theta2 = CR3[0:Row2,0:Col2]     # Cut CR3 (raw matrix) to fit size
    
    
    # Create Theta3 Matrix if layer 2 and 3 active
    
    if CR0[4] == 1 and CR0[2] == 1: # if layers 2 and 3 are active
        Row3 = int(CR0[5])              # number of neurons in layer 3
        Col3 = Row2+1                   # number of neurons in layer 2 + bias
        Theta3 = CR3[0:Row3,0:Col3]     # Cut CR3 (raw matrix) to fit size
    
    
    # Create Output Matrix (Theta 2 or 3 or 4) from CR4
    
    if type(Theta3) != int and type(Theta2) != int and type(Theta1) != int: # if all 3 layers are active
         Col4 = Row3+1                                                          # number of neurons in layer 3 + bias
         Theta4 = CR4[:,0:Col4]                                                 # Cut CR4 to size
    elif type(Theta2) != int and type(Theta1) != int:                       # if 2 layers are active
         Col4 = Row2+1                                                          # number of neurons in layer 2 or 3 + bias
         Theta3 = CR4[:,0:Col4]                                                 # Cut CR4 to size
    elif type(Theta1) != int:                                               # if only 1 layer active
         Col4 = Row1+1                                                          # number of neurons in layer 1 + bias
         Theta2 = CR4[:,0:Col4]                                                 # Cut CR4 to size
    
    
    # Print CR0 and sizes of weight matrices
    if loud:
        print(CR0)
        if type(Theta1) != int:
            print('Theta 1:',Theta1.shape)
        if type(Theta2) != int:    
            print('Theta 2:',Theta2.shape)
        if type(Theta3) != int:
            print('Theta 3:',Theta3.shape)
        if type(Theta4) != int:
            print('Theta 4:',Theta4.shape)
    
    # Return all Thetas. Thetas that are not active are integer zeros.
    brain = (Theta1,Theta2,Theta3,Theta4)
    return(brain)


def player(gameboard,brain, loud = False):
    ''' This function takes the gameboard and uses the brain to make decisions
    about how to move in Tetris
    Input: 
        gameboard: the gameboard vector
        brain: the brain outputted from "create", a list of weight matrices
        loud: boolean for optional print output
    Output:
        output: a vector of probabilities for each possible move
    '''
    
    # Determine the size of the brain for number of operations
    int_count = sum(isinstance(x, int) for x in brain)
    brain_size = 4 - int_count
    
    # Add bias to gameboard
    gameboard = np.insert(gameboard, 0, 1)
    
    # Compute Layer 1 operations
    layer1 = gameboard@brain[0].T # MatMult of gameboard and weights
    layer1 = sigmoid(layer1) # take sigmoid of output
    layer1 = np.insert(layer1, 0, 1) # insert bias for next computation
   
    layer2 = layer1@brain[1].T # MatMult of layer1 and weights
    layer2 = sigmoid(layer2) # take sigmoid of output
    output = layer2 # define final output
    
    # print brain size
    if loud:
        print('size =', brain_size)
    
    # if brain is big enough, continue operations
    if brain_size >= 3:
        layer2 = np.insert(layer2, 0, 1) # insert bias for next computation
        layer3 = layer2@brain[2].T # MatMult of layer2 and weights
        layer3 = sigmoid(layer3) # take sigmoid of output
        output = layer3 # define final output
        
    if brain_size == 4:
        layer3 = np.insert(layer3, 0, 1) # insert bias for next computation
        layer4 = layer3@brain[3].T # MatMult of layer3 and weights
        layer4 = sigmoid(layer4) # take sigmoid of output
        output = layer4 # define final output
        
    #output = np.round(output/np.sum(output)*100,0) # normalize output    
    return(output)