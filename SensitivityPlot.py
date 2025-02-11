# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:25:40 2024

@author: cekna
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    """
    Source: ChatGPT
    Calculate the moving average of a given list of numbers.
    
    Input:
    - data: list or numpy array of numbers
    - window_size: the size of the moving window
    
    Returns:
    - A list of moving averages.
    """
    if window_size <= 0:
        raise ValueError("Window size must be greater than 0")
    if window_size > len(data):
        raise ValueError("Window size must be smaller or equal to the length of data")
    
    # Using numpy to compute the moving average
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


varnames = ['Population', 'mutation_rate_0','mutation_rate_14','bounds_mag', 'bounds_sign','neurons']
varrange = [[5,120],[0,0.25],[0,0.4],[-2,2],[-2,2],[10,200]]

Fitness = False
Neurons = True
NeuronMovingAverage = True

for i in range(len(varnames)):
    
    if i == 3 or i == 4:
        span = np.logspace(varrange[i][0],varrange[i][1], 3)
    else:
        span = np.linspace(varrange[i][0],varrange[i][1], 3)
    
    name = varnames[i]
    
    if Fitness:
        # Load the object from the file
        for j in range(3):
            name = '!'+varnames[i]+str(j) 
            with open(name+'.pkl', 'rb') as f:
                genchamps = pickle.load(f)
            fitness = []
            for k in range(len(genchamps)):
                fitness.append(genchamps[k][1])
            data = moving_average(fitness,200)
            
            numj = np.round(span[j],2)
            labelj = '# = '+str(numj)
            
            plt.plot(data, label = labelj)
            
            name = varnames[i]+': Best Fitness(Moving Average)'
            plt.title(name)
            plt.legend()
            plt.xlabel('Generations (Moving Average)')
            plt.ylabel('Fitness')
            
        plt.show()
        
    if Neurons:
        # Load the object from the file
        for j in range(10):
            name = '!'+varnames[i]+str(j) 
            with open(name+'.pkl', 'rb') as f:
                genchamps = pickle.load(f)
            layer1 = []
            layer2 = []
            layer3 = []
            for k in range(len(genchamps)):
                layer1.append(genchamps[k][0][0][1])
                layer2.append(genchamps[k][0][0][3])
                layer3.append(genchamps[k][0][0][5])
            if NeuronMovingAverage:
                data1 = moving_average(layer1,500)
                data2 = moving_average(layer2,500)
                data3 = moving_average(layer3,500)
            else:
                data1 = layer1
                data2 = layer2
                data3 = layer3
            numj = np.round(span[j],2)
            labelj = '# = '+str(numj)
            
            plt.plot(data1, label = 'Layer1')
            plt.plot(data2, label = 'Layer2')
            plt.plot(data3, label = 'Layer3')
            
            
            name = varnames[i]+' = '+str(numj)+': Best Fitness(Moving Average) '
            plt.title(name+str(j))
            plt.legend()
            plt.xlabel('Generations (Moving Average)')
            plt.ylabel('Number of Neurons')
            
            plt.show()
        