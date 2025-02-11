# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:54:29 2024

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

import numpy as np

def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

neuron_trend = False
name = 'Stepwise1' 
with open(name+'.pkl', 'rb') as f:
    genchamps = pickle.load(f)

if neuron_trend:
    fitnesses = []
    neurons1 = []
    neurons2 = []
    neurons3 = []
    neuronsum = []
    for i in range(len(genchamps)):
        #print(genchamps[i][0])
        fitnesses.append(genchamps[i][1])
        neurons1.append(genchamps[i][0][0][1])
        neurons2.append(genchamps[i][0][0][3])
        neurons3.append(genchamps[i][0][0][5])
        neuronsum.append(neurons1[i]+neurons2[i]+neurons3[i])
    r2 = r_squared(neurons3,fitnesses)
    label1 = [np.corrcoef(neurons3,fitnesses), r2]
    plt.scatter(neurons3,fitnesses, label = label1, s = 2)
    plt.legend()
    plt.show()

layer1 = []
layer2 = []
layer3 = []
data = []
avg = []
for k in range(len(genchamps)):
    layer1.append(genchamps[k][0][0][1]) 
    layer2.append(genchamps[k][0][0][3])
    layer3.append(genchamps[k][0][0][5])
    data.append(genchamps[k][1])
    avg.append(genchamps[k][3])
data1 = moving_average(layer1,2)
data2 = moving_average(layer2,2)
data3 = moving_average(layer3,2)
avgMA = moving_average(avg,3)

plt.plot(data1, label = 'Layer1')
plt.plot(data2, label = 'Layer2')
plt.plot(data3, label = 'Layer3')
plt.title(name)
plt.legend()
plt.xlabel('Generations (Moving Average)')
plt.ylabel('Number of Neurons')
plt.show()

plt.plot(data, label = 'Raw Data')
plt.plot(avg, label = 'Average Pop Fitness')
plt.plot(avgMA, label = 'Moving Average')

name = name+': Fitness'
plt.title(name)
plt.legend()
plt.xlabel('Generations (Moving Average)')
plt.ylabel('Fitness')

plt.show()