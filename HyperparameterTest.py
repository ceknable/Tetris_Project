# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:44:16 2024

@author: cekna
"""

import GeneticAlgorithm as GA
import matplotlib.pyplot as plt
import pickle
import numpy as np

'''This script runs a sensitivity analysis on the genetic algorithm'''

# names of variables to be tested
varnames = ['Population', 'mutation_rate_0','mutation_rate_14','bounds_mag', 'bounds_sign','neurons']
# ranges tested for each variable
varrange = [[5,120],[0,0.25],[0,0.4],[-2,2],[-2,2],[10,200]]
n = 3 # number of tests per variable (i.e. how many numbers in your range)
gen_pop = 400 # number of generations for the population test
gen = 400 # number of generations for everything else

# loop through each variable
for i in range(0,6):
    # make logspace if looking at weight bounds
    if i == 3 or i == 4:
        span = np.logspace(varrange[i][0],varrange[i][1], n)
    else: # make linspace
        span = np.linspace(varrange[i][0],varrange[i][1], n)
    
    name = varnames[i]
    
    # loop through each value to be tested in the analysis for that variable
    for j in range(len(span)):

        # Default Parameters
        population_size = 16
        if i == 0:
            generations = gen_pop
        else:
            generations = gen
        mutation_rate_0 = 0.05
        mutation_rate_14 = 0.1
        low_bound = -10
        up_bound = 10
        low_N = 4
        up_N = 30
        
        # Variable parameters (based on test)
        if i == 0:
            population_size = int(span[j])
        elif i == 1:
            mutation_rate_0 = span[j]
        elif i == 2:
            mutation_rate14 = span[j]
        elif i == 3:
            low_bound = -span[j]
            up_bound = span[j]
        elif i == 4:
            low_bound = 0.01
            up_bound = span[j]
        elif i ==5:
            low_N = int(span[j]*0.2)
            up_N = int(span[j])
        
        # Run the algorithm
        best_brain, best_fitness, genchamps = GA.genetic_algorithm(population_size, generations, mutation_rate_0, mutation_rate_14, low_bound, up_bound, low_N, up_N)
        
        # Plot results
        fitness_hist = []
        for i in range(len(genchamps)):
            fitness_hist.append(genchamps[i][1])
        numj = np.round(span[j],2)
        labelj = 'Var = '+str(numj)
        plt.plot(fitness_hist, label = labelj )
        plt.title(name+': Fitness of Best Player')
        plt.legend()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        
        # Save genchamps as pickle file
        with open('!'+name+str(j)+'.pkl', 'wb') as f:
            pickle.dump(genchamps, f)
    plt.show()
    # Save plots
    plt.savefig(name+'.jpeg', format='jpeg', dpi=300)



