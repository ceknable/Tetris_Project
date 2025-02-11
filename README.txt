README

Current as of 02/11/2025

This is the collection of scripts needed to run a Genetic Algorithm to optimize a Neural Network to play Tetris.
All files must be in the same folder.
All files were run in Spyder from the Anaconda distribution.
Authors: Carl Knable, Matthew LaCapra, Abdulrahman Atassi, and Evan Wood

If you have not installed them, install:
pip install opencv_contrib_python --user
pip install gymnasium
pip install tetris_gymnasium

Files and 
    functions

Run: Executes the GA

GeneticAlgorithm: Contains functions to run the GA
    Rand_CRs:                   Makes random chromosomes
    create_intial_population:   creates the initial population
    simulate_brain:             simulates the game of Tetris using the Neural Network
    select_parents:             picks the best of the population
    crossover:                  crosses genetic material of two parents
    mutate14:                   mutates chromosomes 1-4
    mutate0:                    mutates chromosome 0
    mutate:                     mutates DNA
    genetic_algorithm:          runs the GA

NeuralNet: Contains functions to run the NN
    player: plays the game by taking the input of the game and choosing a move
    create: creates the weight matrices for the NN that player takes as inputs
    sigmoid: used in the function player to transform the feature matrices

Tetris: Contains functions to run Tetris in python
    initialize_board: starts the game of tetris
    tetris: plays the game of Tetris with the neural network
    featurization: transform the board into useable info for the NN and GA

Replay: Script to visualize a Tetris game played by the NN

HyperparameterTest: Script to run a sensitivity analysis

SensitivityPlot: Script to plot moving averages of the sensitivity analysis results (fitness and number of neurons)

SingleRunNNPlot: Plots the number of neurons over generations for a single test