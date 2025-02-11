# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 00:27:37 2024

@author: cekna
"""
# Import Libraries
import numpy as np
import sys
import cv2
import time
import os
import random
import pickle

# Tetris Library
import gymnasium as gym
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.envs.tetris import TetrisState
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

# Import Files
import NeuralNet as NN
import Tetris as T

'''This script takes a given test and will replay a selected game in order to 
visualize results from the GA.'''

def tetris_replay(brain,i_seed):
    '''
    Inputs: 
      brain: The NN that takes in the current board state and outputs an action.
      i_seed: seed that the individual played on 
    
    Other: 
      TIInput: A 1x6 array where all but one entries are 0. The non-zero position defines the action to take. 
               TInput = [Left,Right,Down,Rotate,SDown,Swap]
    
    Return: 
      total_score: the final score of the game
      holes_hist: The number of holes at that frame 
     
      
    '''
    # Initialize game
    if __name__ == "__main__":
        # Create an instance of Tetris
        env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
        env.reset(seed = i_seed)

        start_clip =0
        
        # Main game loop
        #Initialize total score
        total_score = 0

        # # Main game loop
        terminated = False

        while not terminated:
            # Render the current state of the game as text
            env.render()

            # Render active tetromino (because it's not on self.board)
            projection = env.unwrapped.project_tetromino()

            # Crop padding away as we don't want to render it
            projection = env.unwrapped.crop_padding(projection)
            gameboard = projection.flatten()
             #convert blocks with a tetromnio in them into 1
            gameboard[np.flatnonzero(gameboard)] = 1
             # Pick an action from user input mapped to the keyboard
            action = None
            while action is None:
                # wait 1 ms to see if key is pressed
                key = cv2.waitKey(1)
                # check if key press = a to start
                if start_clip==0:
                    if key == ord("a"):
                        action = env.unwrapped.actions.move_down
                        start_clip=1
                else:
                    # Get action input from brain
                    action_prob = NN.player(gameboard, brain) # Output from NEURAL NETWORK
                    TInput = np.zeros(6)
                    TInput[np.argmax(action_prob)] = 1 # 1x6 array. Only one element will be 1. Everything else is 0. 

                    if TInput[0]:
                        action = env.unwrapped.actions.move_left
                    elif TInput[1]:
                        action = env.unwrapped.actions.move_right
                    elif TInput[2]:
                        action = env.unwrapped.actions.hard_drop
                    elif TInput[3]:
                        action = env.unwrapped.actions.rotate_clockwise
                    elif TInput[4]:
                        action = env.unwrapped.actions.move_down
                    elif TInput[5]:
                        action = env.unwrapped.actions.swap
                
                if (
                    cv2.getWindowProperty(env.unwrapped.window_name, cv2.WND_PROP_VISIBLE)
                    == 0
                ):
                    sys.exit()
            time.sleep(.08) # time to wait before seeing next move 
            # Perform the action
            observation, reward, terminated, truncated, info = env.step(action)

            #Update the score:
            total_score += reward
            env1 = FeatureVectorObservation(env) #create an instance of the class
            #Calculate the holes after each frame
            holes_hist = env1.calc_holes(env.unwrapped.board) 
    
    return holes_hist, total_score

# Input name of pickle file (no extension)
name = 'interrupt'
with open(name+'.pkl', 'rb') as f:
    genchamps = pickle.load(f)
    #DNA, fitness, seed

# Create fitness history array
fitness = []
for k in range(len(genchamps)):
    fitness.append(genchamps[k][1])

# Find individual with best fitness
index = np.argmax(fitness)#random.randint(0,len(fitness))
#index = len(genchamps)-10
#index = 0
i_seed = genchamps[index][2] #don't include 

#Create the brain from the DNA
brain = NN.create(genchamps[index][0])

# Run game and record fitness
print('Open popup window and press a to start playback!')
#i_seed = random.randint(0,4000000)
hol_i , sco_i = tetris_replay(brain, int(i_seed[1]))
fitness = (sco_i)*3 - hol_i

# Check if fitness is correct to make sure it is playing correctly
print(i_seed)
print('Actual Fitness:',fitness)
print('Recorded Fitness:',genchamps[index][1])