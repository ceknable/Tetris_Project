"""
Created on Mon Nov 30 15:38 2024

@author: KingRamenXIV
"""
# Import Libraries
import numpy as np
import random

# Tetris Library
import gymnasium as gym
from tetris_gymnasium.envs import Tetris 
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

# Import Functions
import NeuralNet as NN
'''Basic Function descriptions (for more info look at doc strings)
Functions:
    initialize_board: starts the game of tetris
    tetris: plays the game of Tetris with the neural network
    featurization: transform the board into useable info for the NN and GA'''
    
def initialize_board():
    ''' 
    Function that initializes the tetris board. 

    Returns: 
        gameboard: The initilized board 
        env: The Tetris env to be played in
    
    '''
    # Create an instance of Tetris
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    #env.reset(seed=42) # Fixed Seed
    i_seed = random.randint(0,4000000000) #42
    #print(i_seed)
    env.reset(seed = i_seed)
    
    terminated = False
    # Move the first block down one cell
    action = env.unwrapped.actions.move_down
    observation, reward, terminated, truncated, info = env.step(action)

    # Get flattened board to feed to TETRIS
    gameboard = featurization(observation, env)[0]
    
    return gameboard, env, i_seed

def tetris(brain, gameboard, env, turns):
    '''
    Takes a brain and uses neural network to play Tetris.
    Inputs: 
      brain: The NN that takes in the current board state and outputs an action.
      gameboard: Initial game board with the first block moved down. Flattened array of 1 and 0. 
      env: the environment of tetris obtained from initialize board
    Other: 
      TInput: A 1x6 array where all but one entries are 0. The non-zero position defines the action to take. 
               TInput = [Left,Right,SDown,Rotate,Down,Swap] (DISABLED: Down,Swap)
    Return: 
      gameboard: The final gameboard at game over
      height_hist: The history of the column heights over every step of the game
      holes_hist: The history of the number of holes in each column over every step of the game. 
      
    '''

    #Initialize total score
    total_score = 0
    height_hist = []
    # # Main game 
    terminated = False
    count = 0
    start_clip = 0
    
    while not terminated and count < turns:
        # Render the current state of the game as text
        env.render()
        
        # Reset parameters 
        action = None
        
        while action == None:

            
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
            
            #print(gameboard)
            #print(action_prob)
            #print(action)
        
        # Perform the action
        observation, reward, terminated, truncated, info = env.step(action)
        count += 1
        
        #Update the score:
        total_score += reward
        # Get the flattened board for this step, height and holes histroy for the game. 
        
        gameboard, height_i, holes_hist = featurization(observation, env)
        env1 = FeatureVectorObservation(env)
        holes_hist = env1.calc_holes(env.unwrapped.board)
        height_hist.append(height_i)
    # Game over
    #print("Game Over!")
    #print('Move')
    #print(total_score)
    return gameboard, height_hist, holes_hist, total_score

def featurization(observation, env):
  ''' Function that takes the observation from the game and the game environment
  to featurize for NN input and fitness evaluation.
  Inputs:
      observation: obtained from Tetris function
      env: the Tetris environment obtained from Tetris function
  Output:
      gameboard: 200 length array of 1s and 0s representing the gameboard of Tetris
      height_i: the height of the board at that moment
      holes_i: number of holes at that moment'''

  ## FLATTEN THE BOARD INTO A LIST OF 1S AND 0S
  # Render active tetromino (because it's not on self.board)
  projection = env.unwrapped.project_tetromino()
  # Crop padding away as we don't want to render it
  projection = env.unwrapped.crop_padding(projection)
  
  #flatten board array into a one dimensional list
  gameboard = projection.flatten()
  #convert blocks with a tetromnio in them into 1
  gameboard[np.flatnonzero(gameboard)] = 1

  ## HEIGHT OF EACH COLUMN AFTER EACH FRAME. Every rows is a frame, every column is a column in the game. Last row is endgame frame.
  #print the height after each frame --> the board doesn't include the piece (the piece is a projection onto the board)
  env1 = FeatureVectorObservation(env) #create an instance of the class
  height_i = env1.calc_height(env.unwrapped.board) #call a function from the class with the board as an input
  height_i = (height_i[4:14])-4 #crop out the padding on either side (the bedrock) and normalize by the 4 layers of bedrock underneath
  

  ## HOLES OF BOARD AFTER EVERY FRAME
  env1 = FeatureVectorObservation(env) #create an instance of the class
  #Calculate the holes after each frame
  holes_i = env1.calc_holes(env.unwrapped.board) # Returns the number of cells that are empty and have a filled cell above it i.e. only want the holes
  
  return gameboard, height_i, holes_i

