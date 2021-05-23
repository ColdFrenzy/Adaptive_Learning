# -*- coding: utf-8 -*-

class Config:
    # =============================================================================
    # LSTM PARAMETERS
    # =============================================================================
    LSTM_HIDDEN = [32,20,15,10] #[80,32,20,15]
    BATCH_SIZE = 32
    # sequence len for the lstm 
    LSTM_TIMESTEPS = 3
    FEATURES_LEN = 21 # Connect 3 Width X Heigh + 1(action) 
    FEATURES_LEN_2 = 22 # Connect3 Width X Heigh + 1 action + 1 outcome
    OUTPUT_LEN = 3  # 3 #depth1, depth 4, detph 6
    OUTCOME_AS_FEATURE = False
    
    # =============================================================================
    # EVALUATION PARAMETERS
    # =============================================================================
    NUMBER_OF_EVALUATION_GAMES = 100
    # how many games the player should play before we can evaluate it
    NUMBER_OF_GAMES_TO_TEST = [1,2,3]#,4,5] 
    # minimax depth that we are going to consider as class output 
    DEPTH_LIST = [1,4,6]#[1,4,6]