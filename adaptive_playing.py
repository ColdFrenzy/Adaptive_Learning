# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(1, os.path.abspath(os.path.curdir))

import numpy as np

from models import Connect4ActionMaskModel
from config.connect4_config import Connect3Config
from utils.learning_behaviour_utils import LSTM_model,split_train_val,\
    minimax_vs_minimax_connect3_single_game,return_one_hot,\
    minimax_vs_minimax_connect3_single_game_plus_outcome,count_elem_in_dataset
from config.learning_behaviour_config import Config
from config.custom_config import Config as C
from env.connect4_multiagent_env import Connect4Env
from tensorflow import keras
from numpy import random
from ray.rllib.agents.ppo import PPOTrainer
from config.trainer_config import TrainerConfig




if __name__ == "__main__":
    _ = Connect4ActionMaskModel
    data_dir = C.DATA_DIR
    lstm_hidden = Config.LSTM_HIDDEN[-2] # best weights
    batch_size = Config.BATCH_SIZE
    lstm_timesteps = Config.LSTM_TIMESTEPS
    outcome_as_feature = Config.OUTCOME_AS_FEATURE
    if outcome_as_feature:
        features_len = Config.FEATURES_LEN_2
    else:
        features_len = Config.FEATURES_LEN
    output_len = Config.OUTPUT_LEN
    best_weights_npy = os.path.join(data_dir,"lstm_best_weights.npy")
    lstm_weights = np.load(best_weights_npy,allow_pickle=True)
    number_of_evaluation_games = Config.NUMBER_OF_EVALUATION_GAMES #  100 
    number_of_games_to_test = Config.NUMBER_OF_GAMES_TO_TEST #[1,2,3,4,5]
    depth_list = Config.DEPTH_LIST # [1,4,6]
    number_of_stochastic_moves = 6
    sequence_len = lstm_timesteps
    
    npy_weights_file = os.path.join(data_dir,"weights.npy")
    weights = np.load(npy_weights_file,allow_pickle=True)[()]  
    
    play = True
    
    trainer_obj = PPOTrainer(
        config=TrainerConfig.PPO_TRAINER_CONNECT3,
    )
    model = trainer_obj.get_policy("player1").model

    # =============================================================================
    # TEST THE MODEL 
    # =============================================================================
    import tensorflow as tf 
    lstm_model = LSTM_model(batch_size,(lstm_timesteps,features_len),output_len,lstm_hidden,False)  
    
    # generate a fake input to define the model stucture and then load the weights 
    # [batch,timestep,features]
    # random_input = np.random.rand(1,lstm_timesteps,features_len)
    random_input = np.random.rand(1,lstm_timesteps,features_len)
    random_input = random_input.astype('float32')
    lstm_model(random_input)
    lstm_model.set_weights(lstm_weights[()])
    
    


    # =============================================================================
    # SETTINGS
    # =============================================================================
    randomize = True
    player1_ID = Connect3Config.PLAYER1_ID
    player2_ID = Connect3Config.PLAYER2_ID
    player1 = Connect3Config.PLAYER1
    player2 = Connect3Config.PLAYER2
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )
    
    possible_answ = ["y","n"]
    
    def indx_to_lvl(indx):
        if indx < 4:
            return 1
        elif indx < 8:
            return 4
        elif indx <12: 
            return 6
        
    def lvl_to_indx(lvl):
        if lvl == 0:
            return random.choice(range(4))
        elif lvl == 1:
            return random.choice(range(4,8))
        elif lvl == 2:
            return random.choice(range(8,12))
        
    
    number_of_games = 0
    while play:    

        if number_of_games == 0:
            w2_indx = np.random.choice(range(len(weights)))
            w2_key = list(weights.keys())[w2_indx]
            w2 = weights[w2_key]
            lvl = indx_to_lvl(w2_indx)
            # lvl = weights.keys().index(w2)
        else:
            w2_indx = lvl_to_indx(int(predicted_indx))
            w2_key = list(weights.keys())[w2_indx]
            w2 = weights[w2_key]
            lvl = indx_to_lvl(w2_indx)
            
        print("You are facing the opponent of level " + str(lvl))
        full_game = []
        timestep = 0
        game_over = False
        actions = {}
        if randomize:
            starting_player = random.choice([player1_ID, player2_ID])
        else:
            starting_player = player1_ID
        game.reset(starting_player=starting_player,randomize=False)
        
        board_plus_action_total = []
        while not game_over:
            if timestep == 0:
                print(game)
            timestep += 1
            actual_player = game.current_player
            board = game.board
            board_p2 = game.board_p2

            #game.render()
            
            if actual_player == player1_ID:
                action_mask = game.get_moves(False) 
                choosing_action = True
                act = None
                action_mask_1 = [x+1 for x in action_mask]
                while choosing_action:

                    act = input("select an action:")
                    act = int(act)

                    if act in action_mask_1:
                        choosing_action = False
                    
                act = act-1
                flattened_board = np.ndarray.flatten(board)
                board_plus_actions = np.append(flattened_board,float(act))
                board_plus_action_total.append([board_plus_actions])
                
                actions[player1] = act                
                _, rew, done, _ = game.step(actions)
                print(game)  
                # game.render()
                           
                
                
            elif actual_player == player2_ID:
                input_dict = {"obs": {}}
                action_mask = game.get_moves(True)
                input_dict["obs"]["state"] = board_p2 #reshaped_board
                input_dict["obs"]["action_mask"] = action_mask
                action_logits, _ = model.forward(input_dict, None, None)
                if timestep > number_of_stochastic_moves:
                    act = np.argmax(action_logits[0])
                elif timestep <= number_of_stochastic_moves:
                    action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                    act = np.random.choice([0,1,2,3,4],1,p=action_prob)[0]  
                    
                actions[player2] = act
                _, rew, done, _ = game.step(actions)
                print(game)
                #game.render()
                
            else:
                raise ValueError("Player index is not valid, should be 0 or 1")

            if done["__all__"]:

                # ADD ENCODED GAME TO THE LISt
                if rew[player1] == 1:
                    print("Player 1 won!!!")
                elif rew[player1] == -1:
                    print("Player 2 won!!!")
                elif rew[player1] == 0:
                    print("Draw")
                if len(board_plus_action_total) < sequence_len:
                    print("Game finished too early, restarting...")
                    timestep = 0
                    game.reset(randomize=True)
                    game_over = False
                    continue
                
                game_over = True
                board_plus_action_and_outcome = board_plus_action_total
                
                for j in range(len(board_plus_action_and_outcome)-(sequence_len-1)):
                    full_game.append([])
                    full_game[-1].append(board_plus_action_and_outcome[j:j+sequence_len])
                        


            
        full_game = np.asarray(full_game)
        full_game = np.squeeze(full_game)
        if len(full_game.shape) == 2:
            full_game = np.expand_dims(full_game, axis=0)
        full_game = full_game.astype("float32")
        y = lstm_model(full_game,training=False)
                        
        predicted_values = tf.math.reduce_mean(y,axis=0)
        predicted_indx = tf.math.argmax(predicted_values) 
        
        answ = None
        waiting_answ = True
        while waiting_answ:
            answ = input("Do you want to play another game?\n")
            if answ in possible_answ:
                waiting_answ = False
            else:
                print("command unknown please only print y or n")
        if answ == "y":
            play = True
            number_of_games += 1
        elif answ == "n":
            play = False
            print("Thanks you for playing")
            

 

    
