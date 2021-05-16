# -*- coding: utf-8 -*-
import sys
import os
import random
import numpy as np
import logging
import math
from policies.minimax_policy import minimax
from policies.minimax_connect3 import minimax_connect3


sys.path.insert(1, os.path.abspath(os.pardir))

from config.custom_config import Config
from config.connect4_config import Connect4Config,Connect3Config
from env.connect4_multiagent_env import Connect4Env
from utils.pre_compute_elo import board_print


player1 = Connect4Config.PLAYER1
player2 = Connect4Config.PLAYER2
player1_ID = Connect4Config.PLAYER1_ID
player2_ID = Connect4Config.PLAYER2_ID

from tensorflow import keras
from tensorflow.keras import layers

def LSTM_model(batch_size, input_shape,output_shape,lstm_hidden,show_summary):
    model = keras.Sequential()
    # model.add(keras.Input(input_shape,batch_size=batch_size))
    # time major == True => [timestep,batch,features]
    # time_major == False => [batch,timestep,features]
    model.add(layers.LSTM(lstm_hidden,time_major=False))
    # 3 classes (different depths)
    model.add(layers.Dense(output_shape))
    model.add(layers.Softmax())
    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.CosineSimilarity(),
        #loss=keras.losses.CategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    if show_summary:
        model.summary()
    return model


def model_vs_model_connect3_generate_data(model1,model2,weights, number_of_games,sequence_len,randomize=True,number_of_stochastic_moves=0,logger=None):
    """ 
    generates game data from games with the board seen by model1 pov

    """
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )

    dataset_no_clones = []
    dataset = []
    # game encoded as a string 
    games_list = []
    number_of_equal_games = 0
 
    # sequence of configuration encoded as a string 
    sequence_list = []
    number_of_equal_sequences = 0
    
    total_sequence_list = {}
    total_games_list = {}
    
    label = 0
    # Tournament between agents with different weights
    for w1_indx,w1 in enumerate(weights):
        model1.base_model.set_weights(weights[w1])
        if logger:
            logger.info("STARTING WITH WEIGHTS: " + w1)
        
        # before updating the label, we check if we have a new one, in that case
        # we reset both 
        new_label = int(w1.split("_")[0][-1])
        if new_label != label:
            if label != 0:
                total_games_list[label] = games_list
                total_sequence_list[label] = sequence_list
                games_list = [] 
                sequence_list = []
    
        label = int(w1.split("_")[0][-1])
        for w2_indx,w2 in enumerate(weights): 
            # DELETE
            # if w2_indx > 3:
            #     continue
            # if w2_indx < w1_indx:
            #     continue
            print("Starting weights " + str(w1) + " vs weights " + str(w2))
            model2.base_model.set_weights(weights[w2])

            
            for i in range(number_of_games):
                # temp action and board state, 
                #action_total = []
                board_plus_action_total = []
                
                timestep = 0
                game_over = False
                actions = {}
                encoded_game = []
                if randomize:
                    starting_player = random.choice([player1_ID, player2_ID])
                    if starting_player == player1_ID:
                        encoded_game.append("p1_")
                    elif starting_player == player2_ID:
                        encoded_game.append("p2_")
                else:
                    starting_player = player1_ID
                    encoded_game.append("p1_")
                game.reset(starting_player=starting_player,randomize=False)
                
                
                while not game_over:
                    timestep += 1
                    actual_player = game.current_player
                    board = game.board
                    board_p2 = game.board_p2
                    
                    
                    if actual_player == player1_ID:
                        input_dict = {"obs": {}}
                        action_mask = game.get_moves(True)
                        input_dict["obs"]["state"] = board
                        input_dict["obs"]["action_mask"] = action_mask
                        action_logits, _ = model1.forward(input_dict, None, None)
                        if timestep > number_of_stochastic_moves:
                            act = np.argmax(action_logits[0])
                        elif timestep <= number_of_stochastic_moves:
                            action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                            act = np.random.choice([0,1,2,3,4],1,p=action_prob)[0]  
                        
                        encoded_game.append(str(act))

                        flattened_board = np.ndarray.flatten(board)
                        board_plus_actions = np.append(flattened_board,float(act))
                        board_plus_action_total.append([board_plus_actions])
                        
                        actions[player1] = act                
                        _, rew, done, _ = game.step(actions)
                        
                        
                        
                        
                    elif actual_player == player2_ID:
                        input_dict = {"obs": {}}
                        action_mask = game.get_moves(True)
                        input_dict["obs"]["state"] = board_p2 #reshaped_board
                        input_dict["obs"]["action_mask"] = action_mask
                        action_logits, _ = model2.forward(input_dict, None, None)
                        if timestep > number_of_stochastic_moves:
                            act = np.argmax(action_logits[0])
                        elif timestep <= number_of_stochastic_moves:
                            action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                            act = np.random.choice([0,1,2,3,4],1,p=action_prob)[0]  
                            
                        encoded_game.append(str(act))
                        actions[player2] = act
                        _, rew, done, _ = game.step(actions)
                        
                    else:
                        raise ValueError("Player index is not valid, should be 0 or 1")
                        
                    if done["__all__"]:
                        game_over = True
                        game_str = ''.join(encoded_game)
                        # ADD ENCODED GAME TO THE LISt
                        if game_str in games_list:
                            number_of_equal_games += 1
                        elif game_str not in games_list:
                            games_list.append(game_str)
                            # if the game is too short, just discard it
                            if len(board_plus_action_total) < sequence_len:
                                continue
                            
                            for j in range(len(board_plus_action_total)-(sequence_len-1)):
                                to_string=str(board_plus_action_total[j:j+sequence_len])
                                if to_string in sequence_list:
                                    number_of_equal_sequences += 1
                                else:
                                    sequence_list.append(to_string)
                                    dataset_no_clones.append([])
                                    dataset_no_clones[-1].append(board_plus_action_total[j:j+sequence_len])
                                    dataset_no_clones[-1].append(label)
                                    
                                dataset.append([])
                                dataset[-1].append(board_plus_action_total[j:j+sequence_len])
                                dataset[-1].append(label)
     
    total_games_list[label] = games_list
    total_sequence_list[label] = sequence_list
    
    print("The number of equal games is: " + str(number_of_equal_games))
    print("The number of equal sequences is: " + str(number_of_equal_sequences))


        
    return total_sequence_list, total_games_list, dataset,dataset_no_clones


def model_vs_model_connect3_generate_data_v2(model1,model2,weights, number_of_games,sequence_len,randomize=True,number_of_stochastic_moves=0,logger=None):
    """ 
    generates game data from games with the board seen by model1 pov

    """
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )

    dataset_no_clones = []
    dataset = []
    # game encoded as a string 
    games_list = []
    number_of_equal_games = 0

    lv1_skipped = 0    
    lv4_skipped = 0    
    lv6_skipped = 0    
    

    # sequence of configuration encoded as a string 
    sequence_list = []
    number_of_equal_sequences = 0
    
    
    # Tournament between agents with different weights
    for w1_indx,w1 in enumerate(weights):
        model1.base_model.set_weights(weights[w1])
        if logger:
            logger.info("STARTING WITH WEIGHTS: " + w1)
            
        # DELETE 
        # if w1_indx > 2:
        #     break
        label = int(w1.split("_")[0][-1])
        for w2_indx,w2 in enumerate(weights): 
            # DELETE
            # if w2_indx > 3:
            #     continue
            # if w2_indx < w1_indx:
            #     continue
            print("Starting weights " + str(w1) + " vs weights " + str(w2))
            model2.base_model.set_weights(weights[w2])

            
            for i in range(number_of_games):
                # temp action and board state, 
                #action_total = []
                board_plus_action_total = []
                
                timestep = 0
                game_over = False
                actions = {}
                encoded_game = []
                if randomize:
                    starting_player = random.choice([player1_ID, player2_ID])
                    if starting_player == player1_ID:
                        encoded_game.append("p1_")
                    elif starting_player == player2_ID:
                        encoded_game.append("p2_")
                else:
                    starting_player = player1_ID
                    encoded_game.append("p1_")
                game.reset(starting_player=starting_player,randomize=False)
                
                
                while not game_over:
                    timestep += 1
                    actual_player = game.current_player
                    board = game.board
                    board_p2 = game.board_p2
                    
                    
                    if actual_player == player1_ID:
                        input_dict = {"obs": {}}
                        action_mask = game.get_moves(True)
                        input_dict["obs"]["state"] = board
                        input_dict["obs"]["action_mask"] = action_mask
                        action_logits, _ = model1.forward(input_dict, None, None)
                        if timestep > number_of_stochastic_moves:
                            act = np.argmax(action_logits[0])
                        elif timestep <= number_of_stochastic_moves:
                            action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                            act = np.random.choice([0,1,2,3,4],1,p=action_prob)[0]  
                        
                        encoded_game.append(str(act))

                        flattened_board = np.ndarray.flatten(board)
                        board_plus_actions = np.append(flattened_board,float(act))
                        board_plus_action_total.append([board_plus_actions])
                        
                        actions[player1] = act                
                        _, rew, done, _ = game.step(actions)
                        
                        
                        
                        
                    elif actual_player == player2_ID:
                        input_dict = {"obs": {}}
                        action_mask = game.get_moves(True)
                        input_dict["obs"]["state"] = board_p2 #reshaped_board
                        input_dict["obs"]["action_mask"] = action_mask
                        action_logits, _ = model2.forward(input_dict, None, None)
                        if timestep > number_of_stochastic_moves:
                            act = np.argmax(action_logits[0])
                        elif timestep <= number_of_stochastic_moves:
                            action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                            act = np.random.choice([0,1,2,3,4],1,p=action_prob)[0]  
                            
                        encoded_game.append(str(act))
                        actions[player2] = act
                        _, rew, done, _ = game.step(actions)
                        
                    else:
                        raise ValueError("Player index is not valid, should be 0 or 1")
                        
                    if done["__all__"]:
                        game_over = True
                        game_str = ''.join(encoded_game)
                        # ADD ENCODED GAME TO THE LISt
                        if game_str in games_list:
                            number_of_equal_games += 1
                        elif game_str not in games_list:
                            games_list.append(game_str)
                            # if the game is too short, just discard it
                            if len(board_plus_action_total) < sequence_len:
                                continue
                            
                            for j in range(len(board_plus_action_total)-(sequence_len-1)):
                                to_string=str(board_plus_action_total[j:j+sequence_len])
                                if to_string in sequence_list:
                                    number_of_equal_sequences += 1
                                    if label == 1:
                                        lv1_skipped += 1
                                    elif label == 4:
                                        lv4_skipped += 1
                                    elif label == 6:
                                        lv6_skipped += 1
                                else:
                                    sequence_list.append(to_string)
                                    dataset_no_clones.append([])
                                    dataset_no_clones[-1].append(board_plus_action_total[j:j+sequence_len])
                                    dataset_no_clones[-1].append(label)
                                    
                                dataset.append([])
                                dataset[-1].append(board_plus_action_total[j:j+sequence_len])
                                dataset[-1].append(label)
                        
    
    print("The number of equal games is: " + str(number_of_equal_games))
    print("The number of equal sequences is: " + str(number_of_equal_sequences))
    print("depth 1 skipped: " + str(lv1_skipped))
    print("depth 4 skipped: " + str(lv4_skipped))
    print("depth 6 skipped: " + str(lv6_skipped))

        
    return  games_list, sequence_list, dataset,dataset_no_clones



def model_vs_model_connect3_generate_data_v3(model1,model2,weights, number_of_games,sequence_len,randomize=True,number_of_stochastic_moves=0,logger=None):
    """ 
    In this version we also added the outcome of the game to the vector 
    of the sequence of moves
    

    """
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )

    dataset_no_clones = []
    dataset = []
    # game encoded as a string 
    games_list = []
    number_of_equal_games = 0
 
    # sequence of configuration encoded as a string 
    sequence_list = []
    number_of_equal_sequences = 0
    
    total_sequence_list = {}
    total_games_list = {}
    
    label = 0
    # Tournament between agents with different weights
    for w1_indx,w1 in enumerate(weights):
        model1.base_model.set_weights(weights[w1])
        if logger:
            logger.info("STARTING WITH WEIGHTS: " + w1)
        
        # before updating the label, we check if we have a new one, in that case
        # we reset both 
        new_label = int(w1.split("_")[0][-1])
        if new_label != label:
            if label != 0:
                total_games_list[label] = games_list
                total_sequence_list[label] = sequence_list
                games_list = [] 
                sequence_list = []
    
        label = int(w1.split("_")[0][-1])
        for w2_indx,w2 in enumerate(weights): 
            # DELETE
            # if w2_indx > 3:
            #     continue
            # if w2_indx < w1_indx:
            #     continue
            print("Starting weights " + str(w1) + " vs weights " + str(w2))
            model2.base_model.set_weights(weights[w2])

            
            for i in range(number_of_games):
                # temp action and board state, 
                #action_total = []
                board_plus_action_total = []
                
                timestep = 0
                game_over = False
                actions = {}
                encoded_game = []
                if randomize:
                    starting_player = random.choice([player1_ID, player2_ID])
                    if starting_player == player1_ID:
                        encoded_game.append("p1_")
                    elif starting_player == player2_ID:
                        encoded_game.append("p2_")
                else:
                    starting_player = player1_ID
                    encoded_game.append("p1_")
                game.reset(starting_player=starting_player,randomize=False)
                
                
                while not game_over:
                    timestep += 1
                    actual_player = game.current_player
                    board = game.board
                    board_p2 = game.board_p2
                    
                    
                    if actual_player == player1_ID:
                        input_dict = {"obs": {}}
                        action_mask = game.get_moves(True)
                        input_dict["obs"]["state"] = board
                        input_dict["obs"]["action_mask"] = action_mask
                        action_logits, _ = model1.forward(input_dict, None, None)
                        if timestep > number_of_stochastic_moves:
                            act = np.argmax(action_logits[0])
                        elif timestep <= number_of_stochastic_moves:
                            action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                            act = np.random.choice([0,1,2,3,4],1,p=action_prob)[0]  
                        
                        encoded_game.append(str(act))

                        flattened_board = np.ndarray.flatten(board)
                        board_plus_actions = np.append(flattened_board,float(act))
                        board_plus_action_total.append([board_plus_actions])
                        
                        actions[player1] = act                
                        _, rew, done, _ = game.step(actions)
                        
                        
                        
                        
                    elif actual_player == player2_ID:
                        input_dict = {"obs": {}}
                        action_mask = game.get_moves(True)
                        input_dict["obs"]["state"] = board_p2 #reshaped_board
                        input_dict["obs"]["action_mask"] = action_mask
                        action_logits, _ = model2.forward(input_dict, None, None)
                        if timestep > number_of_stochastic_moves:
                            act = np.argmax(action_logits[0])
                        elif timestep <= number_of_stochastic_moves:
                            action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                            act = np.random.choice([0,1,2,3,4],1,p=action_prob)[0]  
                            
                        encoded_game.append(str(act))
                        actions[player2] = act
                        _, rew, done, _ = game.step(actions)
                        
                    else:
                        raise ValueError("Player index is not valid, should be 0 or 1")
                        
                    if done["__all__"]:
                        # we use 7,8,9 as value for the outcome in order to
                        # not confuse them with the values of actions
                        if rew["player1"] == 1:
                            encoded_game.append("_7")
                            outcome = 7.0
                        elif rew["player1"] == -1:
                            encoded_game.append("_8")
                            outcome = 8.0
                        elif rew["player1"] == 0:
                            encoded_game.append("_9")
                            outcome = 9.0
                        game_over = True
                        game_str = ''.join(encoded_game)
                        # ADD ENCODED GAME TO THE LISt
                        if game_str in games_list:
                            number_of_equal_games += 1
                        elif game_str not in games_list:
                            games_list.append(game_str)
                            # if the game is too short, just discard it
                            if len(board_plus_action_total) < sequence_len:
                                continue
                            
                            board_plus_action_and_outcome = []
                            for elem in board_plus_action_total:
                                new_elem = np.append(elem,outcome)
                                board_plus_action_and_outcome.append([new_elem])
                            
                            for j in range(len(board_plus_action_and_outcome)-(sequence_len-1)):
                                to_string=str(board_plus_action_and_outcome[j:j+sequence_len])
                                if to_string in sequence_list:
                                    number_of_equal_sequences += 1
                                else:
                                    sequence_list.append(to_string)
                                    dataset_no_clones.append([])
                                    dataset_no_clones[-1].append(board_plus_action_and_outcome[j:j+sequence_len])
                                    dataset_no_clones[-1].append(label)
                                    
                                dataset.append([])
                                dataset[-1].append(board_plus_action_and_outcome[j:j+sequence_len])
                                dataset[-1].append(label)
     
    total_games_list[label] = games_list
    total_sequence_list[label] = sequence_list
    
    print("The number of equal games is: " + str(number_of_equal_games))
    print("The number of equal sequences is: " + str(number_of_equal_sequences))


        
    return total_sequence_list, total_games_list, dataset,dataset_no_clones


def model_vs_model_connect3_generate_data_v4(model1,model2,use_outcome,weights, number_of_games,sequence_len,randomize=True,number_of_stochastic_moves=0,logger=None):
    """ 
    In this version we also add the outcome of the game to the vector  
    

    """
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )

    dataset_no_clones = []
    dataset = []
    # game encoded as a string 
    games_list = []
    number_of_equal_games = 0
 
    # sequence of configuration encoded as a string 
    sequence_list = []
    number_of_equal_sequences = 0
    
    # index of data that are equals and we need to remove from dataset later
    indx_to_remove = []
    
    
    label = 0
    sequences_per_player = {}
    # Tournament between agents with different weights
    for w1_indx,w1 in enumerate(weights):
        model1.base_model.set_weights(weights[w1])
        if logger:
            logger.info("STARTING WITH WEIGHTS: " + w1)
        
    
        label = int(w1.split("_")[0][-1])
        if label not in sequences_per_player:
            sequences_per_player[label] = []
        
        for w2_indx,w2 in enumerate(weights): 
            print("Starting weights " + str(w1) + " vs weights " + str(w2))
            model2.base_model.set_weights(weights[w2])

            
            for i in range(number_of_games):
                # temp action and board state, 
                #action_total = []
                board_plus_action_total = []
                
                timestep = 0
                game_over = False
                actions = {}
                encoded_game = []
                if randomize:
                    starting_player = random.choice([player1_ID, player2_ID])
                    if starting_player == player1_ID:
                        encoded_game.append("p1_")
                    elif starting_player == player2_ID:
                        encoded_game.append("p2_")
                else:
                    starting_player = player1_ID
                    encoded_game.append("p1_")
                game.reset(starting_player=starting_player,randomize=False)
                
                
                while not game_over:
                    timestep += 1
                    actual_player = game.current_player
                    board = game.board
                    board_p2 = game.board_p2
                    
                    
                    if actual_player == player1_ID:
                        input_dict = {"obs": {}}
                        action_mask = game.get_moves(True)
                        input_dict["obs"]["state"] = board
                        input_dict["obs"]["action_mask"] = action_mask
                        action_logits, _ = model1.forward(input_dict, None, None)
                        if timestep > number_of_stochastic_moves:
                            act = np.argmax(action_logits[0])
                        elif timestep <= number_of_stochastic_moves:
                            action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                            act = np.random.choice([0,1,2,3,4],1,p=action_prob)[0]  
                        
                        encoded_game.append(str(act))

                        flattened_board = np.ndarray.flatten(board)
                        board_plus_actions = np.append(flattened_board,float(act))
                        board_plus_action_total.append([board_plus_actions])
                        
                        actions[player1] = act                
                        _, rew, done, _ = game.step(actions)
                        
                        
                        
                        
                    elif actual_player == player2_ID:
                        input_dict = {"obs": {}}
                        action_mask = game.get_moves(True)
                        input_dict["obs"]["state"] = board_p2 #reshaped_board
                        input_dict["obs"]["action_mask"] = action_mask
                        action_logits, _ = model2.forward(input_dict, None, None)
                        if timestep > number_of_stochastic_moves:
                            act = np.argmax(action_logits[0])
                        elif timestep <= number_of_stochastic_moves:
                            action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                            act = np.random.choice([0,1,2,3,4],1,p=action_prob)[0]  
                            
                        encoded_game.append(str(act))
                        actions[player2] = act
                        _, rew, done, _ = game.step(actions)
                        
                    else:
                        raise ValueError("Player index is not valid, should be 0 or 1")
                        
                    if done["__all__"]:
                        # we use 7,8,9 as value for the outcome in order to
                        # not confuse them with the values of actions
                        if use_outcome:
                            if rew["player1"] == 1:
                                encoded_game.append("_7")
                                outcome = 7.0
                            elif rew["player1"] == -1:
                                encoded_game.append("_8")
                                outcome = 8.0
                            elif rew["player1"] == 0:
                                encoded_game.append("_9")
                                outcome = 9.0
                        game_over = True
                        game_str = ''.join(encoded_game)
                        # ADD ENCODED GAME TO THE LISt
                        if game_str in games_list:
                            number_of_equal_games += 1
                        elif game_str not in games_list:
                            games_list.append(game_str)
                            # if the game is too short, just discard it
                            if len(board_plus_action_total) < sequence_len:
                                continue
                            if use_outcome:
                                board_plus_action_and_outcome = []
                                for elem in board_plus_action_total:
                                    new_elem = np.append(elem,outcome)
                                    board_plus_action_and_outcome.append([new_elem])
                            else:
                                board_plus_action_and_outcome = board_plus_action_total
                            
                            for j in range(len(board_plus_action_and_outcome)-(sequence_len-1)):
                                to_string=str(board_plus_action_and_outcome[j:j+sequence_len])
                                if to_string in sequence_list:
                                    indx = sequence_list.index(to_string)
                                    number_of_equal_sequences += 1
                                    if indx not in indx_to_remove:
                                        indx_to_remove.append(indx)
                                else:
                                    sequences_per_player[label].append(to_string)
                                    sequence_list.append(to_string)
                                    dataset_no_clones.append([])
                                    dataset_no_clones[-1].append(board_plus_action_and_outcome[j:j+sequence_len])
                                    dataset_no_clones[-1].append(label)
                                    
                                dataset.append([])
                                dataset[-1].append(board_plus_action_and_outcome[j:j+sequence_len])
                                dataset[-1].append(label)
                                
    final_dataset = []
    for i,elem in enumerate(dataset_no_clones):
        if i not in indx_to_remove:
            final_dataset.append(elem)
            
    count = 0
    final_sequences = {}
    for elem in sequences_per_player:
        final_sequences[elem] = []
        for indx,elem_2 in enumerate(sequences_per_player[elem]):
            if count not in indx_to_remove:
                final_sequences[elem].append(elem_2)
            count += 1
            
     

    print("Original dataset length " + str(len(dataset_no_clones)) + " without clones: "  + str(len(final_dataset)))
    print("The number of equal games is: " + str(number_of_equal_games))
    print("The number of equal sequences is: " + str(number_of_equal_sequences))


        
    return sequence_list, games_list, dataset,final_dataset,final_sequences


def split_train_val(loaded_data,depth_list,val_indx=0.15,shuffle = True):
    """
    val_indx is the percentage of elements in the validation set
    :params:
        loaded_data: list
             list of lists, where the first element are the encoded games
             and the second is the label 
    """
    shuffled_dataset = [ [elem[0],elem[1]] for elem in loaded_data]
    val_elems = int(len(shuffled_dataset)*val_indx)
    train_elems = len(shuffled_dataset) - val_elems
    unbalanced = True
    
    # shuffle the dataset
    if shuffle:
        # while unbalanced:
        np.random.shuffle(shuffled_dataset)
            
    
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    for i,elem in enumerate(shuffled_dataset):
        one_hot_vec = [0] * len(depth_list)
        if i < train_elems:
            x_train.append(elem[0])
            indx = depth_list.index(elem[1])
            one_hot_vec[indx] = 1
            y_train.append(one_hot_vec)
        else:
            x_val.append(elem[0])
            indx = depth_list.index(elem[1])
            one_hot_vec[indx] = 1
            y_val.append(one_hot_vec)
        
    
    return x_train,y_train,x_val,y_val




    
def minimax_vs_minimax_connect3_generate_data(depth_list,number_of_games,sequence_len,randomize= True,logger = None):
    """
        simulates 'number_of_games' between all combinations of minimax depth 
        inside the depth list. 
        :params:
            depth_list: list of int
                minimax depths
            number_of_games: int
            

    """
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )
    
    
    dataset_no_clones = []
    dataset = []
    dataset_full_games = []
    # game encoded as a string 
    games_list = []
    number_of_equal_games = 0

    # sequence of configuration encoded as a string 
    sequence_list = []
    number_of_equal_sequences = 0
    
    
    # index of data that are equals and we need to remove from dataset later
    indx_to_remove = []
    
    # we record here all the different unique sequences done by every single player 
    sequences_per_player = {}
    
    for depth1 in depth_list:
        label = int(depth1)
        if label not in sequences_per_player:
            sequences_per_player[label] = []
        for depth2 in depth_list:
            print("Starting minimax depth " + str(depth1)  + " vs minimax " + str(depth2))
            if logger:
                logger.info(
                    "**********MINIMAX_depth_"
                    + str(depth1)
                    + "_(X) VS (O)_MINIMAX_depth_"
                    + str(depth2)
                    + "**********"
                )
            
            for i in range(number_of_games):
                # temp action and board state, 
                #action_total = []
                board_plus_action_total = []
                game_over = False
                actions = {}
                encoded_game = []
                if randomize:
                    starting_player = random.choice([player1_ID, player2_ID])
                    if starting_player == player1_ID:
                        encoded_game.append("p1_")
                    elif starting_player == player2_ID:
                        encoded_game.append("p2_")
                else:
                    starting_player = player1_ID
                    encoded_game.append("p1_")
                game.reset(starting_player=starting_player,randomize=False)
 
                while not game_over:
                    actual_player = game.current_player
                    board = game.board
                    board_p2 = game.board_p2
                    
                    if actual_player == player1_ID:
                        act, _ = minimax_connect3(board, player1_ID, True, depth=depth1)
                        actions[player1] = act
                        
                        encoded_game.append(str(act))
                        flattened_board = np.ndarray.flatten(board)
                        board_plus_actions = np.append(flattened_board,float(act))
                        board_plus_action_total.append([board_plus_actions])
                        
                        _, _, done, _ = game.step(actions)
                        
                    elif actual_player == player2_ID:
                        act, _ = minimax_connect3(board, player2_ID, True, depth=depth2)
                        actions[player2] = act
                        _, _, done, _ = game.step(actions)
                    else:
                        raise ValueError("Player index is not valid, should be 0 or 1")
                    if logger:
                        logger.info("Game number " + str(i) + "/" + str(number_of_games))
                        logger.info(
                            "Player " + str(actual_player + 1) + " actions: " + str(act)
                        )
                        logger.info("\n" + repr(board))
                        logger.info(board_print(board,Connect3Config.HEIGHT,Connect3Config.WIDTH))
        
                    if done["__all__"]:
                        if logger:
                            logger.info("PLAYER " + str(game.winner + 1) + " WON...")
                            logger.info(
                                "CURRENT SCORE: "
                                + str(game.score[player1])
                                + " VS "
                                + str(game.score[player2])
                            )
                        game_over = True
                        game_str = ''.join(encoded_game)
                        # ADD ENCODED GAME TO THE LISt
                        if game_str in games_list:
                            number_of_equal_games += 1
                        elif game_str not in games_list:
                            games_list.append(game_str)
                            # if the game is too short, just discard it
                            if len(board_plus_action_total) < sequence_len:
                                continue
                            dataset_full_games.append([])
                            dataset_full_games[-1].append(board_plus_action_total)
                            dataset_full_games[-1].append(label)
                            
                            for j in range(len(board_plus_action_total)-(sequence_len-1)):
                                to_string=str(board_plus_action_total[j:j+sequence_len])
                                if to_string in sequence_list:
                                    indx = sequence_list.index(to_string)
                                    number_of_equal_sequences += 1
                                    if indx not in indx_to_remove:
                                        indx_to_remove.append(indx)

                                else:
                                    sequences_per_player[label].append(to_string)
                                    sequence_list.append(to_string)
                                    dataset_no_clones.append([])
                                    dataset_no_clones[-1].append(board_plus_action_total[j:j+sequence_len])
                                    dataset_no_clones[-1].append(label)
                                    

                                    
                                dataset.append([])
                                dataset[-1].append(board_plus_action_total[j:j+sequence_len])
                                dataset[-1].append(label)
                                
    final_dataset = []
    for i,elem in enumerate(dataset_no_clones):
        if i not in indx_to_remove:
            final_dataset.append(elem)
            
    count = 0
    final_sequences = {}
    for elem in sequences_per_player:
        final_sequences[elem] = []
        for indx,elem_2 in enumerate(sequences_per_player[elem]):
            if count not in indx_to_remove:
                final_sequences[elem].append(elem_2)
            count += 1
                        
    print("Original dataset length " + str(len(dataset_no_clones)) + " without clones: "  + str(len(final_dataset)))
    print("The number of equal games is: " + str(number_of_equal_games))
    print("The number of equal sequences is: " + str(number_of_equal_sequences))


    return  games_list, sequence_list, dataset,dataset_no_clones,final_dataset,dataset_full_games,final_sequences




def minimax_vs_model_connect3_generate_data(depth_list,weights,model,number_of_stochastic_moves,number_of_games,sequence_len,randomize= True,logger = None):
    """
        simulates 'number_of_games' between all combinations of minimax depth 
        inside the depth list. 
        :params:
            depth_list: list of int
                minimax depths
            number_of_games: int
            

    """
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )
    
    
    dataset_no_clones = []
    dataset = []
    dataset_full_games = []
    # game encoded as a string 
    games_list = []
    number_of_equal_games = 0

    # sequence of configuration encoded as a string 
    sequence_list = []
    number_of_equal_sequences = 0
    
    
    # index of data that are equals and we need to remove from dataset later
    indx_to_remove = []
    
    # we record here all the different unique sequences done by every single player 
    sequences_per_player = {}
    
    for depth1 in depth_list:
        label = int(depth1)
        if label not in sequences_per_player:
            sequences_per_player[label] = []
        for w2_indx,w2 in enumerate(weights): 
            print("Starting minimax depth_" + str(depth1) + " vs model with weights " + str(w2))
            model.base_model.set_weights(weights[w2]) 
            
            for i in range(number_of_games):
                # temp action and board state, 
                #action_total = []
                timestep = 0
                board_plus_action_total = []
                game_over = False
                actions = {}
                encoded_game = []
                if randomize:
                    starting_player = random.choice([player1_ID, player2_ID])
                    if starting_player == player1_ID:
                        encoded_game.append("p1_")
                    elif starting_player == player2_ID:
                        encoded_game.append("p2_")
                else:
                    starting_player = player1_ID
                    encoded_game.append("p1_")
                game.reset(starting_player=starting_player,randomize=False)
 
                while not game_over:
                    timestep += 1
                    actual_player = game.current_player
                    board = game.board
                    board_p2 = game.board_p2
                    
                    if actual_player == player1_ID:
                        act, _ = minimax_connect3(board, player1_ID, True, depth=depth1)
                        actions[player1] = act
                        
                        encoded_game.append(str(act))
                        flattened_board = np.ndarray.flatten(board)
                        board_plus_actions = np.append(flattened_board,float(act))
                        board_plus_action_total.append([board_plus_actions])
                        
                        _, _, done, _ = game.step(actions)
                        
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
                            
                        encoded_game.append(str(act))
                        actions[player2] = act
                        _, rew, done, _ = game.step(actions)
                    else:
                        raise ValueError("Player index is not valid, should be 0 or 1")
                    if logger:
                        logger.info("Game number " + str(i) + "/" + str(number_of_games))
                        logger.info(
                            "Player " + str(actual_player + 1) + " actions: " + str(act)
                        )
                        logger.info("\n" + repr(board))
                        logger.info(board_print(board,Connect3Config.HEIGHT,Connect3Config.WIDTH))
        
                    if done["__all__"]:
                        if logger:
                            logger.info("PLAYER " + str(game.winner + 1) + " WON...")
                            logger.info(
                                "CURRENT SCORE: "
                                + str(game.score[player1])
                                + " VS "
                                + str(game.score[player2])
                            )
                        game_over = True
                        game_str = ''.join(encoded_game)
                        # ADD ENCODED GAME TO THE LISt
                        if game_str in games_list:
                            number_of_equal_games += 1
                        elif game_str not in games_list:
                            games_list.append(game_str)
                            # if the game is too short, just discard it
                            if len(board_plus_action_total) < sequence_len:
                                continue
                            dataset_full_games.append([])
                            dataset_full_games[-1].append(board_plus_action_total)
                            dataset_full_games[-1].append(label)
                            
                            for j in range(len(board_plus_action_total)-(sequence_len-1)):
                                to_string=str(board_plus_action_total[j:j+sequence_len])
                                if to_string in sequence_list:
                                    indx = sequence_list.index(to_string)
                                    number_of_equal_sequences += 1
                                    if indx not in indx_to_remove:
                                        indx_to_remove.append(indx)

                                else:
                                    sequences_per_player[label].append(to_string)
                                    sequence_list.append(to_string)
                                    dataset_no_clones.append([])
                                    dataset_no_clones[-1].append(board_plus_action_total[j:j+sequence_len])
                                    dataset_no_clones[-1].append(label)
                                    

                                    
                                dataset.append([])
                                dataset[-1].append(board_plus_action_total[j:j+sequence_len])
                                dataset[-1].append(label)
                                
    final_dataset = []
    for i,elem in enumerate(dataset_no_clones):
        if i not in indx_to_remove:
            final_dataset.append(elem)
            
    count = 0
    final_sequences = {}
    for elem in sequences_per_player:
        final_sequences[elem] = []
        for indx,elem_2 in enumerate(sequences_per_player[elem]):
            if count not in indx_to_remove:
                final_sequences[elem].append(elem_2)
            count += 1
                        
    print("Original dataset length " + str(len(dataset_no_clones)) + " without clones: "  + str(len(final_dataset)))
    print("The number of equal games is: " + str(number_of_equal_games))
    print("The number of equal sequences is: " + str(number_of_equal_sequences))


    return  games_list, sequence_list, dataset,dataset_no_clones,final_dataset,dataset_full_games,final_sequences


def minimax_vs_minimax_connect3_single_game(depth1,depth2,sequence_len,discarded_moves=2,randomize= True,logger = None):
    """
        return observation plus actions data of a single game that are
        with shape readable by the lstm 
    """
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )
    

    board_plus_action_total = []
    game_over = False
    actions = {}
    single_game = []
    #label = depth1
    if randomize:
        starting_player = random.choice([player1_ID, player2_ID])
    else:
        starting_player = player1_ID
    game.reset(starting_player=starting_player,randomize=False)
 
    while not game_over:
        actual_player = game.current_player
        board = game.board
        board_p2 = game.board_p2
        
        if actual_player == player1_ID:
            act, _ = minimax_connect3(board, player1_ID, True, depth=depth1)
            actions[player1] = act
            
            flattened_board = np.ndarray.flatten(board)
            board_plus_actions = np.append(flattened_board,float(act))
            board_plus_action_total.append([board_plus_actions])
            
            _, _, done, _ = game.step(actions)
            
        elif actual_player == player2_ID:
            act, _ = minimax_connect3(board, player2_ID, True, depth=depth2)
            actions[player2] = act
            _, _, done, _ = game.step(actions)
        else:
            raise ValueError("Player index is not valid, should be 0 or 1")
        if logger:
            logger.info(
                "Player " + str(actual_player + 1) + " actions: " + str(act)
            )
            logger.info("\n" + repr(board))
            logger.info(board_print(board,Connect3Config.HEIGHT,Connect3Config.WIDTH))

        if done["__all__"]:
            if logger:
                logger.info("PLAYER " + str(game.winner + 1) + " WON...")
                logger.info(
                    "CURRENT SCORE: "
                    + str(game.score[player1])
                    + " VS "
                    + str(game.score[player2])
                )
            game_over = True
            # ADD ENCODED GAME TO THE LISt

            # if the game is too short, just discard it
            if len(board_plus_action_total) < sequence_len:
                return None
            single_game.append([])
            for j in range(len(board_plus_action_total)-(sequence_len-1)):
                if j >= discarded_moves:
                    single_game[-1].append(board_plus_action_total[j:j+sequence_len])

            if len(single_game[-1]) == 0:
                return None
                
            #single_game[-1].append(label)
                
    
    return  single_game

def minimax_vs_minimax_connect3_single_game_plus_outcome(depth1,depth2,sequence_len,discarded_moves=2,randomize= True,logger = None):
    """
        return observation plus actions data of a single game that are
        with shape readable by the lstm.
        It also discard some of the initial moves since they could be the 
        same for different level of playing
    """
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )
    

    board_plus_action_total = []
    game_over = False
    actions = {}
    single_game = []
    #label = depth1
    if randomize:
        starting_player = random.choice([player1_ID, player2_ID])
    else:
        starting_player = player1_ID
    game.reset(starting_player=starting_player,randomize=False)
 
    while not game_over:
        actual_player = game.current_player
        board = game.board
        board_p2 = game.board_p2
        
        if actual_player == player1_ID:
            act, _ = minimax_connect3(board, player1_ID, True, depth=depth1)
            actions[player1] = act
            
            flattened_board = np.ndarray.flatten(board)
            board_plus_actions = np.append(flattened_board,float(act))
            board_plus_action_total.append([board_plus_actions])
            
            _, rew, done, _ = game.step(actions)
            
        elif actual_player == player2_ID:
            act, _ = minimax_connect3(board, player2_ID, True, depth=depth2)
            actions[player2] = act
            _, rew, done, _ = game.step(actions)
        else:
            raise ValueError("Player index is not valid, should be 0 or 1")
        if logger:
            logger.info(
                "Player " + str(actual_player + 1) + " actions: " + str(act)
            )
            logger.info("\n" + repr(board))
            logger.info(board_print(board,Connect3Config.HEIGHT,Connect3Config.WIDTH))

        

        if done["__all__"]:
            if logger:
                logger.info("PLAYER " + str(game.winner + 1) + " WON...")
                logger.info(
                    "CURRENT SCORE: "
                    + str(game.score[player1])
                    + " VS "
                    + str(game.score[player2])
                )
            game_over = True
            if rew["player1"] == 1:
                outcome = 7.0
            elif rew["player1"] == -1:
                outcome = 8.0
            elif rew["player1"] == 0:
                outcome = 9.0
            # ADD ENCODED GAME TO THE LISt

            # if the game is too short, just discard it
            if len(board_plus_action_total) < sequence_len:
                return None
            board_plus_action_and_outcome = []
            for elem in board_plus_action_total:
                new_elem = np.append(elem,outcome)
                board_plus_action_and_outcome.append([new_elem])
            single_game.append([])
            for j in range(len(board_plus_action_and_outcome)-(sequence_len-1)):
                
                if j >= discarded_moves:
                    single_game[-1].append(board_plus_action_and_outcome[j:j+sequence_len])
                
            
            #single_game[-1].append(label)
            if len(single_game[-1]) == 0:
                return None
    
    return  single_game

def minimax_tournament(depth_list,number_of_games,randomize=True):
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )
    
    
    score_total = {}
    elo_diff_total = {}
    for depth2 in depth_list:
        for depth1 in depth_list:
            if depth2 >= depth1:
                continue
            print("Starting minimax depth " + str(depth1)  + " vs minimax " + str(depth2))
            game.reset_score()
            for i in range(number_of_games):
                # temp action and board state, 
                #action_total = []

                game_over = False
                actions = {}
                game.reset(randomize=randomize)
 
                while not game_over:
                    actual_player = game.current_player
                    board = game.board
                    
                    if actual_player == player1_ID:
                        act, _ = minimax_connect3(board, player1_ID, True, depth=depth1)
                        actions[player1] = act                        
                        _, _, done, _ = game.step(actions)
                        
                    elif actual_player == player2_ID:
                        act, _ = minimax_connect3(board, player2_ID, True, depth=depth2)
                        actions[player2] = act
                        _, _, done, _ = game.step(actions)
                    else:
                        raise ValueError("Player index is not valid, should be 0 or 1")

                    if done["__all__"]:
                        game_over = True
                        

            score = game.score[player1] / number_of_games + game.num_draws / (
                    2 * number_of_games
                )
        
            if score >= 10 / 11:
                elo_diff = 400
            elif score == 0 or score <= 1 / 11:
                elo_diff = -400
            else:
                elo_diff = -400 * math.log((1 / score - 1), 10)
                
            print("minimax depth " + str(depth1)  + " vs minimax " + str(depth2) + " elo difference: " + str(elo_diff))
            win_rate = game.score[player1]/number_of_games
            print("minimax depth " + str(depth1)  + " win rate: " + str(win_rate))
            score_total["depth" + str(depth1)+"_vs_depth" + str(depth2)] = win_rate
            elo_diff_total["depth" + str(depth1)+"_vs_depth" + str(depth2)] = elo_diff
                        
    return score_total,elo_diff_total



def return_one_hot(depth_list,depth_indx):
    one_hot_vec = [0]*len(depth_list)
    indx = depth_list.index(depth_indx)
    one_hot_vec[indx] = 1
    
    return one_hot_vec
    
def count_elem_in_dataset(dataset,classes):
    dataset_classes = {}
    for c in classes:
        dataset_classes[c] = 0
    for elem in dataset:
        if elem[1] in classes:
            dataset_classes[elem[1]] +=1
            
            
    return dataset_classes

if __name__ == "__main__":
    depth_list = [1,2,3,4,5,6]
    score, elo_diff = minimax_tournament(depth_list,100)
    