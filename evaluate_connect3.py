# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(1, os.path.abspath(os.path.curdir))

import tensorflow as tf 


import numpy as np


from config.connect4_config import Connect3Config
from utils.learning_behaviour_utils import LSTM_model,split_train_val,\
    minimax_vs_minimax_connect3_single_game,return_one_hot,\
    minimax_vs_minimax_connect3_single_game_plus_outcome,count_elem_in_dataset
from config.learning_behaviour_config import Config
from config.custom_config import Config as C
from tensorflow import keras




if __name__ == "__main__":
    
    data_dir = C.DATA_DIR
    lstm_hidden = Config.LSTM_HIDDEN[1] # best weights
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


    # =============================================================================
    # TEST THE MODEL 
    # =============================================================================
    
    model = LSTM_model(batch_size,(lstm_timesteps,features_len),output_len,lstm_hidden,False)  
    
    # generate a fake input to define the model stucture and then load the weights 
    # [batch,timestep,features]
    # random_input = np.random.rand(1,lstm_timesteps,features_len)
    random_input = np.random.rand(1,lstm_timesteps,features_len)
    model(random_input)
    model.set_weights(lstm_weights[()])
    
    confusion_matrix_total = {}
    accuracy = {}
    single_accuracy = 0
    total_sequences = 0
    for n in number_of_games_to_test: #[1,2,3,4,5]
        confusion_matrix = np.zeros((2,2))
        current_accuracy = 0        
        for t in range(number_of_evaluation_games): #[100]
            depth1 = np.random.choice(depth_list)           
            game = []
            for j in range(n):
                # we ensure that if we do more than 3 matches, we do at least 
                # one match against every one of the depth. 
                if n >=2 and j < 2:
                    depth2 = depth_list[j]
                else:
                    depth2 = np.random.choice(depth_list)
                    
                single_game = None
                # control needed because games with a size lower than the lstm_timesteps
                # are discarded
                while single_game == None:          
                    # single_game = \
                    # minimax_vs_minimax_connect3_single_game_plus_outcome(depth1,\
                    #                     depth2,lstm_timesteps,discarded_moves=0)
                    single_game=minimax_vs_minimax_connect3_single_game(depth1,\
                                    depth2,lstm_timesteps,0)
                for g in single_game[0]:
                    game.append(g)
            
            game = np.asarray(game)
            game = np.squeeze(game)
            if len(game.shape) == 2:
                game = np.expand_dims(game, axis=0)
            y = model(game,training=False)
            
            for p in y:
                total_sequences += 1
                p_indx = tf.math.argmax(p)
                r = return_one_hot(depth_list,depth1)
                r_indx = tf.math.argmax(r)
                if p_indx == r_indx:
                    single_accuracy += 1
            
            predicted_values = tf.math.reduce_mean(y,axis=0)
            predicted_indx = tf.math.argmax(predicted_values) 
            one_hot_out = return_one_hot(depth_list,depth1)
            real_indx = tf.math.argmax(one_hot_out)
            confusion_matrix[real_indx][predicted_indx] += 1
            if predicted_indx == real_indx:
                current_accuracy += 1
        confusion_matrix_total[n] = confusion_matrix
                
        accuracy[n] = current_accuracy/number_of_evaluation_games
        seq_accuracy = single_accuracy/total_sequences
        print("Accuracy over simple sequences: " + str(seq_accuracy))
        print("Accuracy with " + str(n) + " game is: " + str(accuracy[n]))
    
    
    
    
    dataset_file = os.path.join(data_dir,"dataset.npy")
    dataset = np.load(dataset_file,allow_pickle=True)

    
    
    # accuracy = {}
    # for n in number_of_games_to_test:
    #     current_accuracy = 0        
    #     for t in range(number_of_evaluation_games):
    #         depth1 = np.random.choice(depth_list)
            
    #         game = []
    #         for j in range(n):
    #             depth2 = np.random.choice(depth_list)
    #             single_game = None
    #             # control needed because games with a size lower than the lstm_timesteps
    #             # are discarded
    #             while single_game == None:          
    #                 single_game = minimax_vs_minimax_connect3_single_game(depth1,depth2,lstm_timesteps)
    #             for g in single_game[0]:
    #                 game.append(g)
            
    #         game = np.asarray(game)
    #         game = np.squeeze(game)
    #         if len(game.shape) == 2:
    #             game = np.expand_dims(game, axis=0)
    #         y = model(game)
    #         predicted_values = tf.math.reduce_mean(y,axis=0)
    #         predicted_indx = tf.math.argmax(predicted_values) 
    #         one_hot_out = return_one_hot(depth_list,depth1)
    #         real_indx = tf.math.argmax(one_hot_out)
    #         if predicted_indx == real_indx:
    #             current_accuracy += 1
                
    #     accuracy[n] = current_accuracy/number_of_evaluation_games
    #     print("Accuracy with " + str(n) + " game is: " + str(accuracy[n]))
        
        
    # npy_test_file = os.path.join(data_dir,"testset_full_games.npy")
    # test_set = np.load(npy_test_file,allow_pickle=True)
    
    # x_test, y_test,_,_= split_train_val(test_set,val_indx=0)
    # test_accuracy = 0.0
    

    # for elem in zip(x_test,y_test):
    #     single_game = []
    #     for j in range(len(elem[0])-(lstm_timesteps-1)):
    #         single_game.append(elem[0][j:j+lstm_timesteps])
    #     single_game = np.asarray(single_game)
    #     single_game = np.squeeze(single_game)
    #     if len(single_game.shape) == 2:
    #         single_game = np.expand_dims(single_game, axis=0)
    #     y = model(single_game)
    #     predicted_values = tf.math.reduce_mean(y,axis=0)
    #     predicted_indx = tf.math.argmax(predicted_values)
    #     real_indx = tf.math.argmax(elem[1])
    #     if predicted_indx == real_indx:
    #         test_accuracy += 1.0
                
    # test_accuracy = test_accuracy/len(x_test)
        
    
    # npy_test_file_2 = os.path.join(data_dir,"testset.npy")
    # test_set_episodes = np.load(npy_test_file_2,allow_pickle=True)
    
    # x_test_2,y_test_2, _, _ = split_train_val(test_set_episodes,val_indx=0)
    
    # x_test_2 = np.asarray(x_test_2) 
    # x_test_2 = np.squeeze(x_test_2)
    # y_test_2 = np.asarray(y_test_2) 
    # y_test_2 = np.squeeze(y_test_2)
    
    # history2 = model.test_on_batch(x_test_2,y_test_2)