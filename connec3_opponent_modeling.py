# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(1, os.path.abspath(os.path.curdir))

import tensorflow as tf 


import numpy as np


from config.connect4_config import Connect3Config
from utils.learning_behaviour_utils import LSTM_model,split_train_val
from config.learning_behaviour_config import Config
from config.custom_config import Config as C
from tensorflow import keras




if __name__ == "__main__":
    
    data_dir = C.DATA_DIR
    lstm_hidden = Config.LSTM_HIDDEN
    batch_size = Config.BATCH_SIZE
    outcome_as_feature = Config.OUTCOME_AS_FEATURE
    if outcome_as_feature:
        features_len = Config.FEATURES_LEN_2
    else:
        features_len = Config.FEATURES_LEN
    lstm_timesteps = Config.LSTM_TIMESTEPS
    output_len = Config.OUTPUT_LEN
    depth_list = Config.DEPTH_LIST
    epochs = 100
    logdir = os.path.join(data_dir,"summaries")
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    
    
    weight_dict_all = []
    for elem in lstm_hidden:
        run = os.path.join(logdir,"run_"+str(elem))
        if not os.path.exists(run):
            os.mkdir(run)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=run)
        # [timestep, batch_size, features] (features = flattened_board(20) + action(1))
    
        # board_sequence = np.random.rand(10,5,21)
        model = LSTM_model(batch_size,(lstm_timesteps,features_len),output_len,elem,False)    
        #model.summary()
        
        # y = model(board_sequence)
        # =============================================================================
        # TRAIN THE MODEL 
        # =============================================================================
        npy_dataset_file = os.path.join(data_dir,"dataset.npy")
        loaded_data=np.load(npy_dataset_file,allow_pickle=True)
        x_train,y_train,x_val,y_val = split_train_val(loaded_data,depth_list,0.15,True)
        
        # squeeze removes len 1 dimensions 
        x_train = np.asarray(x_train) 
        x_train = np.squeeze(x_train)
        y_train = np.asarray(y_train)
        x_val = np.asarray(x_val)
        x_val = np.squeeze(x_val)
        y_val = np.asarray(y_val)
        
        
        # TODO remove if not using cosine 
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        
        weights_dict = {}
        weight_callback = tf.keras.callbacks.LambdaCallback \
            ( on_epoch_end=lambda epoch, logs:  weights_dict.update({epoch:model.get_weights()}))
        
        
        # don't use validation split without shuffling first. In fact if we
        # use the split in model.fit it will take the last example in the dataset.
        # and if they are not shuffled, this implies that they will all belong
        # to the same class
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle = True,
            verbose=2, # Suppress chatty output; use Tensorboard instead
            #validation_split = 0.15,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(x_val, y_val),
            callbacks=[tensorboard_callback,weight_callback]
        )
        #model.summary()
    
        weight_dict_all.append(weights_dict)
        
        
        
    # TODO automatic save best weights 
    best_weights_npy = os.path.join(data_dir,"lstm_best_weights.npy")
    with open(best_weights_npy,"w") as npyfile:
        pass
    np.save(best_weights_npy,weight_dict_all[1][85]) # or -2 60/95 -1 77
        
    # =============================================================================
    # TEST THE MODEL 
    # =============================================================================
    
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
    
    
    
    
    