# -*- coding: utf-8 -*-
"""
    Generates data from self-play that will be later used for opponent 
    modeling
"""
import sys
import os
import numpy as np
import ray
import logging
import csv

sys.path.insert(1, os.path.abspath(os.path.curdir))

from models import Connect4ActionMaskModel
from config.custom_config import Config
from config.learning_behaviour_config import Config as LBConfig
from config.trainer_config import TrainerConfig
from ray.rllib.agents.ppo import PPOTrainer
#from utils.pre_compute_elo import model_vs_model_connect3_generate_data
from utils.pre_compute_elo import init_logger
from utils.learning_behaviour_utils import model_vs_model_connect3_generate_data,\
    minimax_vs_minimax_connect3_generate_data,model_vs_model_connect3_generate_data_v2,\
        model_vs_model_connect3_generate_data_v3,model_vs_model_connect3_generate_data_v4,\
            count_elem_in_dataset,\
        minimax_vs_model_connect3_generate_data

def create_npy(trainer,data_dir,weights_dirs,npy_weight_file):
    """
    utility function to convert weights in npy files
    """
    weights_dict = {}
    for d in weights_dirs:
        ckpt_dir = os.path.join(data_dir,d)
        weights = os.listdir(ckpt_dir)
        for j,w in enumerate(weights):
            index = w.split("_")[1]
            w_file = os.path.join(ckpt_dir,w,"checkpoint-"+index)
            trainer.restore(w_file)
            weights_dict[d+"_"+str(j)] = trainer.get_policy("player1").get_weights()
    
    # create if not exists
    with open(npy_weights_file,"w") as npyfile:
        pass
    np.save(npy_weight_file,weights_dict)


def load_npy(data_dir,weights_dirs,npy_weight_file):
    weights = np.load(npy_weights_file,allow_pickle=True)
    weights_dict = {}
    weights_name = []
    for d in weights_dirs:
        ckpt_dir = os.path.join(data_dir,d)
        weights_indx = os.listdir(ckpt_dir)
        weights_name.append([])
        for w in weights_indx:
            weights_name[-1].append(w)
            
    for w_depth,j in zip(weights_name,weights_dirs):
        for w in w_depth:
            index = w.split("_")[-1]
            weights_dict[j+"_"+str(index)] = weights[()][w]
    return weights_dict





if __name__ == "__main__":
    _ = Connect4ActionMaskModel
        
    data_dir = Config.DATA_DIR
    log_file= os.path.join(data_dir,"data.log")
    logging = init_logger(log_file)
    file_depths = ["depth1","depth4"]
    depth_list = LBConfig.DEPTH_LIST
    outcome_as_feature = LBConfig.OUTCOME_AS_FEATURE
    weights_file = os.path.join(Config.DATA_DIR)
    logdir = os.path.join(data_dir,"summaries")
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    npy_weights_file = os.path.join(data_dir,"weights.npy")
    npy_dataset_file = os.path.join(data_dir,"dataset.npy")
    npy_dataset_clones_file = os.path.join(data_dir,"dataset_clones.npy")
    npy_testset_file = os.path.join(data_dir,"testset.npy")
    npy_testset_full_games = os.path.join(data_dir,"testset_full_games.npy")
    dataset_file = os.path.join(data_dir,"dataset.csv")
    games_encoded_file = os.path.join(data_dir,"dataset_encoded.json")
    sequence_encoded_file = os.path.join(data_dir,"sequence_encoded.json")
    
    # =============================================================================
    # PARAMETERS
    # =============================================================================
    number_of_games = 300
    minimax_games = 100
    number_of_stochastic_moves = 6
    lstm_timesteps = LBConfig.LSTM_TIMESTEPS
    
    # =============================================================================
    # STARTS
    # =============================================================================
    obs_space = TrainerConfig.OBS_SPACE_CONNECT3
    print("The observation space is: ")
    print(obs_space)
    print("The action space is: ")
    act_space = TrainerConfig.ACT_SPACE_CONNECT3
    print(act_space)
    
    trainer_obj = PPOTrainer(
        config=TrainerConfig.PPO_TRAINER_CONNECT3,
    )
    
    # ray.init()
    # create_npy(trainer_obj,data_dir,file_depths,npy_weights_file)
    # ray.shutdown()
    
    # weights = load_npy(data_dir,file_depths,npy_weights_file)
    # weights_name = list(weights.keys())
    weights = np.load(npy_weights_file,allow_pickle=True)[()]        
    
    # here we insert the game coded as a string where the first two character
    # indicates the player who starts (p1 or p2)  and the continue of the string 
    # is the action picked. Example of a game started by p1
    # p1_01020    
    model1 = trainer_obj.get_policy("player1").model
    model2 = trainer_obj.get_policy("player2").model
    # model1.base_model.set_weights(weights[weights_name[0]])
    # model2.base_model.set_weights(weights[weights_name[1]])
    games_encoded,sequence_list,dataset,dataset_no_clones,final_sequences = \
    model_vs_model_connect3_generate_data_v4(model1,model2,outcome_as_feature,weights,\
    number_of_games,lstm_timesteps,number_of_stochastic_moves=number_of_stochastic_moves)
    
    # np_dataset = []
    # for elem in dataset_no_clones:
    #     np_dataset.append([])
    #     np_dataset[-1].append(np.array(elem[0]))
    #     np_dataset[-1].append(np.array(elem[1]))
        
    

    
    # same thing if we save directly dataset_no_clones
    with open(npy_dataset_file,"w") as npyfile:
        pass
    np.save(npy_dataset_file,dataset_no_clones)
    
    with open(npy_dataset_clones_file,"w") as npyfile:
        pass
    np.save(npy_dataset_clones_file,dataset)
    
    
    data = np.load(npy_dataset_file,allow_pickle=True)
    data_with_clones = np.load(npy_dataset_clones_file,allow_pickle=True)
    element_per_class = count_elem_in_dataset(data,depth_list)
    element_per_class_clones = count_elem_in_dataset(data_with_clones,depth_list)
    

    
    minimax_games_encoded, minimax_sequence_list,minimax_dataset,_,\
    minimax_dataset_no_clones,_,minimax_final_sequences = \
    minimax_vs_minimax_connect3_generate_data(depth_list,number_of_games*10,lstm_timesteps)

    
    agent_final_sequences_file = os.path.join(data_dir,"agent_final_sequences.npy")
    with open(agent_final_sequences_file,"w") as npyfile:
        pass
    np.save(agent_final_sequences_file,final_sequences)
    
    minimax_final_sequences_file = os.path.join(data_dir,"minimax_final_sequences.npy")
    with open(minimax_final_sequences_file,"w") as npyfile:
        pass
    np.save(minimax_final_sequences_file,minimax_final_sequences)
    
    confusion_matrix = np.zeros((len(depth_list),len(depth_list)))
    different_moves = 0
    for s_1, indx_1 in enumerate(final_sequences):
        for elem_1 in final_sequences[indx_1]:
            c = 0
            for s_2,indx_2 in enumerate(minimax_final_sequences):
                if elem_1 in minimax_final_sequences[indx_2]:
                    confusion_matrix[s_1][s_2] += 1
                    c += 1
            if c== 0:
                different_moves += 1
                
                
    confusion_matrix_2 = np.zeros((len(depth_list),len(depth_list)))
    different_moves_2 = 0
    for s_1, indx_1 in enumerate(minimax_final_sequences):
        for elem_1 in minimax_final_sequences[indx_1]:
            c = 0
            for s_2,indx_2 in enumerate(final_sequences):
                if elem_1 in final_sequences[indx_2]:
                    confusion_matrix_2[s_1][s_2] += 1
                    c += 1
            if c== 0:
                different_moves_2 += 1
                
    
# =============================================================================
#     agent_final_sequences_file = os.path.join(data_dir,"agent_final_sequences.npy")
#     final_sequences = np.load(agent_final_sequences_file,allow_pickle=True)[()]
#     
#     minimax_games_encoded, minimax_sequence_list,minimax_dataset,_,\
#     minimax_dataset_no_clones,_,minimax_final_sequences = \
#     minimax_vs_model_connect3_generate_data(depth_list,weights,model1,number_of_stochastic_moves,number_of_games,lstm_timesteps,randomize= True,logger = None)
#     
#     confusion_matrix = np.zeros((len(depth_list),len(depth_list)))
#     different_moves = 0
#     for s_1, indx_1 in enumerate(final_sequences):
#         for elem_1 in final_sequences[indx_1]:
#             c = 0
#             for s_2,indx_2 in enumerate(minimax_final_sequences):
#                 if elem_1 in minimax_final_sequences[indx_2]:
#                     confusion_matrix[s_1][s_2] += 1
#                     c += 1
#             if c== 0:
#                 different_moves += 1
#                 
#     confusion_matrix_2 = np.zeros((len(depth_list),len(depth_list)))
#     different_moves_2 = 0
#     for s_1, indx_1 in enumerate(minimax_final_sequences):
#         for elem_1 in minimax_final_sequences[indx_1]:
#             c = 0
#             for s_2,indx_2 in enumerate(final_sequences):
#                 if elem_1 in final_sequences[indx_2]:
#                     confusion_matrix_2[s_1][s_2] += 1
#                     c += 1
#             if c== 0:
#                 different_moves_2 += 1
#     
# =============================================================================
    
    
    # with open(npy_dataset_file,"w") as npyfile:
    #     pass
    # np.save(npy_dataset_file,dataset_no_clones)

    # with open(npy_testset_file,"w") as npyfile:
    #     pass
    # np.save(npy_testset_file,minimax_dataset_no_clones)
    
    # with open(npy_testset_full_games,"w") as npy_file:
    #     pass 
    # np.save(npy_testset_full_games,minimax_dataset_full_games)
    # load_data=np.load(npy_dataset_file,allow_pickle=True)


    # with open(dataset_file,"w") as f:
    #     wr = csv.writer(f)
    #     wr.writerows(dataset_no_clones)