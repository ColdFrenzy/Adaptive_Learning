# -*- coding: utf-8
import sys
import os

sys.path.insert(1, os.path.abspath(os.path.curdir))

import numpy as np


from ray.rllib.agents.ppo import PPOTrainer
from models import Connect4ActionMaskModel

from utils.pre_compute_elo import init_logger
from utils.pre_compute_elo import model_vs_minimax_connect3
from config.custom_config import Config
from config.trainer_config import TrainerConfig



if __name__ == "__main__":
    my_logger = init_logger("log/policies_vs_minimax.log")
    _ = Connect4ActionMaskModel
    weights_to_keep = Config.WEIGHTS_TO_KEEP
    weights_file = Config.WEIGHTS_FILE
    games_vs_minimax = Config.GAMES_VS_MINIMAX
    minimax_depth = 1
    p1_trainer_name = "Connect3_PPO_Conv_Net"
    p2_trainer_name = "PPO_Conv_Net"
    obs_space = TrainerConfig.OBS_SPACE_CONNECT3
    print("The observation space is: ")
    print(obs_space)
    print("The action space is: ")
    act_space = TrainerConfig.ACT_SPACE_CONNECT3
    print(act_space)
    trainer_obj = PPOTrainer(
        config=TrainerConfig.PPO_TRAINER_CONNECT3,
    )
    restored_weights = []
    weights = np.load(weights_file,allow_pickle=True)
    weights_name = ["p"+ str(i+1) for i in range(weights_to_keep)] 
    for name in weights_name:
        restored_weights.append(weights[()][name])
        trainer_obj.callbacks.add_weights(restored_weights[-1])
        
    for i,weights in enumerate(restored_weights):
        trainer_obj.get_policy("player1").set_weights(weights)

        model_to_evaluate = trainer_obj.get_policy("player1").model
        updated_weights = trainer_obj.get_policy("player1").get_weights()
                        
        print("there are " + str(len(weights)) + " weights")
        
        indx = 0
        equal_weights = []
        for w1, w2 in zip(weights,updated_weights):
            if np.array_equal(w1,w2):
                equal_weights.append(indx)
            indx += 1
        print(equal_weights)


        elo_diff,model_score,minimax_score,draw = model_vs_minimax_connect3(model_to_evaluate, minimax_depth,games_vs_minimax,logger=my_logger)
       
             
        p1_win_rate = model_score/games_vs_minimax
        print("Policy " + str(i+1) + " win rate against minimax: " + str(p1_win_rate))
        