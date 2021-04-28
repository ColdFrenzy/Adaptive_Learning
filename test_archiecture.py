"""
THIS FILE IS USED TO TEST DIFFERENT NETWORK ARCHITECTURE IN ORDER TO CHOOSE
THE BEST ONE. NETWORKS ARE TESTED AGAINST A MINIMAX ALGORITHM OF DEPTH 1 
"""

import sys
import os


sys.path.insert(1, os.path.abspath(os.path.curdir))

from tqdm import tqdm


from ray.rllib.agents.ppo import PPOTrainer
from models import Connect4ActionMaskModel
from utils.utils import custom_log_creator, restore_training, save_checkpoint
from config.custom_config import Config
from config.trainer_config import TrainerConfig



if __name__ == "__main__":

    _ = Connect4ActionMaskModel
    
    # =============================================================================
    # TRAINING PARAMETERS
    # =============================================================================
    epochs = Config.EPOCHS
    ckpt_step = Config.CKPT_STEP
    num_workers = Config.NUM_ENVS_PER_WORKER
    ray_results_dir = Config.RAY_RESULTS_DIR
    ckpt_dir = Config.CKPT_DIR
    ckpt_to_keep = Config.CKPT_TO_KEEP
    custom_metrics_file = Config.CUSTOM_METRICS_FILE
    max_depth = 5
    reward_diff = Config.REWARD_DIFFERENCE
    
    # =============================================================================
    # SELF PLAY
    # =============================================================================
    restore_ckpt = False

        
    # =============================================================================
    # STARTS
    # =============================================================================
    architectures_to_test = ["Simple_Dense","3_Layer_Dense","Conv",]
    for architecture in architectures_to_test:
        new_config = TrainerConfig.PPO_TRAINER_VS_MINIMAX
        if architecture == "Simple_Dense":
            new_config["model"]["custom_model_config"]["use_conv"] = False
            new_config["model"]["custom_model_config"]["hidden_layer_shapes"] = [256]
        elif architecture == "2_layer_Dense":
            new_config["model"]["custom_model_config"]["use_conv"] = False
            new_config["model"]["custom_model_config"]["hidden_layer_shapes"] = [256,128,64]
        else: 
            new_config["model"]["custom_model_config"]["use_conv"] = True
            
        p1_trainer_name = architecture + "_PPO"
        p2_trainer_name = "MiniMax"
        obs_space = TrainerConfig.OBS_SPACE
        print("The observation space is: ")
        print(obs_space)
        print("The action space is: ")
        act_space = TrainerConfig.ACT_SPACE
        print(act_space)
    
        
        trainer_obj = PPOTrainer(
            config=new_config,
            logger_creator=custom_log_creator(
                ray_results_dir, p1_trainer_name, p2_trainer_name, epochs
            ),
        )  # ,env=LogsWrapper)
        print("trainer " + str(p1_trainer_name) + " configured")
    
        if restore_ckpt:
            best_ckpt=restore_training(trainer_obj, ckpt_dir,custom_metrics_file)
            
        else:
            best_ckpt = 0
            print("Starting training from scratch")
            
        
        for epoch in range(best_ckpt + 1, epochs):
            print("Epoch " + str(epoch))
            results = trainer_obj.train()
            p1_score = results["custom_metrics"]["player1_score"]
            minimax_score = results["custom_metrics"]["player2_score"]
            score_difference = results["custom_metrics"]["score_difference"]
            actual_depth = trainer_obj.get_policy("minimax").depth
    
            if epoch % ckpt_step == 0 and epoch != 0:
                custom_metrics = results["custom_metrics"]
                save_checkpoint(trainer_obj,ckpt_dir,custom_metrics_file,custom_metrics,ckpt_to_keep)
    
            if p1_score >= minimax_score:
                print("Player 1 was able to beat MiniMax algorithm with depth " + str(actual_depth))
                break

