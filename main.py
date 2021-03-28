# import gym
import sys
import os

# NO GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(1, os.path.abspath(os.path.curdir))
import shutil

import ray
from tqdm import tqdm


# from env.LogWrapper import LogsWrapper
from ray.rllib.agents.trainer import with_common_config

# from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo import PPOTrainer
from models import Connect4ActionMaskModel
from utils.utils import custom_log_creator, self_play, restore_training,save_checkpoint,multiagent_self_play
from config.custom_config import Config
from config.trainer_config import TrainerConfig




if __name__ == "__main__":
    # You can set the object store size with the `object_store_memory`
    # parameter when starting Ray, and the max Redis size with `redis_max_memory`
    ray.init(ignore_reinit_error=True)  # ,object_store_memory=3000 * 1024 * 1024) #3GB
    # register model
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
    
    # =============================================================================
    # SELF PLAY
    # =============================================================================
    is_self_play = Config.SELF_PLAY
    use_score = Config.USE_SCORE
    restore_ckpt = False
    weight_update_step = Config.WEIGHT_UPDATE_STEP
    reward_diff = Config.REWARD_DIFFERENCE
        
    # =============================================================================
    # STARTS
    # =============================================================================
    p1_trainer_name = "PPO_Conv_Net"
    p2_trainer_name = "PPO_Conv_Net"
    obs_space = TrainerConfig.OBS_SPACE
    print("The observation space is: ")
    print(obs_space)
    print("The action space is: ")
    act_space = TrainerConfig.ACT_SPACE
    print(act_space)

    # new_config = with_common_config(TrainerConfig.PPO_TRAINER)
    # Trainable class
    trainer_obj = PPOTrainer(
        config=TrainerConfig.PPO_TRAINER,
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
        
    

    for epoch in tqdm(range(best_ckpt + 1, epochs)):
        print("Epoch " + str(epoch))
        results = trainer_obj.train()
        score_diff = results["custom_metrics"]["score_difference"]
        if epoch % ckpt_step == 0 and epoch != 0:
            custom_metrics = results["custom_metrics"]
            save_checkpoint(trainer_obj,ckpt_dir,custom_metrics_file,custom_metrics,ckpt_to_keep)


        if is_self_play:
            # if the updated network is able to beat it's previous self "reward_diff"
            # times we update the weights of the previous network
            if use_score:
                if score_diff >= reward_diff:
                    # =============================================================================
                    # UPDATE WEIGHTS FOR SELF-PLAY
                    # =============================================================================
                    # try:
                    multiagent_self_play(trainer_obj)
                    # we also reset the score
                    trainer_obj.callbacks.reset_values()
            else:
                if epoch%weight_update_step==0 and epoch != 0:
                   multiagent_self_play(trainer_obj) 

    ray.shutdown()
