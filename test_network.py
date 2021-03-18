import json
import sys
import os

# NO GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(1, os.path.abspath(os.path.curdir))

import ray
from tqdm import tqdm
from env.LogWrapper import LogsWrapper

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from models import Connect4ActionMaskModel
from models import Connect4ActionMaskQModel
from utils.utils import custom_log_creator
from config.custom_config import Config
from config.trainer_config import TrainerConfig

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.dqn import DQNTrainer

if __name__ == "__main__":
    # register model
    _ = Connect4ActionMaskModel
    _ = Connect4ActionMaskQModel
    
    epochs = Config.EPOCHS
    reward_diff = Config.REWARD_DIFFERENCE
    ckpt_step = Config.CKPT_STEP
    reward_diff_reached = False
    ray_results_dir = os.path.join(Config.RAY_RESULTS_DIR,"algorithm_test")
    if not os.path.exists(ray_results_dir):
        os.mkdir(ray_results_dir)
    ckpt_dir = Config.CKPT_DIR
    as_test = Config.as_test
    # create or reset file

    p2_trainer_name = "Random"
    obs_space = TrainerConfig.OBS_SPACE
    print("The observation space is: ")
    print(obs_space)
    print("The action space is: ")
    act_space = TrainerConfig.ACT_SPACE
    print(act_space)
    trainers_config =  {"PPO_Dense_Net": [PPOTrainer,TrainerConfig.PPO_TEST],
                        "DQN_Dense_Net": [DQNTrainer,TrainerConfig.DQN_TEST],
                       "PG_Dense_Net" : [PGTrainer,TrainerConfig.PG_TEST]
                       }


    for trainer_name in trainers_config:
        p1_trainer_name = trainer_name
        new_config = with_common_config(trainers_config[trainer_name][1])
        trainer_obj = trainers_config[trainer_name][0](config=new_config,logger_creator=custom_log_creator(
                ray_results_dir, p1_trainer_name, p2_trainer_name, epochs),env=LogsWrapper)
        print("trainer " + str(trainer_name) + " configured")
        env = trainer_obj.workers.local_worker().env
        print("local_worker environment acquired: \n" + str(env))
    
        for epoch in tqdm(range(1, epochs)):
            print("Epoch " + str(epoch))
            results = trainer_obj.train()


            if "goal_reached_mean" in results["custom_metrics"]:
                print("Goal reached, beated the random agent in " + str(epoch) + " epochs")
                break
                    


            
