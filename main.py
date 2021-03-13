# import gym
import argparse
import sys
import os

# NO GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(1, os.path.abspath(os.path.curdir))
import shutil

import ray
from tqdm import tqdm

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from models import Connect4ActionMaskModel
from utils.utils import custom_log_creator, self_play
from config.custom_config import Config
from config.trainer_config import TrainerConfig


# =============================================================================
# PARSER
# =============================================================================
# parser = argparse.ArgumentParser(description="Execute the training of 2 bots and save the results\
#                                  in a tensorboard format for easy visualization.")
# parser.add_argument("-p1","--player1-policy", type=str,choices=Config.AVAILABLE_POLICIES,\
#                     help="Policy for player 1",required=True)
# parser.add_argument("-p2","--player2-policy", type=str,choices=Config.AVAILABLE_POLICIES,\
#                     help="Policy for player 2",required=True)
# parser.add_argument("--stop-iters", type=int, default=150)
# parser.add_argument("--stop-reward", type=float, default=1000.0)
# parser.add_argument("--stop-timesteps", type=int, default=100000)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    # register model
    _ = Connect4ActionMaskModel
    epochs = Config.EPOCHS
    reward_diff = Config.REWARD_DIFFERENCE
    ckpt_step = Config.CKPT_STEP
    reward_diff_reached = False
    ray_results_dir = Config.RAY_RESULTS_DIR
    ckpt_dir = Config.CKPT_DIR

    as_test = Config.as_test
    p1_trainer_name = "Dense_Network1"
    p2_trainer_name = "Dense_Network2"
    obs_space = TrainerConfig.OBS_SPACE
    print("The observation space is: ")
    print(obs_space)
    print("The action space is: ")
    act_space = TrainerConfig.ACT_SPACE
    print(act_space)

    new_config = with_common_config(TrainerConfig.PG_TRAINER)
    trainer = build_trainer(name=p1_trainer_name, default_config=new_config)
    trainer_obj = trainer(
        config=new_config,
        logger_creator=custom_log_creator(
            ray_results_dir, p1_trainer_name, p2_trainer_name, epochs
        ),
    )
    print("trainer configured")
    env = trainer_obj.workers.local_worker().env
    print("local_worker environment acquired: \n" + str(env))

    for epoch in tqdm(range(1, epochs)):
        print("Epoch " + str(epoch))
        results = trainer_obj.train()
        if epoch % ckpt_step == 0 and epoch != 0:
            trainer_obj.save(ckpt_dir)
            ckpts = os.listdir(ckpt_dir)
            # keep only the last 5 ckpts and delete the older ones
            if len(ckpts) > 20:
                for elem in ckpts:
                    if elem == ".gitkeep":
                        continue
                    index = int(elem.split("_")[1])
                    if index < epoch - 5:
                        dir_to_remove = os.path.join(ckpt_dir, elem)
                        shutil.rmtree(dir_to_remove)

        # if the updated network is able to beat it's previous self "reward_diff"
        # times we update the weights of the previous network
        if env.score[env.player1] - env.score[env.player2] >= reward_diff:
            # =============================================================================
            # UPDATE WEIGHTS FOR SELF-PLAY
            # =============================================================================
            try:
                self_play(trainer_obj)
                # we also reset the score
                env.reset_score()
            except:
                print("Error while updating weights")

    ray.shutdown()
