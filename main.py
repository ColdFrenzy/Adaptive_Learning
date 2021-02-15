# import gym

# import argparse
import os

# import shutil
from datetime import datetime
import tempfile
from typing import Type

from ray.rllib.agents.pg import PGTFPolicy  # PGTorchPolicy
from ray.rllib.agents.trainer import with_common_config, Trainer
from ray.rllib.agents.trainer_template import build_trainer
from ray.tune.logger import UnifiedLogger

from env.LogWrapper import LogsWrapper
from models import Connect4ActionMaskModel
from policies.random_policy import RandomPolicy


# import gym
# import gym_connect4
# import sys
# TO CHECK CONDA ENV INFO
# project_dir = os.getenv("PROJECT_DIR")
# env = os.getenv("CONDA_DEFAULT_ENV")
# conda_path = os.getenv("PATH")
# curdir = os.path.abspath(os.getcwd())
# connect4 = os.path.join(curdir, "gym-connect4-master","gym_connect4",
#                         "envs","connect4_env")
# # insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, connect4)
# =============================================================================
# CONNECT 4 CUSTOM ENV IMPORT
# =============================================================================
# env_dict = gym.envs.registration.registry.env_specs.copy()
# for env in env_dict:
#     if 'Connect4-v0' in env:
#         print("Remove {} from registry".format(env))
#         del gym.envs.registration.registry.env_specs[env]
# from gym_connect4.envs import connect4_env
# =============================================================================
# PARSER
# =============================================================================
# parser = argparse.ArgumentParser()
# parser.add_argument("--torch", action="store_true")
# parser.add_argument("--as-test", action="store_true")
# parser.add_argument("--stop-iters", type=int, default=150)
# parser.add_argument("--stop-reward", type=float, default=1000.0)
# parser.add_argument("--stop-timesteps", type=int, default=100000)
CURDIR = os.path.abspath(os.path.curdir)
# LOGDIR = os.path.join(CURDIR,"log")
# if not os.path.exists(LOGDIR):
#     os.mkdir(LOGDIR)
# LOG_FILENAME = datetime.now().strftime("logfile_%H_%M_%S_%d_%m_%Y.log")
# =============================================================================
#  IMPORT CUSTOM MODEL, ENV, POLICIES AND CALLBACKS
# =============================================================================


def self_play(trainer: Type[Trainer]):
    # self learning     from 1 to 0
    # Copy weights from "player1" to "player2" after each training iteration
    P2key_P1val = {}  # Temp storage with "player2" keys & "player1" values

    # todo: check if this works with two policies of the same kind
    if not trainer.get_policy("player1") == trainer.get_policy("player2"):
        return

    for (p2_k, p2_v), (p1_k, p1_v) in zip(
        trainer.get_policy("player2").get_weights().items(),
        trainer.get_policy("player1").get_weights().items(),
    ):
        P2key_P1val[p2_k] = p1_v

    # set weights
    trainer.set_weights(
        {
            "player2": P2key_P1val,  # weights or values from "player1" with "player2" keys
            "player1": trainer.get_policy("player1").get_weights(True),  # no change
        }
    )

    print("Weight succesfully updated")
    # To check
    for (p2_k, p2_v), (p1_k, p1_v) in zip(
        trainer.get_policy("player1").get_weights().items(),
        trainer.get_policy("player2").get_weights().items(),
    ):
        assert (p2_v == p1_v).all()


def select_policy(agent_id):

    if agent_id == "player1":
        return "player1"
    else:
        return "player2"


def custom_log_creator(custom_path, trainer_name, env_id):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}_{}".format(trainer_name, env_id, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


if __name__ == "__main__":
    # register model
    _ = Connect4ActionMaskModel

    multiagent_connect4 = LogsWrapper(None)

    use_lstm = False
    as_test = True
    trainer_name = "PG"

    obs_space = multiagent_connect4.observation_space
    print("The observation space is: ")
    print(obs_space)
    print("The action space is: ")
    act_space = multiagent_connect4.action_space
    print(act_space)

    config = {
        # === Settings for Rollout Worker processes ===
        # "log_level": "INFO",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # parallel workers
        "num_workers": 0,
        # number of vectorwise environment per worker (for batching)
        "num_envs_per_worker": 1,
        "rollout_fragment_length": 10,
        "train_batch_size": 200,
        # === Environment Settings ===
        "env": LogsWrapper,
        # discounter factor of the MDP
        "gamma": 0.9,
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": ["player1"],  # ,"player2"],
            # MultiAgentPolicyConfigDict = Dict[PolicyID, Tuple[Union[
            # type, None], gym.Space, gym.Space, PartialTrainerConfigDict]]
            # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
            # of (policy_cls, obs_space, act_space, config)
            "policies": {
                # the last argument accept a policy config dict
                "player1": (
                    PGTFPolicy,
                    obs_space,
                    act_space,
                    {
                        "model": {
                            "custom_model": "connect4_mask",
                            "custom_model_config": {},
                        },
                    },
                ),
                "player2": (
                    RandomPolicy,
                    obs_space,
                    act_space,
                    {
                        "model": {
                            "custom_model": "connect4_mask",
                            "custom_model_config": {},
                        },
                    },
                ),
            },
            "policy_mapping_fn": select_policy,
        },
        # "callbacks": Connect4Callbacks,
        "framework": "tf2",
        # allow tracing in eager mode
        # "eager_tracing": True,
    }

    # =============================================================================
    # CHECKPOINT DIR
    # =============================================================================
    # i.e. serialize a policy to disk
    ckpt_dir = os.path.join(CURDIR, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    # shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # windows default result directory in C:/Users/*UserName*/ray_results
    # =============================================================================
    # RESULTS DIR
    # =============================================================================
    ray_results_dir = os.path.join(CURDIR, "ray_results")
    if not os.path.exists(ray_results_dir):
        os.mkdir(ray_results_dir)
    # shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    new_config = with_common_config(config)
    trainer = build_trainer(name=trainer_name, default_config=new_config)
    trainer_obj = trainer(
        config=new_config,
        logger_creator=custom_log_creator(ray_results_dir, trainer_name, "Connect4Env"),
    )
    print("trainer configured")
    env = trainer_obj.workers.local_worker().env
    print("local_worker environment acquired: " + str(env))
    epochs = 100
    reward_diff = 100
    weight_update_steps = 5
    reward_diff_reached = False

    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        results = trainer_obj.train()
        trainer_obj.save(ckpt_dir)
        # =============================================================================
        # UPDATE WEIGHTS FOR SELF-PLAY
        # =============================================================================
        if epoch % weight_update_steps == 0 and epoch != 0:
            self_play(trainer_obj)

        print(results)

        # Reward (difference) reached -> all good, return.
        if env.score[env.player1] - env.score[env.player2] > reward_diff:
            reward_diff_reached = True
            break

    # Reward (difference) not reached: Error if `as_test`.
    if not reward_diff_reached:
        print(
            "Desired reward difference {} not reached! Only got to {}.".format(
                reward_diff, env.score[env.player1] - env.score[env.player2]
            )
        )

        # raise ValueError(
        #     "Desired reward difference ({}) not reached! Only got to {}.".
        #     format(reward_diff, env.score[env.player1] - env.score[env.player2]))
    else:
        print(f"Desired reward difference {reward_diff} reached")
