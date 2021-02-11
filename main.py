# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 10:17:03 2020

@author: Francesco
"""


# import argparse
import os
import random
from datetime import datetime

# import gym
import numpy as np
from gym.spaces import Box

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
LOGDIR = os.path.join(CURDIR, "log")
if not os.path.exists(LOGDIR):
    os.mkdir(LOGDIR)
LOG_FILENAME = datetime.now().strftime("logfile_%H_%M_%S_%d_%m_%Y.log")


# REGISTER CUSTOM MODEL AND CUSTOM ENV
from models.action_mask_model import Connect4ActionMaskModel
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
# =============================================================================
# RAY IMPORTS
# =============================================================================
# import ray
# from ray.rllib import agents
# from ray import tune
from ray.rllib.agents.pg import PGTFPolicy, PGTrainer  # PGTorchPolicy
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
# from ray.rllib.policy.policy import Policy
# from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("connect4_mask", Connect4ActionMaskModel)


# connect4 = gym.make('Connect4-v0')
from env.connect4_multiagent_env import Connect4Env


def AI_vs_AI(
    multiagent_env,
    use_lstm=False,
    P1_policy="PG",
    P2_policy="DQN",
    trainer="PG",
    stop_iters=10,
    stop_timesteps=100,
    reward_diff=10,
    as_test=True,
):
    def select_policy(agent_id):
        if agent_id == "player1":
            return P1_policy
        else:
            return P2_policy

    obs_space = multiagent_env.observation_space
    print(obs_space)
    act_space = multiagent_env.action_space
    print(act_space)

    config = {
        "env": Connect4Env,
        # discounter factor of the MDP
        "gamma": 0.9,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "num_envs_per_worker": 3,
        "rollout_fragment_length": 10,
        "train_batch_size": 200,
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": [P1_policy, P2_policy],
            # MultiAgentPolicyConfigDict = Dict[PolicyID, Tuple[Union[
            #   type, None], gym.Space, gym.Space, PartialTrainerConfigDict]]
            # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
            # of (policy_cls, obs_space, act_space, config)
            "policies": {
                # the last argument accept a policy config dict
                # "PG": (PGTFPolicy, Box(low=-1, high=1,shape=(7, 6),dtype=np.int32),Discrete(8), {}),
                "player1": (PGTFPolicy, obs_space, act_space, {}),
                "player2": (A3CTFPolicy, obs_space, act_space, {}),
                # "learned": (None, Box(low=-1, high=1,shape=(7, 6),dtype=np.int32), Discrete(7), {
                #     # Arguments to pass to the policy model. See models/catalog.py for a full
                #     # list of the available model options. There is also the option for a
                #     # custom model
                #     "model": {
                #         "use_lstm": use_lstm
                #     },
                #     "framework": "tf2",
                # }),
            },
            "policy_mapping_fn": select_policy,
        },
        "framework": "tf2",
    }
    # return the right agent corresponding to the algorithm chosen
    cls = get_agent_class(trainer) if isinstance(trainer, str) else trainer
    print("trainer acquired")
    trainer_obj = cls(config=config)
    print("trainer configured")
    env = trainer_obj.workers.local_worker().env
    print("local_worker environment acquired: " + str(env))
    for i in range(stop_iters):
        print("starting iteration number " + str(i))
        results = trainer_obj.train()
        print(results)
        # Timesteps reached.
        if results["timesteps_total"] > stop_timesteps:
            break
        # Reward (difference) reached -> all good, return.
        elif env.score[env.player1] - env.score[env.player2] > reward_diff:
            return

    # Reward (difference) not reached: Error if `as_test`.
    if as_test:
        raise ValueError(
            "Desired reward difference ({}) not reached! Only got to {}.".format(
                reward_diff, env.reward[env.player1] - env.reward[env.player2]
            )
        )


# Function used to print the some infos like the action distribution
def on_postprocess_traj(info):
    """
    info["episode"] = MultiAgentEpisode Object
    info["agent_id"] = int
    info["pre_batch"] = [PGTFPolicy Object, SampleBatch Object ]
    info["post_batches"] = [SampleBatch Object]
    info["all_pre_batches"] = [PGTFPolicy Object, SampleBatch Object]

    SampleBatch = {'obs': ,
                   'actions':
                   'rewards':
                   'dones':
                   'agent_index'
                   'eps_id':
                   'unroll_id':
                   'advantages':
                   'value_targets':}

    # https://github.com/ray-project/ray/blob/ee8c9ff7320ec6a2d7d097cd5532005c6aeb216e/rllib/policy/sample_batch.py
    Dictionaries in a sample_obj, k:
        t
        eps_id
        agent_index
        obs
        actions
        rewards
        prev_actions
        prev_rewards
        dones
        infos
        new_obs
        action_prob
        action_logp
        vf_preds
        behaviour_logits
        unroll_id
    """
    """

    """
    agt_id = info["agent_id"]
    eps_id = info["episode"].episode_id
    policy_obj = info["pre_batch"][0]
    sample_obj = info["pre_batch"][1]

    # to see what's inside a given object just print obj.__dict__

    if agt_id == "player1":
        print("agent_id = {}".format(agt_id))
        print("episode = {}".format(eps_id))
        # #print("on_postprocess_traj info = {}".format(info))
        # #print("on_postprocess_traj sample_obj = {}".format(sample_obj))
        print("actions = {}".format(sample_obj.columns(["actions"])))

    elif agt_id == "player2":
        print("agent_id = {}".format(agt_id))
        print("episode = {}".format(eps_id))

        # #print("on_postprocess_traj info = {}".format(info))
        # #print("on_postprocess_traj sample_obj = {}".format(sample_obj))
        print("actions = {}".format(sample_obj.columns(["actions"])))
        # print('action_logs = {}'.format(sample_obj.columns(["action_dist_inputs"])))
        # print("on_postprocess_traj policy_obj = {}".format(policy_obj))
    return


def on_episode_step(info):
    """
    info["env"] = MultiAgentEnvToBaseEnv
    info["episode"] = MultiAgentEpisode
    """
    # print(info["episode"].__dict__)
    # print(info["env"].__dict__)

    # The action distribution is the same that we get with a print inside the
    # step function of our environment
    try:
        print(
            "Player 1 action distribution: "
            + str(
                info["episode"].__dict__["_agent_to_last_pi_info"]["player1"][
                    "action_dist_inputs"
                ]
            )
        )
        print(
            "Player 1 action logp: "
            + str(
                info["episode"].__dict__["_agent_to_last_pi_info"]["player1"][
                    "action_logp"
                ]
            )
        )
        print(
            "Player 1 action prob: "
            + str(
                info["episode"].__dict__["_agent_to_last_pi_info"]["player1"][
                    "action_prob"
                ]
            )
        )
    except:
        pass
    try:
        print(
            "Player 2 action distribution: "
            + str(
                info["episode"].__dict__["_agent_to_last_pi_info"]["player2"][
                    "action_dist_inputs"
                ]
            )
        )
        print(
            "Player 2 action logp: "
            + str(
                info["episode"].__dict__["_agent_to_last_pi_info"]["player2"][
                    "action_logp"
                ]
            )
        )
        print(
            "Player 2 action prob: "
            + str(
                info["episode"].__dict__["_agent_to_last_pi_info"]["player2"][
                    "action_prob"
                ]
            )
        )
    except:
        pass

    return


def on_episode_start(info):
    """
    Useful to check if everything was setted correctly
    info["env"] = MultiAgentEnvToBaseEnv
    info["policy"] = {"player1": P1Policy,
                      "player2": P2Policy},
    info["episode"] = MultiAgentEpisode

    """
    try:
        print("Player 1 " + str(info["policy"]["player1"].__dict__))
    except:
        pass
    try:
        print("Player 2 " + str(info["policy"]["player2"].__dict__))
    except:
        pass


if __name__ == "__main__":
    multiagent_connect4 = Connect4Env()

    use_lstm = False
    as_test = True
    trainer_name = "PG"

    def select_policy(agent_id):
        if agent_id == "player1":
            return "player1"
        else:
            return "player2"

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
        "env": Connect4Env,
        # discounter factor of the MDP
        "gamma": 0.9,
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": ["player1"],  # ,"player2"],
            # MultiAgentPolicyConfigDict = Dict[PolicyID, Tuple[Union[
            #   type, None], gym.Space, gym.Space, PartialTrainerConfigDict]]
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
                        # "framework": "tf2",
                    },
                ),
                "player2": (
                    PGTFPolicy,
                    obs_space,
                    act_space,
                    {
                        "model": {
                            "custom_model": "connect4_mask",
                            "custom_model_config": {},
                        },
                        # "framework": "tf2",
                    },
                ),
            },
            "policy_mapping_fn": select_policy,
        },
        "callbacks": {  # "on_episode_start": on_episode_start,
            # "on_episode_step": on_episode_step,
            # "on_episode_end": on_episode_end,
            # "on_sample_end": on_sample_end,
            "on_postprocess_traj": on_postprocess_traj,
            # "on_train_result": on_train_result,
        },
        "framework": "tf2",
        # allow tracing in eager mode
        # "eager_tracing": True,
    }

    new_config = with_common_config(config)
    trainer = build_trainer(name=trainer_name, default_config=new_config)
    trainer_obj = trainer(config=new_config)
    print("trainer configured")
    env = trainer_obj.workers.local_worker().env
    print("local_worker environment acquired: " + str(env))
    stop_iters = 100
    stop_timesteps = 500
    reward_diff = 10
    weight_update_steps = 10

    for i in range(stop_iters):
        print("starting iteration number " + str(i))
        results = trainer_obj.train()

        if i % weight_update_steps == 0 and i != 0:
            # self learning     from 1 to 0
            # Copy weights from "player1" to "player2" after each training iteration
            P2key_P1val = {}  # Temp storage with "player2" keys & "player1" values

            for (p2_k, p2_v), (p1_k, p1_v) in zip(
                trainer_obj.get_policy("player2").get_weights(True).items(),
                trainer_obj.get_policy("player1").get_weights(True).items(),
            ):
                P2key_P1val[p2_k] = p1_v

            # set weights
            trainer_obj.set_weights(
                {
                    "player2": P2key_P1val,  # weights or values from "player1" with "player2" keys
                    "player1": trainer_obj.get_policy("player1").get_weights(
                        True
                    ),  # no change
                }
            )

            print("Weight succesfully updated")
            # To check
            for (p2_k, p2_v), (p1_k, p1_v) in zip(
                trainer_obj.get_policy("player1").get_weights(True).items(),
                trainer_obj.get_policy("player2").get_weights(True).items(),
            ):
                assert (p2_v == p1_v).all()

        # print(results)
        # Timesteps reached.
        if results["timesteps_total"] > stop_timesteps:
            break
        # Reward (difference) reached -> all good, return.
        elif env.score[env.player1] - env.score[env.player2] > reward_diff:
            break

    # Reward (difference) not reached: Error if `as_test`.
    if as_test:
        print(
            "Desired reward difference ({}) not reached! Only got to {}.".format(
                reward_diff, env.score[env.player1] - env.score[env.player2]
            )
        )

        # raise ValueError(
        #     "Desired reward difference ({}) not reached! Only got to {}.".
        #     format(reward_diff, env.score[env.player1] - env.score[env.player2]))
