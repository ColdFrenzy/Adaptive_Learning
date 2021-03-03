# import gym
import argparse
import os

# import shutil
from datetime import datetime
import tempfile
from typing import Type

from ray.rllib.agents.pg import PGTFPolicy  # PGTorchPolicy
from ray.rllib.agents.trainer import with_common_config, Trainer
from ray.rllib.agents.trainer_template import build_trainer
from ray.tune.logger import UnifiedLogger

# =============================================================================
#  IMPORT CUSTOM MODEL, ENV, POLICIES AND CALLBACKS
# =============================================================================
from env.LogWrapper import LogsWrapper
from models import Connect4ActionMaskModel
from policies.random_policy import RandomPolicy
from policies.minimax_policy import MiniMaxPolicy
from callbacks.custom_callbacks import Connect4Callbacks
from config.custom_config import Config

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
CURDIR = os.path.abspath(os.path.curdir)


def self_play(trainer: Type[Trainer]):
    # check if the two policies have the same model (by comparing the models name)
    if (
        not trainer.get_policy("player1").model.base_model.name
        == trainer.get_policy("player2").model.base_model.name
    ):
        return

    # get weights
    p1_weights = trainer.get_policy("player1").model.base_model.get_weights()

    # set weights
    trainer.get_policy("player2").model.base_model.set_weights(p1_weights)

    print("Weight succesfully updated")
    # To check
    for w1, w2 in zip(
        trainer.get_policy("player1").model.base_model.get_weights(),
        trainer.get_policy("player2").model.base_model.get_weights(),
    ):
        assert (w1 == w2).all()


def select_policy(agent_id):

    if agent_id == "player1":
        return "player1"
    else:
        return "player2"


def custom_log_creator(custom_path, p1_trainer_name, p2_trainer_name, env_id):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_vs_{}_{}_{}".format(
        p1_trainer_name, p2_trainer_name, env_id, timestr
    )

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

    use_lstm = Config.use_lstm
    as_test = Config.as_test
    p1_trainer_name = "Minimax"
    p2_trainer_name = "Random"
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
        "num_workers": Config.NUM_WORKERS,
        # number of vectorwise environment per worker (for batching)
        "num_envs_per_worker": Config.NUM_ENVS_PER_WORKER,
        "rollout_fragment_length": Config.ROLLOUT_FRAGMENT_LENGTH,
        "train_batch_size": Config.TRAIN_BATCH_SIZE,
        # === Environment Settings ===
        "env": LogsWrapper,
        # discounter factor of the MDP
        "gamma": Config.GAMMA,
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": ["player1", "player2"],  # ,"player2"],
            # MultiAgentPolicyConfigDict = Dict[PolicyID, Tuple[Union[
            # type, None], gym.Space, gym.Space, PartialTrainerConfigDict]]
            # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
            # of (policy_cls, obs_space, act_space, config)
            "policies": {
                # the last argument accept a policy config dict
                "player1": (
                    MiniMaxPolicy,
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
        "callbacks": Connect4Callbacks,
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
    trainer = build_trainer(name=p1_trainer_name, default_config=new_config)
    trainer_obj = trainer(
        config=new_config,
        logger_creator=custom_log_creator(
            ray_results_dir, p1_trainer_name, p2_trainer_name, "Connect4Env"
        ),
    )
    print("trainer configured")
    env = trainer_obj.workers.local_worker().env
    print("local_worker environment acquired: \n" + str(env))
    epochs = Config.EPOCHS
    reward_diff = Config.REWARD_DIFFERENCE
    weight_update_steps = Config.WEIGHT_UPDATE_STEP
    reward_diff_reached = False

    for epoch in range(epochs):
        print("Epoch " + str(epoch))
        results = trainer_obj.train()
        trainer_obj.save(ckpt_dir)
        # =============================================================================
        # UPDATE WEIGHTS FOR SELF-PLAY
        # =============================================================================
        # if epoch % weight_update_steps == 0 and epoch != 0:
        #     try:
        #         self_play(trainer_obj)
        #     except:
        #         print("Error while updating weights")
        print(results)

        # Reward (difference) reached -> all good, return.
        if env.score[env.player1] - env.score[env.player2] >= reward_diff:
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
