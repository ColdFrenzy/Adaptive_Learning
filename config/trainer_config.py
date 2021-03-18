import os
from ray.rllib.agents.pg import PGTFPolicy  # PGTorchPolicy

# from policies.random_policy import RandomPolicy
from policies.minimax_policy import MiniMaxPolicy
from callbacks.custom_callbacks import Connect4Callbacks
from evaluation.custom_eval import Connect4Eval
from utils.utils import select_policy, select_evaluation_policy
from env.LogWrapper import LogsWrapper
from config.custom_config import Config


class TrainerConfig:
    ENV = LogsWrapper(None)
    OBS_SPACE = ENV.observation_space
    ACT_SPACE = ENV.action_space

    PG_TRAINER = {
        # === Settings for Rollout Worker processes ===
        # "log_level": "INFO",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # evaluation workers
        "evaluation_num_workers": Config.NUM_EVAL_WORKERS,
        # parallel workers, if 0 it will force rollouts to be done
        # in the trainer actor
        "num_workers": Config.NUM_WORKERS,
        # number of vectorwise environment per worker (for batching)
        "num_envs_per_worker": Config.NUM_ENVS_PER_WORKER,
        "rollout_fragment_length": Config.ROLLOUT_FRAGMENT_LENGTH,
        "train_batch_size": Config.TRAIN_BATCH_SIZE,
        # === Environment Settings ===
        "env": LogsWrapper,
        # discounter factor of the MDP
        "gamma": Config.GAMMA,
        "lr": Config.LEARNING_RATE[0],
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            # None for all policies
            "policies_to_train": ["player1"],  # ,"player2"],  # ,"player2"],
            # MultiAgentPolicyConfigDict = Dict[PolicyID, Tuple[Union[
            # type, None], gym.Space, gym.Space, PartialTrainerConfigDict]]
            # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
            # of (policy_cls, OBS_SPACE, ACT_SPACE, config)
            "policies": {
                # the last argument accept a policy config dict
                "player1": (
                    PGTFPolicy,
                    OBS_SPACE,
                    ACT_SPACE,
                    {
                        "model": {
                            "custom_model": "connect4_mask",
                            "custom_model_config": {},
                        },
                    },
                ),
                "player2": (
                    PGTFPolicy,
                    OBS_SPACE,
                    ACT_SPACE,
                    {
                        "model": {
                            "custom_model": "connect4_mask",
                            "custom_model_config": {},
                        },
                    },
                ),
                "minimax": (
                    MiniMaxPolicy,
                    OBS_SPACE,
                    ACT_SPACE,
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
        # === Evaluation Settings ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        # Note that evaluation is currently not parallelized, and that for Ape-X
        # metrics are already only reported for the lowest epsilon workers.
        "evaluation_interval": Config.EVALUATION_INTERVAL,
        # Number of episodes to run per evaluation period. If using multiple
        # evaluation workers, we will run at least this many episodes total.
        "evaluation_num_episodes": Config.EVALUATION_NUMBER_OF_EPISODES,
        # Internal flag that is set to True for evaluation workers.
        "in_evaluation": False,
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions.
        # IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
        # policy, even if this is a stochastic one. Setting "explore=False" here
        # will result in the evaluation workers not using this optimal policy!
        "evaluation_config": {
            # Example: overriding env_config, exploration, etc:
            # "env_config": {...},
            "explore": False,
            "multiagent": {
                # None for all policies
                "policies_to_train": ["player1"],  # "player2"],  # ,"player2"],
                # MultiAgentPolicyConfigDict = Dict[PolicyID, Tuple[Union[
                # type, None], gym.Space, gym.Space, PartialTrainerConfigDict]]
                # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
                # of (policy_cls, OBS_SPACE, ACT_SPACE, config)
                "policies": {
                    # the last argument accept a policy config dict
                    "player1": (
                        PGTFPolicy,
                        OBS_SPACE,
                        ACT_SPACE,
                        {
                            "model": {
                                "custom_model": "connect4_mask",
                                "custom_model_config": {},
                            },
                        },
                    ),
                    "player2": (
                        PGTFPolicy,
                        OBS_SPACE,
                        ACT_SPACE,
                        {
                            "model": {
                                "custom_model": "connect4_mask",
                                "custom_model_config": {},
                            },
                        },
                    ),
                    "minimax": (
                        MiniMaxPolicy,
                        OBS_SPACE,
                        ACT_SPACE,
                        {
                            "model": {
                                "custom_model": "connect4_mask",
                                "custom_model_config": {},
                            },
                        },
                    ),
                },
                "policy_mapping_fn": select_evaluation_policy,
            },
        },
        "custom_eval_function": Connect4Eval,
        "callbacks": Connect4Callbacks,
        "framework": "tf2",
        # allow tracing in eager mode
        # "eager_tracing": True,
    }
