import os
from ray.rllib.agents.pg import PGTFPolicy
from ray.rllib.agents.ppo import PPOTFPolicy
from ray.rllib.agents.dqn import DQNTFPolicy

from policies.random_policy import RandomPolicy
from policies.minimax_policy import MiniMaxPolicy
from callbacks.custom_callbacks import Connect4Callbacks,Connect4MiniMaxCallbacks
from callbacks.custom_test_callbacks import Connect4TestCallbacks

from evaluation.custom_eval import Connect4Eval
from utils.utils import select_policy, select_evaluation_policy, select_multiagent_policy
from env.LogWrapper import LogsWrapper,LogsWrapper_Connect3
from config.custom_config import Config
from config.connect4_config import Connect3Config


class TrainerConfig:
    # Connect4 
    ENV = LogsWrapper(None)
    OBS_SPACE = ENV.observation_space
    ACT_SPACE = ENV.action_space
    
    OPPONENT_POLICIES = {}
    for opp in Config.OPPONENT_POLICIES_NOT_TRAINABLE:
        OPPONENT_POLICIES[opp] = (None,OBS_SPACE,ACT_SPACE,{})

    # Connect3
    ENV_CONNECT3 = LogsWrapper_Connect3(None)
    OBS_SPACE_CONNECT3 = ENV_CONNECT3.observation_space
    ACT_SPACE_CONNECT3 = ENV_CONNECT3.action_space
        
    OPPONENT_POLICIES_CONNECT3 = {}
    for opp in Config.OPPONENT_POLICIES_NOT_TRAINABLE:
        OPPONENT_POLICIES_CONNECT3[opp] = (None,OBS_SPACE_CONNECT3,ACT_SPACE_CONNECT3,{})
        
        
    # PPO PARAMETERS TAKEN FROM RLLIB TUNED EXAMPLES
    # https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/atari-ppo.yaml
    # + standard PPO Params
    PPO_TRAINER = {
        # === PPO Specific Parameter ===
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
        "lambda": 0.95,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.5,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": Config.SGD_MINIBATCH_SIZE,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": Config.NUM_SGD_ITER,
        # Learning rate schedule.
        "lr_schedule": None,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.01,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.1,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": 5,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "complete_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # === Settings for Rollout Worker processes ===
        "num_gpus": Config.NUM_GPUS,#int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "evaluation_num_workers": Config.NUM_EVAL_WORKERS,
        "num_workers": Config.NUM_WORKERS,
        "num_envs_per_worker": Config.NUM_ENVS_PER_WORKER,
        "rollout_fragment_length": Config.ROLLOUT_FRAGMENT_LENGTH,
        "train_batch_size": Config.TRAIN_BATCH_SIZE,
        # === Environment Settings ===
        "env": LogsWrapper,
        "gamma": Config.GAMMA,
        "lr": Config.LEARNING_RATE[0],
        # === Model Settings ===
        "model": {
            "custom_model": "connect4_mask",
            "custom_model_config": {
                "hidden_layer_shapes" : Config.HIDDEN_LAYER_SHAPES,
                "use_conv" : Config.USE_CONV
                },
        },
    
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": Config.POLICIES_TO_TRAIN,
            "policies": {
                # the last argument accept a policy config dict
                "player1": (
                    PPOTFPolicy,
                    OBS_SPACE,
                    ACT_SPACE,
                    {
                    },
                ),
                **OPPONENT_POLICIES,
                
                #"minimax": (MiniMaxPolicy, OBS_SPACE, ACT_SPACE, {},),
            },
            "policy_mapping_fn": select_policy, #select_multiagent_policy,
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
            "explore": False,
            # "multiagent": {"policy_mapping_fn": select_multiagent_policy},
        },
        #"custom_eval_function": Connect4Eval,
        "callbacks": Connect4Callbacks,
        "output": Config.OUTPUT_DIR,
        "framework": "tf2",
    }
    
    PPO_TRAINER_CONNECT3 = {
        # === PPO Specific Parameter ===
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
        "lambda": 0.95,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.5,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": Config.SGD_MINIBATCH_SIZE,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": Config.NUM_SGD_ITER,
        # Learning rate schedule.
        "lr_schedule": None,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.01,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.1,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": 5,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "complete_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # === Settings for Rollout Worker processes ===
        "num_gpus": Config.NUM_GPUS,#int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "evaluation_num_workers": Config.NUM_EVAL_WORKERS,
        "num_workers": Config.NUM_WORKERS,
        "num_envs_per_worker": Config.NUM_ENVS_PER_WORKER,
        "rollout_fragment_length": Config.ROLLOUT_FRAGMENT_LENGTH,
        "train_batch_size": Config.TRAIN_BATCH_SIZE,
        # === Environment Settings ===
        "env": LogsWrapper_Connect3,
        "gamma": Config.GAMMA,
        "lr": Config.LEARNING_RATE[0],
        # === Model Settings ===
        "model": {
            "custom_model": "connect3_mask",
            "custom_model_config": {
                "hidden_layer_shapes" : Config.HIDDEN_LAYER_SHAPES,
                "use_conv" : Config.USE_CONV
                },
        },
    
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": Config.POLICIES_TO_TRAIN,
            "policies": {
                # the last argument accept a policy config dict
                "player1": (
                    PPOTFPolicy,
                    OBS_SPACE_CONNECT3,
                    ACT_SPACE_CONNECT3,
                    {
                    },
                ),
                **OPPONENT_POLICIES_CONNECT3,
                
                #"minimax": (MiniMaxPolicy, OBS_SPACE, ACT_SPACE, {},),
            },
            "policy_mapping_fn": select_policy, #select_multiagent_policy,
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
            "explore": False,
            # "multiagent": {"policy_mapping_fn": select_multiagent_policy},
        },
        #"custom_eval_function": Connect4Eval,
        "callbacks": Connect4Callbacks,
        "output": Config.OUTPUT_DIR,
        "framework": "tf2",
    }
    
    
    
    PPO_TRAINER_OPTIMIZED_SP = {
        # === PPO Specific Parameter ===
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
        "lambda": 0.95,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.5,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": Config.SGD_MINIBATCH_SIZE,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": Config.NUM_SGD_ITER,
        # Learning rate schedule.
        "lr_schedule": None,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.01,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.1,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": 5,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "complete_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # === Settings for Rollout Worker processes ===
        "num_gpus": Config.NUM_GPUS,#int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "evaluation_num_workers": Config.NUM_EVAL_WORKERS,
        "num_workers": Config.NUM_WORKERS,
        "num_envs_per_worker": Config.NUM_ENVS_PER_WORKER,
        "rollout_fragment_length": Config.ROLLOUT_FRAGMENT_LENGTH,
        "train_batch_size": Config.TRAIN_BATCH_SIZE,
        # === Environment Settings ===
        "env": LogsWrapper,
        "gamma": Config.GAMMA,
        "lr": Config.LEARNING_RATE[0],
        # === Model Settings ===
        "model": {
            "custom_model": "connect4_mask",
            "custom_model_config": {
                "hidden_layer_shapes" : Config.HIDDEN_LAYER_SHAPES,
                "use_conv" : Config.USE_CONV
                },
        },
    
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": Config.POLICIES_TO_TRAIN,
            "policies": {
                # the last argument accept a policy config dict
                "player1": (
                    PPOTFPolicy,
                    OBS_SPACE,
                    ACT_SPACE,
                    {
                    },
                ),
                **OPPONENT_POLICIES,
                
                "minimax": (MiniMaxPolicy, OBS_SPACE, ACT_SPACE, {},),
            },
            "policy_mapping_fn": select_multiagent_policy,
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
            "explore": False,
            # "multiagent": {"policy_mapping_fn": select_multiagent_policy},
        },
        "custom_eval_function": Connect4Eval,
        "callbacks": Connect4Callbacks,
        "output": Config.OUTPUT_DIR,
        "framework": "tf2",
    }
    
    

    PPO_TRAINER_VS_MINIMAX = {
        # === PPO Specific Parameter ===
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
        "lambda": 0.95,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.5,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": Config.SGD_MINIBATCH_SIZE,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": Config.NUM_SGD_ITER,
        # Learning rate schedule.
        "lr_schedule": None,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.01,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.1,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": 5,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "complete_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # === Settings for Rollout Worker processes ===
        "num_gpus": Config.NUM_GPUS,#int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "evaluation_num_workers": Config.NUM_EVAL_WORKERS,
        "num_workers": Config.NUM_WORKERS,
        "num_envs_per_worker": Config.NUM_ENVS_PER_WORKER,
        "rollout_fragment_length": Config.ROLLOUT_FRAGMENT_LENGTH,
        "train_batch_size": Config.TRAIN_BATCH_SIZE,
        # === Environment Settings ===
        "env": LogsWrapper,
        "gamma": Config.GAMMA,
        "lr": Config.LEARNING_RATE[0],
        # === Model Settings ===
        "model": {
            "custom_model": "connect4_mask",
            "custom_model_config": {
                "hidden_layer_shapes" : Config.HIDDEN_LAYER_SHAPES,
                "use_conv" : Config.USE_CONV
                },
        },
    
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": Config.POLICIES_TO_TRAIN,
            "policies": {
                # the last argument accept a policy config dict
                "player1": (
                    PPOTFPolicy,
                    OBS_SPACE,
                    ACT_SPACE,
                    {
                    },
                ),                
                "minimax": (MiniMaxPolicy, OBS_SPACE, ACT_SPACE, {},),
            },
            "policy_mapping_fn": select_evaluation_policy,
        },
        # === Evaluation Settings ===
        # NO EVALUATION
        "callbacks": Connect4MiniMaxCallbacks,
        "output": Config.OUTPUT_DIR,
        "framework": "tf2",
    }


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

    DQN_TEST = {
        # === Model ===
        # Number of atoms for representing the distribution of return. When
        # this is greater than 1, distributional Q-learning is used.
        # the discrete supports are bounded by v_min and v_max
        "num_atoms": 1,
        "v_min": -10.0,
        "v_max": 10.0,
        # Whether to use noisy network
        "noisy": False,
        # control the initial value of noisy nets
        "sigma0": 0.5,
        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [],
        # Whether to use double dqn
        "double_q": True,
        # N-step Q learning
        "n_step": 1,
        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.
        },
        # Minimum env steps to optimize for per train call. This value does
        # not affect learning, only the length of iterations.
        "timesteps_per_iteration": 1000,
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 500,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 50000,
        # The number of contiguous environment steps to replay at once. This may
        # be set to greater than 1 to support recurrent models.
        "replay_sequence_length": 1,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": True,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.6,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.4,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-6,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # Callback to run before learning on a multi-agent batch of experiences.
        "before_learn_on_batch": None,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,
        # === Optimization ===
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": 40,
        # How many steps of the model to sample before learning starts.
        "learning_starts": 1000,
        # === Parallelism ===
        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 1,
        # === Settings for Rollout Worker processes ===
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "evaluation_num_workers": Config.NUM_EVAL_WORKERS,
        "num_workers": Config.NUM_WORKERS,
        "num_envs_per_worker": Config.NUM_ENVS_PER_WORKER,
        "rollout_fragment_length": Config.ROLLOUT_FRAGMENT_LENGTH,
        "train_batch_size": Config.TRAIN_BATCH_SIZE,
        # === Environment Settings ===
        "env": LogsWrapper,
        "gamma": Config.GAMMA,
        "lr": Config.LEARNING_RATE[0],
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": ["player1"],
            "policies": {
                "player1": (
                    DQNTFPolicy,
                    OBS_SPACE,
                    ACT_SPACE,
                    {
                        "model": {
                            "custom_model": "connect4_q_mask",
                            "custom_model_config": {},
                        },
                    },
                ),
                "player2": (
                    RandomPolicy,
                    OBS_SPACE,
                    ACT_SPACE,
                    {
                        "model": {
                            "custom_model": "connect4_q_mask",
                            "custom_model_config": {},
                        },
                    },
                ),
            },
            "policy_mapping_fn": select_policy,
        },
        "callbacks": Connect4TestCallbacks,
        "framework": "tf2",
    }

    PPO_TEST = {
        # === PPO Specific Parameter ===
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
        "lambda": 1.0,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 128,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Learning rate schedule.
        "lr_schedule": None,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 1.0,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 10.0,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Target value for KL divergence.
        "kl_target": 0.01,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # === Settings for Rollout Worker processes ===
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "evaluation_num_workers": Config.NUM_EVAL_WORKERS,
        "num_workers": Config.NUM_WORKERS,
        "num_envs_per_worker": Config.NUM_ENVS_PER_WORKER,
        "rollout_fragment_length": Config.ROLLOUT_FRAGMENT_LENGTH,
        "train_batch_size": Config.TRAIN_BATCH_SIZE,
        # === Environment Settings ===
        "env": LogsWrapper,
        "gamma": Config.GAMMA,
        "lr": Config.LEARNING_RATE[0],
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": ["player1"],
            "policies": {
                "player1": (
                    PPOTFPolicy,
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
                    RandomPolicy,
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
        "callbacks": Connect4TestCallbacks,
        "framework": "tf2",
    }
    PG_TEST = {
        # === Settings for Rollout Worker processes ===
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "evaluation_num_workers": Config.NUM_EVAL_WORKERS,
        "num_workers": Config.NUM_WORKERS,
        "num_envs_per_worker": Config.NUM_ENVS_PER_WORKER,
        "rollout_fragment_length": Config.ROLLOUT_FRAGMENT_LENGTH,
        "train_batch_size": Config.TRAIN_BATCH_SIZE,
        # === Environment Settings ===
        "env": LogsWrapper,
        "gamma": Config.GAMMA,
        "lr": Config.LEARNING_RATE[0],
        # === Settings for Multi-Agent Environments ===
        "multiagent": {
            "policies_to_train": ["player1"],
            "policies": {
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
                    RandomPolicy,
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
        "callbacks": Connect4TestCallbacks,
        "framework": "tf2",
    }
