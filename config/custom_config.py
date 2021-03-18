import os
import sys
import json


class Config:

    use_lstm = False
    as_test = True

    # =============================================================================
    # ROLLOUT WORKERS
    # =============================================================================
    NUM_WORKERS = 0
    NUM_EVAL_WORKERS = 0
    NUM_ENVS_PER_WORKER = 1
    ROLLOUT_FRAGMENT_LENGTH = 200
    TRAIN_BATCH_SIZE = 4000

    # =============================================================================
    # TRAINING PARAMS
    # =============================================================================
    EPOCHS = 100000
    REWARD_DIFFERENCE = 100
    WEIGHT_UPDATE_STEP = 10
    AVAILABLE_POLICIES = ["RANDOM", "MINIMAX", "PG"]
    EVALUATION_INTERVAL = 1000
    EVALUATION_NUMBER_OF_EPISODES = 200
    ELO_DIFF_UPB = 100
    ELO_DIFF_LWB = -100

    CKPT_STEP = 100
    GAMMA = 0.9
    LEARNING_RATE = [0.0001, 0.001, 0.01]

    # =============================================================================
    # MINIMAX PARAMS
    # =============================================================================
    SEARCH_DEPTH = 1

    # =============================================================================
    # PATHS
    # =============================================================================
    # ROOT_DIR = os.path.abspath(os.path.pardir)
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    IMPORTANT_CKPT_PATH = os.path.join(ROOT_DIR, "important_checkpoints")
    if not os.path.exists(IMPORTANT_CKPT_PATH):
        os.mkdir(IMPORTANT_CKPT_PATH)
    MINIMAX_DEPTH_PATH = os.path.join(IMPORTANT_CKPT_PATH, "minimax_depth.json")
    if not os.path.exists(MINIMAX_DEPTH_PATH):
        with open(MINIMAX_DEPTH_PATH, "w") as json_file:
            minimax_dict = {"minimax_depth": 1, "skipped_depth": []}
            json.dump(minimax_dict, json_file)

    # i.e. serialize a policy to disk
    CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    if not os.path.exists(CKPT_DIR):
        os.mkdir(CKPT_DIR)

    # windows default result directory in C:/Users/*UserName*/ray_results
    RAY_RESULTS_DIR = os.path.join(ROOT_DIR, "ray_results")
    if not os.path.exists(RAY_RESULTS_DIR):
        os.mkdir(RAY_RESULTS_DIR)


# if __name__ == "__main__":
#     my_config = Config
#     with open(Config.MINIMAX_DEPTH_PATH,) as json_file:
#         data = json.load(json_file)
#         data["minimax_depth"] += 1

#     with open(Config.MINIMAX_DEPTH_PATH,"w") as json_file:
#         json.dump(data,json_file)
