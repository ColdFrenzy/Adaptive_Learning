import os
import sys
import json


class Config:

    use_lstm = False
    as_test = True

    # =============================================================================
    # SYSTEM PARAMETERS
    # =============================================================================
    NUM_GPUS = 0


    # =============================================================================
    # ROLLOUT WORKERS
    # =============================================================================
    NUM_WORKERS = 0 #0 
    NUM_EVAL_WORKERS = 0
    NUM_ENVS_PER_WORKER = 1 #
    # Divide episodes into fragments of this many steps each during rollouts.
    # Sample batches of this size are collected from rollout workers and
    # combined into a larger batch of `train_batch_size` for learning.
    ROLLOUT_FRAGMENT_LENGTH = 200 #500,200
    # Training batch size, if applicable. Should be >= rollout_fragment_length.
    # Samples batches will be concatenated together to a batch of this size,
    # which is then passed to SGD. 
    # Train_batch_size also control the steps required before metrics are saved
    TRAIN_BATCH_SIZE = 2000 #5000,400
    SGD_MINIBATCH_SIZE = 200 #400
    NUM_SGD_ITER = 10 # 10 

    # =============================================================================
    # TRAINING PARAMS
    # =============================================================================
    EPOCHS = 10000
    EVALUATION_INTERVAL = None
    EVALUATION_NUMBER_OF_EPISODES = 100
    ELO_DIFF_UPB = 100
    ELO_DIFF_LWB = -100
    CKPT_STEP = 100
    CKPT_TO_KEEP = 5
    GAMMA = 0.9
    LEARNING_RATE = [0.0001, 0.001, 0.01]
    # shapes of the internal dense layer of the network
    HIDDEN_LAYER_SHAPES = [256,128,64]
    # if true will use a convolutional model instead of a dense one
    USE_CONV = True
    
    # =============================================================================
    # SELFPLAY
    # =============================================================================
    # if self_play is true, weights are regularly copied from the first to the
    # second network
    SELF_PLAY = True
    # number of steps before updating opponents policies weights,
    # if use_score is true this is ignored
    WEIGHT_UPDATE_STEP = 10  #50
    # difference in score to reach before updating the opponent policies weights
    REWARD_DIFFERENCE = 100
    # 0.6 is a good value in order to avoid overtraining
    WIN_RATE = 0.6
    # use REWARD_DIFFERENCE as metrics to copy weights, otherwise use 
    # WEIGHT_UPDATE_STEP or WIN_RATE 
    USE_SCORE = True
    # number of games against minimax
    GAMES_VS_MINIMAX = 100
    # number of previous weights to keep before computing the win-rate matrix
    HISTORY_LEN = 20 #20
    # number of most recent weights to keep after reaching HISTORY_LEN 
    # weights
    WEIGHTS_TO_KEEP = 10

    # self play is useless if both policies are already being trained
    POLICIES_TO_TRAIN = ["player1"]
    
    #POLICIES_TO_TRAIN_PROB = [.33, .33, .33]
    # past copies of the trained agent 
    OPPONENT_POLICIES_NOT_TRAINABLE = ["player2"]#,"player2_2","player2_3","player2_4"]
    # probability distribution of picking one of the opponent agents 
    # should have the same size of opponent_policies
    # OPPONENT_POLICIES_PROB = [.6, .4/3, .4/3, .4/3]
    
    
    

    # =============================================================================
    # MINIMAX PARAMS
    # =============================================================================
    SEARCH_DEPTH = 1
    MAX_DEPTH = 4

    # =============================================================================
    # CUSTOM METRICS
    # =============================================================================
    CUSTOM_METRICS = {}
    CUSTOM_METRICS["player1_score"] = 0.0
    CUSTOM_METRICS["player2_score"] = 0.0
    CUSTOM_METRICS["score_difference"] = 0.0
    CUSTOM_METRICS["number_of_draws"] = 0.0
    # CUSTOM_METRICS["opponent_policies"] = {}

    # Don't need to store evaluation metrics since they are reset at the start 
    # of every evaluation iteration
    # EVAL_METRICS = {}
    # EVAL_METRICS["player1_score"] = p1_score
    # EVAL_METRICS["minimax_score"] = p2_score
    # EVAL_METRICS["number_of_draws"] = draws
    # EVAL_METRICS["elo_difference"] = elo_diff
    # EVAL_METRICS["minimax_depth"] = depth

    # =============================================================================
    # PATHS
    # =============================================================================
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
        
    # File that stores the custom metrics from the last run 
    CUSTOM_METRICS_FILE = os.path.join(ROOT_DIR, "custom_metrics.json")
    if not os.path.exists(CUSTOM_METRICS_FILE):
        with open(CUSTOM_METRICS_FILE, "w") as json_file:
            json.dump(CUSTOM_METRICS, json_file)
            
    OUTPUT_DIR = os.path.join(ROOT_DIR,"output_dir")
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
        
    WEIGHTS_FILE = os.path.join(ROOT_DIR,"weights.npy")
    if not os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, "w") as json_file:
            pass
            
        
   # directory for tensorboard data      
    

# if __name__ == "__main__":
#     my_config = Config
#     with open(Config.MINIMAX_DEPTH_PATH,) as json_file:
#         data = json.load(json_file)
#         data["minimax_depth"] += 1

#     with open(Config.MINIMAX_DEPTH_PATH,"w") as json_file:
#         json.dump(data,json_file)
