# -*- coding: utf-8
# import gym
import sys
import os

# NO GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(1, os.path.abspath(os.path.curdir))



import json
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import io
import itertools
import ray

# from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo import PPOTrainer
from models import Connect4ActionMaskModel
from utils.utils import custom_log_creator, restore_training,\
    save_checkpoint,multiagent_self_play, compute_best_policies
   
from utils.pre_compute_elo import compute_win_rate_matrix_connect3,\
    model_vs_minimax_connect3
from config.custom_config import Config
from config.trainer_config import TrainerConfig

def plot_matrix(wm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    wm (array, shape = [n, n]): a win rate matrix of floats
    class_names (array, shape = [n]): String names of the integer classes
  """
  blue = cm.get_cmap('Blues', 128)
  red = cm.get_cmap('Reds', 128)

  newcolors = np.vstack((blue(np.linspace(1, 0, 128)),
                       red(np.linspace(0, 1, 128))))

  new_cmp = ListedColormap(newcolors)
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(wm,cmap=new_cmp) #interpolation='nearest', cmap=plt.cm.Reds)
  plt.title("Win Rate Matrix ")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Transform in np.array in order to use some np functions
  wm = np.array(wm)
  # Compute the labels from the normalized confusion matrix.
  labels = np.around(wm,decimals=1)
  #labels = np.around(wm.astype("float")/ wm.sum(axis=1)[:, np.newaxis], decimals=2)
  # Use white text if squares are dark; otherwise black.
  threshold = wm.max() / 2.
  for i, j in itertools.product(range(wm.shape[0]), range(wm.shape[1])):
    color = "black" if wm[i, j] > threshold else "white"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  return figure


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

if __name__ == "__main__":

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
    weights_file = Config.WEIGHTS_FILE
    important_ckpt_path = Config.IMPORTANT_CKPT_PATH
    
    # =============================================================================
    # SELF PLAY
    # =============================================================================
    # is_self_play = Config.SELF_PLAY
    # use_score = Config.USE_SCORE
    restore_ckpt = True
    weight_update_step = Config.WEIGHT_UPDATE_STEP
    reward_diff = Config.REWARD_DIFFERENCE
    policies_to_train = Config.POLICIES_TO_TRAIN
    win_rate = Config.WIN_RATE
    history_len = Config.HISTORY_LEN
    weights_to_keep = Config.WEIGHTS_TO_KEEP
    games_vs_minimax = Config.GAMES_VS_MINIMAX
    max_depth = 8 #Config.MAX_DEPTH
    
    # =============================================================================
    # STARTS
    # =============================================================================
    p1_trainer_name = "Connect3_PPO_Conv_Net"
    p2_trainer_name = "No_GAE"
    obs_space = TrainerConfig.OBS_SPACE_CONNECT3
    print("The observation space is: ")
    print(obs_space)
    print("The action space is: ")
    act_space = TrainerConfig.ACT_SPACE_CONNECT3
    print(act_space)

    # new_config = with_common_config(TrainerConfig.PPO_TRAINER)
    # Trainable class
    trainer_obj = PPOTrainer(
        config=TrainerConfig.PPO_TRAINER_CONNECT3,
        logger_creator=custom_log_creator(
            ray_results_dir, p1_trainer_name, p2_trainer_name, epochs
        ),
    )  # ,env=LogsWrapper)
    print("trainer " + str(p1_trainer_name) + " configured")

    if restore_ckpt:
        # restoring checkpoints require ray
        ray.init()
        # best_ckpt=restore_training(trainer_obj, ckpt_dir,custom_metrics_file)
        with open(Config.MINIMAX_DEPTH_PATH) as json_file:
            data = json.load(json_file)
            minimax_depth = 3#data["minimax_depth"]
            
        # restore weights from a previous run 
        restored_weights = []
        weights = np.load(weights_file,allow_pickle=True)
        weights_name = ["p"+ str(i+1) for i in range(weights_to_keep)] 
        for name in weights_name:
            restored_weights.append(weights[()][name])
            trainer_obj.callbacks.add_weights(restored_weights[-1])
        # give player 1 the best weights
        trainer_obj.get_policy("player1").set_weights(restored_weights[-1])
            
        ray.shutdown()
            
    else:
        best_ckpt = 0
        minimax_depth = 1
        print("Starting training from scratch")
    
    # import moved here otherwise i get version compatibility issues by using 
    # the log_creator    
    import tensorflow as tf 
    
    
    number_of_stochastic_moves = 5
    
    
    logdir = str(trainer_obj._logdir)
    additional_metrics = {"additional_metrics":{}}
    file_writer = tf.summary.create_file_writer(logdir)
    file_writer.set_as_default()
    
    for epoch in tqdm(range(best_ckpt + 1, epochs)):
        print("Epoch " + str(epoch))
        # when we call the train() methods we are updating the weights but the
        # win_rate is referred to the previous weights used to collect the
        # rollouts
        # if internally the weight changed, also prev_weights change since
        # it is just a reference (tested, prev_weights doesn't change)
        prev_weights = trainer_obj.get_policy("player1").get_weights()
        results = trainer_obj.train()
        
        # CHECK IF Array are different 
        #updated_weights = trainer_obj.get_policy("player1").get_weights()
        # print("there are " + str(len(prev_weights)) + " weights")
        # indx = 0
        # equal_weights = []
        # for w1, w2 in zip(prev_weights,updated_weights):
        #     if np.array_equal(w1,w2)::
        #         equal_weights.append(indx)
        #     indx += 1
        # print(equal_weights)
        # input("Press Enter...")
        


        player1_win_rate = results["custom_metrics"]["player1_win_rate"]
        # instead of score_diff we use the win_ratio 
        if epoch % ckpt_step == 0 and epoch != 0:
            custom_metrics = results["custom_metrics"]
            save_checkpoint(trainer_obj,ckpt_dir,custom_metrics_file,custom_metrics,ckpt_to_keep)



        print("Actual player1 win rate: " + str(player1_win_rate))
        
        

        
        if player1_win_rate >= win_rate:
            # =============================================================================
            # UPDATE WEIGHTS FOR SELF-PLAY
            # =============================================================================
            # add the weights to the
            print("Adding policy to the history...")
            trainer_obj.callbacks.add_weights(prev_weights)


            if len(trainer_obj.callbacks.weights_history) == history_len:
                model1 = trainer_obj.get_policy("player1").model
                model2 = trainer_obj.get_policy("player2").model
                win_matrix = compute_win_rate_matrix_connect3(trainer_obj.callbacks.weights_history, model1,model2,number_of_stochastic_moves)
                # we removes all the weights (i.e. policies) which 
                # on average performed worse
                best_weights_indexes = compute_best_policies(win_matrix,weights_to_keep)
                
                # ordered in increasing score 
                trainer_obj.callbacks.keep_best_weights(best_weights_indexes)
                
                # use the best weights as player1 weights
                trainer_obj.get_policy("player1").set_weights(trainer_obj.callbacks.weights_history[-1])                          
                                                              
                # Save weights in .npy file 
                # clear the previous npy file 
                with open(weights_file,"w") as npyfile:
                    pass
                weights_to_save = {}
                for i,w in enumerate(trainer_obj.callbacks.weights_history):
                    weights_name = "p" + str(i+1)
                    weights_to_save[weights_name] = w
                    np.save(weights_file,weights_to_save)
                
                # with open(weights_file, "w") as json_file:
                #     json.dump(weights_to_save, json_file)

                
                # we want to visualize the win_rate_matrix as an image
                policy_names = ["p" +str(i) for i in range(len(win_matrix[0]))]
                figure = plot_matrix(win_matrix,policy_names)
                wm_image = plot_to_image(figure)
                tf.summary.image("Win Rate Matrix", wm_image, step=epoch)
                
                # best weights of this evaluation 
                best_weight = trainer_obj.callbacks.weights_history[0]
                # set the best weights as player 1 policy weights
                trainer_obj.get_policy("player1").set_weights(best_weight)
                model_to_evaluate = trainer_obj.get_policy("player1").model
                
                elo_diff,model_score,minimax_score,draw = model_vs_minimax_connect3(model_to_evaluate, minimax_depth,games_vs_minimax)
                
                p1_win_rate = model_score/games_vs_minimax
                additional_metrics["additional_metrics"]["minimax_depth"] = minimax_depth
                print("Winrate against minimax: " + str(p1_win_rate))
                if 0.45 <= p1_win_rate <= 0.55:
                    trainer_obj.save(important_ckpt_path)
                if p1_win_rate >= 0.6:
                    minimax_depth += 1
                    
                if minimax_depth == max_depth:
                    break 
                    
                
                additional_metrics["additional_metrics"]["player1_win_rate"]= p1_win_rate
                additional_metrics["additional_metrics"]["minimax_score"]=minimax_score
                additional_metrics["additional_metrics"]["player1_score"]=model_score
                additional_metrics["additional_metrics"]["elo_difference"] = elo_diff
                
                tf.summary.scalar('additional_metrics/player1_win_rate',data=additional_metrics["additional_metrics"]["player1_win_rate"],step=epoch)
                tf.summary.scalar('additional_metrics/minimax_depth', data=additional_metrics["additional_metrics"]["minimax_depth"], step=epoch)
                tf.summary.scalar('additional_metrics/minimax_score',data=additional_metrics["additional_metrics"]["minimax_score"],step=epoch)
                tf.summary.scalar('additional_metrics/player1_score',data=additional_metrics["additional_metrics"]["player1_score"],step=epoch)
                tf.summary.scalar('additional_metrics/elo_difference',data=additional_metrics["additional_metrics"]["elo_difference"],step=epoch)
            # instead of copy weights to policies, we moved this step 
            # inside the callback. We automatically 
            # multiagent_self_play(trainer_obj)
            trainer_obj.callbacks.reset_values()
        
    else:
        if epoch%weight_update_step==0 and epoch != 0:
           multiagent_self_play(trainer_obj) 
           trainer_obj.callbacks.reset_values()

        
