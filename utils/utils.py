from datetime import datetime
import tempfile
import os
import json
import shutil
import numpy as np 
from typing import Type
from ray.tune.logger import UnifiedLogger
from config.custom_config import Config
from ray.rllib.agents.trainer import Trainer


def select_policy(agent_id):

    if agent_id == "player1":
        return "player1"
    # to avoid overfitting over a single strategy, we keep 3 networks trained
    # independently
    else:
        return "player2"
        # return random.choice(["player2_1","player2_2","player2_3"])


def select_multiagent_policy(agent_id):
    """ 
    allows to select between different past opponents with a certain frequency
    """
    if agent_id == "player1":
        return "player1"
    else: 
        return np.random.choice(Config.OPPONENT_POLICIES,1,
                                p=Config.OPPONENT_POLICIES_PROB)[0]

def select_evaluation_policy(agent_id):
    if agent_id == "player1":
        return "player1"
    else:
        return "minimax"


def custom_log_creator(custom_path, p1_trainer_name, p2_trainer_name, epochs):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_vs_{}_{}-Epochs_{}".format(
        p1_trainer_name, p2_trainer_name, epochs, timestr
    )

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def self_play(trainer: Type[Trainer]):
    # check if the two policies have the same model (by comparing the models name)
    assert trainer.get_policy("player1").model.base_model.name\
        == trainer.get_policy("player2").model.base_model.name,\
        "Error: you should use the same model for every player"

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


def multiagent_self_play(trainer: Type[Trainer]):
    """
    Update weights between multiple policies 
    """        
    new_weights = trainer.get_policy("player1").get_weights()
    for opp in Config.OPPONENT_POLICIES:
        prev_weights = trainer.get_policy(opp).get_weights()
        trainer.get_policy(opp).set_weights(new_weights)
        new_weights = prev_weights
            
    print("Weight succesfully updated")
    



def restore_training(trainer_obj,ckpt_dir,metrics_file):
    """
    Restore the latest checkpoint and the latest metrics
    trainer_obj: Trainable
        trainer to resume 
    ckpt_dir: str
        path to the directory with the checkpoints
    metrics_file: str
        path to the directory with the latest custom metrics observed 
    
    """
    best_ckpt = 0
    ckpt_to_restore = None
    # Restore the latest checkpoint if exist:
    for ckpt in os.listdir(ckpt_dir):
        if ckpt == ".gitkeep":
            continue
        ckpt_indx = int(ckpt.split("_")[1])
        if ckpt_indx > best_ckpt:
            best_ckpt = ckpt_indx
    # if the checkpoint exists return the checkpoint and latest metrics
    if best_ckpt > 0:
        ckpt_to_restore = os.path.join(
            ckpt_dir, "checkpoint_" + str(best_ckpt), "checkpoint-" + str(best_ckpt)
        )
        trainer_obj.restore(ckpt_to_restore)
        
        print("Checkpoint number " + str(best_ckpt) + " restored")
        # we also need to restore the custom metrics
        with open(metrics_file) as json_file:
            data = json.load(json_file)
        trainer_obj.callbacks.load_values(data)
        print("Values of the latest custom metrics have been restored")
    else:
        print("No checkpoint found, Training starting from scratch...")
        
    return best_ckpt

def save_checkpoint(trainer_obj,ckpt_dir,metrics_file,custom_metrics,ckpt_to_keep = 5):
    """
    Save the checkpoint in the ckpt_dir and the current metrics in metrics_file
    trainer_obj: Trainable
        trainer to resume 
    ckpt_dir: str
        path to the directory with the checkpoints
    metrics_file: str
        path to the directory with the latest custom metrics observed 
    custom_metrics: dict
        custom metrics to save
    ckpt_to_keep: int
        number of ckpt to keep 
    """
    trainer_obj.save(ckpt_dir)
    ckpts = os.listdir(ckpt_dir)
    # keep only the last ckpt_to_keep ckpts and delete the older ones 
    ckpts.remove(".gitkeep")
    with open(metrics_file, "w") as json_file:
        json.dump(custom_metrics, json_file)
    if len(ckpts) > ckpt_to_keep:
        # sort the checkpoint list 
        ckpts.sort(key=lambda x: int(x.split("_")[1]),reverse=True)
        for i,elem in enumerate(ckpts):
            if i > ckpt_to_keep-1:
                dir_to_remove = os.path.join(ckpt_dir, elem)
                shutil.rmtree(dir_to_remove)
                
        