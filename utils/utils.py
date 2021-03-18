from datetime import datetime
import tempfile
import os
from typing import Type
from ray.tune.logger import UnifiedLogger
from ray.rllib.agents.trainer import Trainer


def select_policy(agent_id):

    if agent_id == "player1":
        return "player1"
    # to avoid overfitting over a single strategy, we keep 3 networks trained
    # independently
    else:
        return "player2"
        # return random.choice(["player2_1","player2_2","player2_3"])


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
