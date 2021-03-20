import os
import numpy as np

import ray
from ray.rllib.agents.ppo import PPOTrainer
from config.custom_config import Config
from config.trainer_config import TrainerConfig
from config.connect4_config import Connect4Config
from ray.rllib.agents.trainer import with_common_config
from models import Connect4ActionMaskModel

ckpt_dir = Config.CKPT_DIR


if __name__ == "__main__":
    _ = Connect4ActionMaskModel
    # ray.init(ignore_reinit_error=True)

    new_config = with_common_config(TrainerConfig.PPO_TRAINER)
    trainer = PPOTrainer(config=new_config)

    best_ckpt = 1

    ckpt_to_restore = None
    # Restore the latest checkpoint if exist:
    for ckpt in os.listdir(ckpt_dir):
        if ckpt == ".gitkeep":
            continue
        ckpt_indx = int(ckpt.split("_")[1])
        if ckpt_indx > best_ckpt:
            best_ckpt = ckpt_indx
    if best_ckpt > 1:
        ckpt_to_restore = os.path.join(
            ckpt_dir, "checkpoint_" + str(best_ckpt), "checkpoint-" + str(best_ckpt)
        )
        trainer.restore(ckpt_to_restore)
        print("Checkpoint number " + str(best_ckpt) + " restored")
    else:
        print("No checkpoint found, Training starting from scratch...")

    # Serving and training loop
    env = trainer.env_creator({})
    # obs_state = {}
    # obs_state["obs"] = obs[list(obs.keys())[0]]
    player1 = Connect4Config.PLAYER1
    player1_id = Connect4Config.PLAYER1_ID
    player2 = Connect4Config.PLAYER2
    player2_id = Connect4Config.PLAYER2_ID
    actual_player = player1
    actual_player_id = player1_id
    obs = env.reset(player1_id)
    obs = {"obs": obs[actual_player]}
    action_dict = {}
    while True:
        # action, state, info_trainer = trainer.get_policy(actual_player).compute_single_action(obs)#compute_action(obs[actual_player],policy_id=actual_player,explore=False)#, full_fetch=True)
        action_logits, _ = trainer.get_policy(actual_player).model.forward(
            obs, None, None
        )
        action = np.argmax(action_logits[0])
        action_dict = {actual_player: action}
        print(
            "Player " + str(actual_player_id + 1) + " picked column: " + str(action + 1)
        )
        obs, reward, done, info = env.step(action_dict)
        print(env)
        if done["__all__"]:
            print("Player " + str(actual_player_id + 1) + " WON!!!!!!")
            obs = env.reset()
            break
        if actual_player == player1:
            actual_player = player2
            actual_player_id = player2_id
        else:
            actual_player = player1
            actual_player_id = player1_id
        obs = {"obs": obs[actual_player]}

    # ray.shutdown()
