import sys
import os
import random
import numpy as np
import math
import logging

sys.path.insert(1, os.path.abspath(os.pardir))

from tqdm import tqdm
from config.custom_config import Config
from config.connect4_config import Connect4Config
import copy

# from policies.random_policy import iniMax
from policies.minimax_policy import minimax
from env.connect4_multiagent_env import Connect4Env

player1 = Connect4Config.PLAYER1
player2 = Connect4Config.PLAYER2
player1_ID = Connect4Config.PLAYER1_ID
player2_ID = Connect4Config.PLAYER2_ID

# =============================================================================
# algorithms to rank:
# random, minimax with deep 1,minimax with deep 2,minimax with deep 3,
# minimax with deep 4, minimax with deep 5
# =============================================================================


def minimax_vs_random_elo(
    depth, number_of_games, logger,
):
    """
    Use the inverse of the elo formula to compute the outcome of a match,
    given that we already know the result of a match.
    It tests the relative elo between random algorithm and a minimax algorithm
    
    Elo formula:
        expected_score = 1/(1+10^((elo_diff)/400)) 
    Inverse formula: 
        elo_diff = -400*log(1/expected_score - 1)
    
    INPUT:
    depth: int
        number of game to play before updating the elo
    number_of_games: int
        number of game to play before updating the elo
    RETURN:
    elo_diff: float
        elo difference
    """
    game = Connect4Env(None)
    if logger:
        logger.info(
            "**********MINIMAX_depth_"
            + str(depth)
            + "_(X) VS (O)_RANDOM"
            + "**********"
        )
    for i in tqdm(range(number_of_games)):
        game_over = False
        actions = {}
        starting_player = random.choice([player1_ID, player2_ID])
        game.reset(starting_player=starting_player)
        print("\nPlayer " + str(starting_player + 1) + " is starting")
        while not game_over:
            actual_player = game.current_player
            board = game.board
            if actual_player == player1_ID:
                act, action_values = minimax(
                    board, player1_ID, True, depth=depth, return_distr=True
                )
                actions[player1] = act
                _, _, done, _ = game.step(actions)
            elif actual_player == player2_ID:
                act = random.choice(game.get_moves(False))
                actions[player2] = act
                _, _, done, _ = game.step(actions)
            else:
                raise ValueError("Player index is not valid, should be 0 or 1")
            if logger:
                logger.info("Game number " + str(i) + "/" + str(number_of_games))
                if actual_player == player1_ID:
                    logger.info("action distribution: " + str(action_values))
                logger.info(
                    "Player " + str(actual_player + 1) + " actions: " + str(act)
                )
                logger.info("\n" + repr(board))
                logger.info(board_print(board))

            if done["__all__"]:
                logger.info("PLAYER " + str(game.winner + 1) + " WON...")
                logger.info(
                    "CURRENT SCORE: "
                    + str(game.score[player1])
                    + " VS "
                    + str(game.score[player2])
                )
                game_over = True

    if game.score[player1] > game.score[player2]:
        score = game.score[player1] / number_of_games + game.num_draws / (
            2 * number_of_games
        )
    elif game.score[player1] < game.score[player2]:
        score = game.score[player2] / number_of_games + game.num_draws / (
            2 * number_of_games
        )
    elif game.score[player1] == game.score[player2]:
        return 0

    if score >= 10 / 11:
        elo_diff = 400
    else:
        elo_diff = -400 * math.log((1 / score - 1), 10)

    print("\nplayer 1 score: " + str(game.score[player1]))
    print("player 2 score: " + str(game.score[player2]))
    print("number of draw: " + str(game.num_draws))
    print(
        "elo difference computed over "
        + str(number_of_games)
        + " between the 2 algortithms is "
        + str(elo_diff)
    )

    return elo_diff


def minimax_vs_minimax_elo(depth1, depth2, number_of_games, logger=None):
    """
    Use the inverse of the elo formula to compute the outcome of a match,
    given that we already know the result of a match.
    It tests the relative elo between minimax algorithm and a minimax algorithm
    
    Elo formula:
        expected_score = 1/(1+10^((elo_diff)/400)) 
    Inverse formula: 
        elo_diff = -400*log(1/expected_score - 1)
    """
    game = Connect4Env(None)
    if logger:
        logger.info(
            "**********MINIMAX_depth_"
            + str(depth1)
            + "_(X) VS (O)_MINIMAX_depth_"
            + str(depth2)
            + "**********"
        )
    for i in tqdm(range(number_of_games)):
        game_over = False
        actions = {}
        starting_player = random.choice([player1_ID, player2_ID])
        game.reset(starting_player=starting_player)
        print("\nPlayer " + str(starting_player + 1) + " is starting")
        while not game_over:
            actual_player = game.current_player
            board = game.board
            if actual_player == player1_ID:
                act, _ = minimax(board, player1_ID, True, depth=depth1)
                actions[player1] = act
                _, _, done, _ = game.step(actions)
            elif actual_player == player2_ID:
                act, _ = minimax(board, player2_ID, True, depth=depth2)
                actions[player2] = act
                _, _, done, _ = game.step(actions)
            else:
                raise ValueError("Player index is not valid, should be 0 or 1")
            if logger:
                logger.info("Game number " + str(i) + "/" + str(number_of_games))
                logger.info(
                    "Player " + str(actual_player + 1) + " actions: " + str(act)
                )
                logger.info("\n" + repr(board))
                logger.info(board_print(board))

            if done["__all__"]:
                logger.info("PLAYER " + str(game.winner + 1) + " WON...")
                logger.info(
                    "CURRENT SCORE: "
                    + str(game.score[player1])
                    + " VS "
                    + str(game.score[player2])
                )
                game_over = True

    if game.score[player1] > game.score[player2]:
        score = game.score[player1] / number_of_games + game.num_draws / (
            2 * number_of_games
        )
    elif game.score[player1] < game.score[player2]:
        score = game.score[player2] / number_of_games + game.num_draws / (
            2 * number_of_games
        )
    elif game.score[player1] == game.score[player2]:
        return 0

    if score >= 10 / 11:
        elo_diff = 400
    else:
        elo_diff = -400 * math.log((1 / score - 1), 10)

    print("\nplayer 1 score: " + str(game.score[player1]))
    print("player 2 score: " + str(game.score[player2]))
    print("number of draw: " + str(game.num_draws))
    print(
        "elo difference computed over "
        + str(number_of_games)
        + " between the 2 algortithms is "
        + str(elo_diff)
    )

    return elo_diff


def model_vs_minimax(
    model, depth, number_of_games, checkpoint=None, logger=None, randomize=True
):
    """
    Use the inverse of the elo formula to compute the outcome of a match,
    given that we already know the result of a match.
    It tests the relative elo between a custom model and a minimax algorithm
    
    Elo formula:
        expected_score = 1/(1+10^((elo_diff)/400)) 
    Inverse formula: 
        elo_diff = -400*log(1/score - 1)
    INPUT:
    model: 
        tensorflow model
    checkpoint:
        path to chekpoint of the model to use
    """

    game = Connect4Env(None)
    model_name = model.name
    if logger:
        logger.info(
            "**********"
            + str(model_name)
            + "_(X) VS (O)_MINIMAX_depth_"
            + str(depth)
            + "**********"
        )
    print("Starting Evaluation")
    for i in range(number_of_games):
        game_over = False
        actions = {}
        if randomize:
            starting_player = random.choice([player1_ID, player2_ID])
        else:
            starting_player = player1_ID
        game.reset(starting_player=starting_player)
        while not game_over:
            actual_player = game.current_player
            board = game.board
            if actual_player == player1_ID:
                input_dict = {"obs": {}}
                # if model.use_conv:
                #     reshaped_board = np.expand_dims(board, axis=(0,-1))

                # else:
                #     reshaped_board = np.reshape(board, (1, board.shape[0] * board.shape[1]))
                action_mask = game.get_moves(True)
                input_dict["obs"]["state"] = board #reshaped_board
                input_dict["obs"]["action_mask"] = action_mask
                action_logits, _ = model.forward(input_dict, None, None)
                # max_act = max(action_logits[0])
                act = np.argmax(action_logits[0])
                # act = [i for i, j in enumerate(action_logits[0]) if j == max_act]
                # act = random.choice(act)
                actions[player1] = act
                _, _, done, _ = game.step(actions)
            elif actual_player == player2_ID:
                act, _ = minimax(board, player2_ID, True, depth=depth)
                actions[player2] = act
                _, _, done, _ = game.step(actions)
            else:
                raise ValueError("Player index is not valid, should be 0 or 1")
            if logger:
                logger.info("Game number " + str(i) + "/" + str(number_of_games))
                logger.info(
                    "Player " + str(actual_player + 1) + " actions: " + str(act)
                )
                logger.info("\n" + repr(board))
                logger.info(board_print(board))

            if done["__all__"]:
                logger.info("PLAYER " + str(game.winner + 1) + " WON...")
                logger.info(
                    "CURRENT SCORE: "
                    + str(game.score[player1])
                    + " VS "
                    + str(game.score[player2])
                )
                game_over = True

    # score = game.score[player1] / number_of_games + game.num_draws / (
    #     2 * number_of_games
    # )
    print("Evaluation Over")

    elo_diff = compute_elo_difference(
        game.score[player1], game.num_draws, number_of_games
    )

    # print("\nplayer 1 score: " + str(game.score[player1]))
    # print("player 2 score: " + str(game.score[player2]))
    # print("number of draw: " + str(game.num_draws))
    # print(
    #     "elo difference computed over "
    #     + str(number_of_games)
    #     + " between the 2 algortithms is "
    #     + str(elo_diff)
    # )

    return elo_diff, game.score[player1], game.score[player2], game.num_draws


def model_vs_minimax_stochastic(
    model, depth, number_of_games, checkpoint=None, logger=None, randomize=True
):
    """
        used to evaluate the model against the minimax algorithm.
        The action chosen by the model is picked following the distribution
        of the action probabilities. 
    
    """

    game = Connect4Env(None)
    model_name = model.name
    if logger:
        logger.info(
            "**********"
            + str(model_name)
            + "_(X) VS (O)_MINIMAX_depth_"
            + str(depth)
            + "**********"
        )
    print("Starting Evaluation")
    for i in range(number_of_games):
        game_over = False
        actions = {}
        if randomize:
            starting_player = random.choice([player1_ID, player2_ID])
        else:
            starting_player = player1_ID
        game.reset(starting_player=starting_player, randomize = False)
        while not game_over:
            actual_player = game.current_player
            board = game.board
            if actual_player == player1_ID:
                input_dict = {"obs": {}}
                # if model.use_conv:
                #     reshaped_board = np.expand_dims(board, axis=(0,-1))

                # else:
                #     reshaped_board = np.reshape(board, (1, board.shape[0] * board.shape[1]))
                action_mask = game.get_moves(True)
                input_dict["obs"]["state"] = board #reshaped_board
                input_dict["obs"]["action_mask"] = action_mask
                action_logits, _ = model.forward(input_dict, None, None)
                # max_act = max(action_logits[0])
                act = np.argmax(action_logits[0])
                # act = [i for i, j in enumerate(action_logits[0]) if j == max_act]
                # act = random.choice(act)
                actions[player1] = act
                _, _, done, _ = game.step(actions)
            elif actual_player == player2_ID:
                act, _ = minimax(board, player2_ID, True, depth=depth)
                actions[player2] = act
                _, _, done, _ = game.step(actions)
            else:
                raise ValueError("Player index is not valid, should be 0 or 1")
            if logger:
                logger.info("Game number " + str(i) + "/" + str(number_of_games))
                logger.info(
                    "Player " + str(actual_player + 1) + " actions: " + str(act)
                )
                logger.info("\n" + repr(board))
                logger.info(board_print(board))

            if done["__all__"]:
                if logger:
                    logger.info("PLAYER " + str(game.winner + 1) + " WON...")
                    logger.info(
                        "CURRENT SCORE: "
                        + str(game.score[player1])
                        + " VS "
                        + str(game.score[player2])
                    )
                game_over = True

    # score = game.score[player1] / number_of_games + game.num_draws / (
    #     2 * number_of_games
    # )
    print("Evaluation Over")

    elo_diff = compute_elo_difference(
        game.score[player1], game.num_draws, number_of_games
    )

    # print("\nplayer 1 score: " + str(game.score[player1]))
    # print("player 2 score: " + str(game.score[player2]))
    # print("number of draw: " + str(game.num_draws))
    # print(
    #     "elo difference computed over "
    #     + str(number_of_games)
    #     + " between the 2 algortithms is "
    #     + str(elo_diff)
    # )

    return elo_diff, game.score[player1], game.score[player2], game.num_draws



def model_vs_model(model1,model2, number_of_games,discount_rate,randomize=True):
    """
    Compute the results of different games between 2 models 

    Parameters
    ----------
    model1 : Neural Network model
    model2 : Neural Network model
    number_of_games : int
        number of games to play 
    discount_rate : float
        scale the final reward 
    randomize : Bool
        decide if the first player to move is decided randomly

    Returns
    -------
    """
    game = Connect4Env(None)
    # how long a game last 
    final_value = 0
    p1_win = 0 
    for i in range(number_of_games):
        timestep = 0
        reward = 0
        game_over = False
        actions = {}
        if randomize:
            starting_player = random.choice([player1_ID, player2_ID])
        else:
            starting_player = player1_ID
        game.reset(starting_player=starting_player)
        while not game_over:
            timestep += 1
            actual_player = game.current_player
            board = game.board
            board_p2 = game.board_p2
            if actual_player == player1_ID:
                input_dict = {"obs": {}}
                # already handeled inside the model
                # if model1.use_conv:
                #     reshaped_board = np.expand_dims(board, axis=(0,-1))

                # else:
                #     reshaped_board = np.reshape(board, (1, board.shape[0] * board.shape[1]))
                action_mask = game.get_moves(True)
                input_dict["obs"]["state"] = board
                input_dict["obs"]["action_mask"] = action_mask
                action_logits, _ = model1.forward(input_dict, None, None)
                # max_act = max(action_logits[0])
                act = np.argmax(action_logits[0])
                # act = [i for i, j in enumerate(action_logits[0]) if j == max_act]
                # act = random.choice(act)
                actions[player1] = act
                _, rew, done, _ = game.step(actions)
            elif actual_player == player2_ID:
                input_dict = {"obs": {}}
                # if model2.use_conv:
                #     reshaped_board = np.expand_dims(board_p2, axis=(0,-1))

                # else:
                #     reshaped_board = np.reshape(board_p2, (1, board_p2.shape[0] * board_p2.shape[1]))
                action_mask = game.get_moves(True)
                input_dict["obs"]["state"] = board_p2 #reshaped_board
                input_dict["obs"]["action_mask"] = action_mask
                action_logits, _ = model2.forward(input_dict, None, None)
                # max_act = max(action_logits[0])
                act = np.argmax(action_logits[0])
                # act = [i for i, j in enumerate(action_logits[0]) if j == max_act]
                # act = random.choice(act)
                actions[player2] = act
                _, rew, done, _ = game.step(actions)
                
            else:
                raise ValueError("Player index is not valid, should be 0 or 1")
            if done["__all__"]:
                game_over = True
                reward = rew["player2"]
                if rew["player1"] == 1.0:
                    p1_win += 1.0

                if discount_rate == 1:
                    final_value += reward
                else:
                    final_value += (discount_rate**timestep)*reward
                
    final_value = final_value / number_of_games
    p1_win_rate = p1_win/number_of_games
        
    return final_value, p1_win_rate 

def model_vs_model_stochastic(model1,model2, number_of_games,discount_rate,randomize=True):
    """
    Compute the results of different games between 2 models. Instead of picking
    the best action, we pick an action with probability equals to the action 
    probability

    Parameters
    ----------
    model1 : Neural Network model
    model2 : Neural Network model
    number_of_games : int
        number of games to play 
    discount_rate : float
        scale the final reward 
    randomize : Bool
        decide if the first player to move is decided randomly

    Returns
    -------
    """
    game = Connect4Env(None)
    # how long a game last 
    final_value = 0
    p1_win = 0 
    for i in range(number_of_games):
        timestep = 0
        reward = 0
        game_over = False
        actions = {}
        if randomize:
            starting_player = random.choice([player1_ID, player2_ID])
        else:
            starting_player = player1_ID
        game.reset(starting_player=starting_player)
        while not game_over:
            timestep += 1
            actual_player = game.current_player
            board = game.board
            board_p2 = game.board_p2
            if actual_player == player1_ID:
                input_dict = {"obs": {}}
                # if model1.use_conv:
                #     reshaped_board = np.expand_dims(board, axis=(0,-1))

                # else:
                #     reshaped_board = np.reshape(board, (1, board.shape[0] * board.shape[1]))
                action_mask = game.get_moves(True)
                input_dict["obs"]["state"] = board #reshaped_board
                input_dict["obs"]["action_mask"] = action_mask
                action_logits, _ = model1.forward(input_dict, None, None)
                action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                act = np.random.choice([0,1,2,3,4,5,6],1,p=action_prob)[0]
                #act = np.argmax(action_logits[0])

                actions[player1] = act
                _, rew, done, _ = game.step(actions)
            elif actual_player == player2_ID:
                input_dict = {"obs": {}}
                # if model2.use_conv:
                #     reshaped_board = np.expand_dims(board_p2, axis=(0,-1))

                # else:
                #     reshaped_board = np.reshape(board_p2, (1, board_p2.shape[0] * board_p2.shape[1]))
                action_mask = game.get_moves(True)
                input_dict["obs"]["state"] = board_p2 #reshaped_board
                input_dict["obs"]["action_mask"] = action_mask
                action_logits, _ = model2.forward(input_dict, None, None)
                action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                act = np.random.choice([0,1,2,3,4,5,6],1,p=action_prob)[0]
                actions[player2] = act
                _, rew, done, _ = game.step(actions)
            else:
                raise ValueError("Player index is not valid, should be 0 or 1")

            if done["__all__"]:
                game_over = True
                reward = rew["player2"]
                if rew["player1"] == 1.0:
                    p1_win += 1.0

                if discount_rate == 1:
                    final_value += reward
                else:
                    final_value += (discount_rate**timestep)*reward
                
    final_value = final_value / number_of_games
    p1_win_rate = p1_win/number_of_games
        
    return final_value, p1_win_rate 

   

                
def compute_win_rate_matrix(weights, model1,model2 ):
    win_rate_matrix = []
    w1_to_restore = model1.base_model.get_weights()
    w2_to_restore = model2.base_model.get_weights()
    # the win rate matrix is antisymmetric
    for n1,p1 in enumerate(weights):
        win_rate_matrix.append([])
        model1.base_model.set_weights(p1)
        for n2,p2 in enumerate(weights):
            if n2 < n1:
                val = 1 - win_rate_matrix[n2][n1]
                win_rate_matrix[n1].append(val)
                continue
            model2.base_model.set_weights(p2)
            _, p1_win_rate = model_vs_model_stochastic(model1,model2, 50 , 1,randomize=True)
            win_rate_matrix[n1].append(p1_win_rate)
            
    model1.base_model.set_weights(w1_to_restore)
    model2.base_model.set_weights(w2_to_restore)
    return win_rate_matrix 


        


             
def compute_elo_difference(player1_score, draws, number_of_games):

    score = (player1_score / number_of_games) + draws / (2 * number_of_games)
    if score >= 10 / 11:
        elo_diff = 400
    elif score == 0 or score <= 1 / 11:
        elo_diff = -400
    else:
        elo_diff = -400 * math.log((1 / score - 1), 10)

    return elo_diff


def board_print(board, height=Connect4Config.HEIGHT, width=Connect4Config.WIDTH):
    """
    Return current game status as class representation
    """
    s = ""
    s += "\n"
    for x in range(height - 1, -1, -1):
        for y in range(width):
            s += {-1: ".", 0: "X", 1: "O",}[board[y][x]]
            s += " "
        s += "\n"
    return s


def init_logger(file_path):
    logger = logging.getLogger("MinimaxLogger")

    f_handler = logging.FileHandler(file_path, "w", "utf-8")
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.WARN)
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)

    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    logger.setLevel(logging.DEBUG)

    return logger




if __name__ == "__main__":
    # We put a cap over the maximum elo difference between 2 players. A player
    # is stronger than another with a difference in elo of 400 if it is able to
    # win 10 games over 11 (10/11), while it is stronger than another player
    # with an elo difference of 800 if it beats the enemy with at 100 victories
    # over 101 games and so on. In order to stop this from growing toward infinity,
    # we cap the maximum difference in elo between 2 player as 400 and we compute
    minimax_logger = init_logger("../log/minimax_vs_minimax.log")
    random_logger = init_logger("../log/minimax_vs_random.log")
    number_of_games = 50
    depth = 1

    # elo = minimax_vs_random_elo(2, number_of_games, random_logger)
    elo = minimax_vs_minimax_elo(4, 2, number_of_games, minimax_logger)
