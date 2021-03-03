import sys
import os
import random
import numpy as np

sys.path.insert(1, os.path.abspath(os.pardir))

from tqdm import tqdm
from config.custom_config import Config

# from policies.random_policy import iniMax
from policies.minimax_policy import minimax
from env.connect4_multiagent_env import Connect4Env

# =============================================================================
# algorithms to rank:
# random, minimax with deep 1,minimax with deep 2,minimax with deep 3,
# minimax with deep 4, minimax with deep 5
# =============================================================================


def minimax_vs_random_elo(depth, number_of_games):
    """
    Use the inverse of the elo formula to compute the outcome of a match,
    given that we already know the result of a match.
    It tests the relative elo between random algorithm and a minimax algorithm
    
    Elo formula:
        expected_score = 1/(1+10^((elo_diff)/400)) 
    Inverse formula: 
        elo_diff = -400*log(1/score - 1)
    
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
    for i in range(number_of_games):
        game.reset()
        game_over = False
        actions = {}
        while not game_over:
            actual_player = game.current_player
            board = game.board
            if actual_player == player1_ID:
                act, _ = minimax(board, player1_ID, True, depth)
                actions[player1] = act
                _, _, done, _ = game.step(actions)
            else:
                actions[player2] = random.choice(game.get_moves(False))
                _, _, done, _ = game.step(actions)

            if done["__all__"]:
                game_over = True

    score = abs(game.score[player1] - game.score[player2])
    # print("Difference in score between minimax of depth 1 and random action is " + str(score))
    # print("player 1 score: " + str(game.score[player1]))
    # print("player 2 score: " + str(game.score[player2]))
    elo_diff = -400 * (1 / score - 1)
    # print("elo difference computed over " + str(number_of_games) + \
    #       " between the 2 algortithms is " + str(elo_diff))
    return elo_diff


def minimax_vs_minimax_elo(depth1, depth2, number_of_games):
    """
    Use the inverse of the elo formula to compute the outcome of a match,
    given that we already know the result of a match.
    It tests the relative elo between minimax algorithm and a minimax algorithm
    
    Elo formula:
        expected_score = 1/(1+10^((elo_diff)/400)) 
    Inverse formula: 
        elo_diff = -400*log(1/score - 1)
    """
    game = Connect4Env(None)
    for i in tqdm(range(number_of_games)):
        game.reset()
        game_over = False
        actions = {}
        while not game_over:
            actual_player = game.current_player
            board = game.board
            if actual_player == player1_ID:
                act, _ = minimax(board, player1_ID, True, depth1)
                actions[player1] = act
                _, _, done, _ = game.step(actions)
            else:
                act, _ = minimax(board, player2_ID, True, depth2)
                actions[player2] = act
                _, _, done, _ = game.step(actions)

            if done["__all__"]:
                game_over = True

    score = abs(game.score[player1] - game.score[player2])
    print(
        "Difference in score between minimax of depth 1 and minimax of depth 2 is "
        + str(score)
    )
    print("player 1 score: " + str(game.score[player1]))
    print("player 2 score: " + str(game.score[player2]))
    elo_diff = -400 * (1 / score - 1)
    print(
        "elo difference computed over "
        + str(number_of_games)
        + " between the 2 algortithms is "
        + str(elo_diff)
    )

    return elo_diff


def model_vs_minimax(model, checkpoint, elo, number_of_games):
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
    pass


def elo_stats(elo_vector):
    """
    compute mean and variance of a vector.
    """
    return np.mean(elo_vector), np.std(elo_vector) ** 2


if __name__ == "__main__":
    number_of_games = 100
    depth = 1
    player1 = Config.PLAYER1
    player2 = Config.PLAYER2
    player1_ID = Config.PLAYER1_ID
    player2_ID = Config.PLAYER2_ID
    minimax_vs_minimax_elo(2, 1, number_of_games)
