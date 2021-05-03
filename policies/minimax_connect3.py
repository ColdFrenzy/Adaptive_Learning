# -*- coding: utf-8 -*-
from ray.rllib import Policy
import random
from config.custom_config import Config
from config.connect4_config import Connect3Config
import math
import logging



def minimax_connect3(
    board,
    current_player,
    maximize,
    alpha=-math.inf,
    beta=math.inf,
    depth=Config.SEARCH_DEPTH,
    return_distr=False,
):
    """
    Minimax algorithm
    board: array
        [width x heigh] array representing the connect 4 board
    current_player: int
        can be 0 or 1 as in the connect4 environment
    maximize: Bool
        True for the player that we want to maximize, False for the one
        that we want to minimize
    depth: int
        minimax depth 
    return_distr: bool
        it only returns the value of the best action if false, otherwise returns 
        the value of every action
    RETURN: (int, int)
        next_action and score. Score is a list if return_distr is True
    """
    opponent_player = 1 - current_player

    valid_actions = available_actions(board)
    terminal_move = len(valid_actions) == 0

    if terminal_move:
        return (None, 0)
    elif depth == 0:
        # If the depth is 0 we are evaluating the position of the last player
        # not the current one. Hence if now maximize is true it means that
        # the previous player wanted to minimize and vice versa.
        if not maximize:
            return (None, score_position(board, opponent_player))
        else:
            # return (None, -score_position(board, opponent_player))
            return (None, score_position(board, current_player))

    random.shuffle(valid_actions)
    if maximize:
        value = -math.inf
        best_val = -math.inf
        # if there are more than one action with the same score
        # best_actions = []
        if return_distr:
            act_distr = {}
        for act in valid_actions:
            y = get_open_row(board, act)
            # board_copy = copy.deepcopy(board)
            drop_piece(board, current_player, act, y)
            winning_move = is_winning(board, act, y)
            if not winning_move:
                value = minimax_connect3(
                    board, opponent_player, False, alpha, beta, depth - 1, False
                )[1]
            else:
                # reward is time-scaled (for example a victory now is better
                # than a victory in 3 turns from now)
                value = 1000 * depth

            if return_distr:
                act_distr[str(act)] = value
            undo_last_move(board, act, y)
            # if value >= best_val:
            if value > best_val:
                best_val = value
                best_action = act
                # else:
                #     best_actions.append(act)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        # best_action = random.choice(best_actions)
        if return_distr:
            return best_action, act_distr

        # return best_action, best_score
        return best_action, best_val

    else:
        value = math.inf
        best_val = math.inf
        # best_actions = []
        if return_distr:
            act_distr = {}
        for act in valid_actions:
            y = get_open_row(board, act)
            # board_copy = copy.deepcopy(board)
            drop_piece(board, current_player, act, y)
            winning_move = is_winning(board, act, y)
            if not winning_move:
                value = minimax_connect3(
                    board, opponent_player, True, alpha, beta, depth - 1, False
                )[1]
            else:
                value = -1000 * depth

            if return_distr:
                act_distr[str(act)] = value
            undo_last_move(board, act, y)
            # if value <= best_val:
            if value < best_val:
                best_val = value
                best_action = act
                # else:
                #     best_actions.append(act)
            beta = min(beta, value)
            if beta <= alpha:
                break
        # best_action = random.choice(best_actions)

        if return_distr:
            return best_action, act_distr
        # return best_action, best_score
        return best_action, best_val


def undo_last_move(board, x, y, empty=Connect3Config.EMPTY):
    board[x][y] = empty


def score_position(
    board,
    player,
    width=Connect3Config.WIDTH,
    height=Connect3Config.HEIGHT,
    connect=Connect3Config.CONNECT,
):
    """
    THE HEURISITIC IS ONLY COMPUTED FOR THE LEAF VALUE
    compute the heuristic of a given board configuration. The score is based
    on the number of open lines. Forks cannot exist in connect3
    """
    score = 0
    # BLOCKING THE OPPONENT MOVES IS MORE VALUABLE THAN MAKE A GOOD MOVE
    # SINCE THE NEXT MOVE WILL BE AN OPPONENT MOVE
    opponent_player = 1 - player
    open_line_score = 80  
    opponent_open_line_score = 100
    open_lines = check_open_line(board, player)
    opponent_open_lines = check_open_line(board, opponent_player)
    score = (
        + open_lines * open_line_score
        - opponent_open_lines * opponent_open_line_score
    )
    # Heuristic function bounded to [-999,999] since a victory and a loss are
    # respectively >= 1000 and <= -1000
    if score >= 1000:
        score = 999
    elif score <= -1000:
        score = -999
    return score



def check_open_line(
    board,
    player,
    empty=Connect3Config.EMPTY,
    width=Connect3Config.WIDTH,
    height=Connect3Config.HEIGHT,
):
    """
    An open line is a serie of 3 consecutive chips from the same player,
    with an empty space on one of the extremity of the consecutive chips
    for example:
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . .X. . . . .
    . . .X. . . . .
    .O. .X.O. .O. .
    X has an open line on the third column 
    RETURN: int
        number of open lines in the current board configuration
    """
    open_lines = 0
    # HORIZONTAL OPEN LINES
    for y in range(height):
        for x in range(width - 2):
            if (
                board[x][y] == player
                and board[x + 1][y] == player
                and board[x + 2][y] == empty
            ):
                open_lines += 1
            elif (
                board[x][y] == empty
                and board[x + 1][y] == player
                and board[x + 2][y] == player
            ):
                open_lines += 1
    # VERTICAL OPEN LINES
    for y in range(height - 2):
        for x in range(width):
            if (
                board[x][y] == player
                and board[x][y + 1] == player
                and board[x][y + 2] == empty
            ):
                open_lines += 1

    # POSITIVE SLOPED DIAGONAL OPEN LINES
    for y in range(height - 2):
        for x in range(width - 2):
            if (
                board[x][y] == player
                and board[x + 1][y + 1] == player
                and board[x + 2][y + 2] == empty
            ):
                if board[x + 2][y + 1] != empty:
                    open_lines += 1
            elif (
                board[x][y] == empty
                and board[x + 1][y + 1] == player
                and board[x + 2][y + 2] == player
            ):
                if y != 0:
                    if board[x][y - 1] != empty:
                        open_lines += 1
                else:
                    open_lines += 1

    # NEGATIVE SLOPED DIAGONAL OPEN LINES
    for y in range(height - 2):
        for x in range(2, width):
            if (
                board[x][y] == player
                and board[x - 1][y + 1] == player
                and board[x - 2][y + 2] == empty
            ):
                if board[x - 2][y + 1] != empty:
                    open_lines += 1
            elif (
                board[x][y] == empty
                and board[x - 1][y + 1] == player
                and board[x - 2][y + 2] == player
            ):
                if y != 0:
                    if board[x][y - 1] != empty:
                        open_lines += 1
                else:
                    open_lines += 1
    return open_lines


def is_winning(board, x, y, connect=Connect3Config.CONNECT):
    """
    check if the current move is a winning move 
    """
    me = board[x][y]
    for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
        p = 1
        while (
            valid_location(x + p * dx, y + p * dy)
            and board[x + p * dx][y + p * dy] == me
        ):
            p += 1
        n = 1
        while (
            valid_location(x - n * dx, y - n * dy)
            and board[x - n * dx][y - n * dy] == me
        ):
            n += 1

        if p + n >= (
            connect + 1
        ):  # want (p-1) + (n-1) + 1 >= 4, or more simply p + n >- 5
            return True

    return False


def available_actions(board, width=Connect3Config.WIDTH, height=Connect3Config.HEIGHT):
    """
    check for the current available actions
    """
    return [col for col in range(width) if board[col][height - 1] == -1]


def valid_location(x, y, width=Connect3Config.WIDTH, height=Connect3Config.HEIGHT):
    return x >= 0 and x < width and y >= 0 and y < height


def get_open_row(board, x, height=Connect3Config.HEIGHT):
    """
    return the index (height) of the first position in a given column
    """
    for y in range(height):
        if board[x][y] == -1:
            return y
    return None


def drop_piece(board, current_player, x, y):
    board[x][y] = current_player


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
