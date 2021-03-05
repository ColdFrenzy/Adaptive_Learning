from ray.rllib import Policy
import random
from config.custom_config import Config
import copy
import math


class MiniMaxPolicy(Policy):
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        actual_player = Config.PLAYER_DICT[
            list(episodes[0]._agent_to_last_obs.keys())[0]
        ]

        action_space_len = self.action_space.n
        # first n elements of a single observation is the action mask
        actions = []
        width, height = Config.WIDTH, Config.HEIGHT
        for obs in obs_batch:
            # first n elements are the action mask.
            board = obs[action_space_len:]
            board = board.reshape((width, height))
            act, score = minimax(board, actual_player, True)
            actions.append(act)

        return actions, [], {}

    def learn_on_batch(self, samples):
        """No learning."""
        # return {}
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


def minimax(
    board,
    current_player,
    maximize,
    alpha=-math.inf,
    beta=math.inf,
    depth=Config.SEARCH_DEPTH,
    return_distr = False,
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
            return (None, -score_position(board, opponent_player))

    if maximize:
        value = -math.inf
        best_val = -math.inf
        # if there are more than one action with the same score
        best_actions = []
        if return_distr:
            act_distr = []
        for act in valid_actions:
            y = get_open_row(board, act)
            # board_copy = copy.deepcopy(board)
            drop_piece(board, current_player, act, y)
            winning_move = is_winning(board, act, y)
            if not winning_move:
                value = minimax(board, opponent_player, False, alpha, beta, depth - 1,False)[1]
            else:
                # reward is time-scaled (for example a victory now is better 
                # than a victory in 3 turns from now)
                value = 100000 * depth
            if return_distr:
                act_distr.append(value)
            undo_last_move(board, act, y)
            if value >= best_val:
                if value > best_val:
                    best_val = value
                    best_actions = [act]
                else:
                    best_actions.append(act)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        best_action = random.choice(best_actions)
        if return_distr:
            return best_action, act_distr

        # return best_action, best_score
        return best_action, best_val

    else:
        value = math.inf
        best_val = math.inf
        best_actions = []
        if return_distr:
            act_distr = []
        for act in valid_actions:
            y = get_open_row(board, act)
            # board_copy = copy.deepcopy(board)
            drop_piece(board, current_player, act, y)
            winning_move = is_winning(board, act, y)
            if not winning_move:
                value = minimax(board, opponent_player, True, alpha, beta, depth - 1,False)[1]
            else:
                value = -100000 * depth
            if return_distr:
                act_distr.append(value)
            beta = min(beta, value)
            undo_last_move(board, act, y)
            if value <= best_val:
                if value < best_val:
                    best_val = value
                    best_actions = [act]
                else:
                    best_actions.append(act)
            if beta <= alpha:
                break
        best_action = random.choice(best_actions)

        if return_distr:
            return best_action, act_distr
        # return best_action, best_score
        return best_action, best_val

def undo_last_move(board, x, y, empty=Config.EMPTY):
    board[x][y] = empty


def score_position(
    board, player, width=Config.WIDTH, height=Config.HEIGHT, connect=Config.CONNECT
):
    """
    compute the heuristic of a given board configuration. The score is based
    on the number of open lines and forks from both players 
    """
    score = 0
    # BLOCKING THE OPPONENT MOVES IS MORE VALUABLE THAN MAKE A GOOD MOVE
    # SINCE THE NEXT MOVE WILL BE AN OPPONENT MOVE
    opponent_player = 1 - player
    open_line_score = 5  # low score since it is easy to block with one move
    opponent_open_line_score = 100  # the score is high because if we leave an open line
    # opened the opponent will connect in the next move. This usually
    # is already predicted by increasing the depth of minimax
    fork_2_score = 10
    fork_3_score = 20
    opponent_fork_2_score = 50
    opponent_fork_3_score = 100
    fork_2, fork_3 = check_forks(board, player)
    open_lines = check_open_line(board, player)
    opponent_fork_2, opponent_fork_3 = check_forks(board, opponent_player)
    opponent_open_lines = check_open_line(board, opponent_player)
    score = (
        fork_2 * fork_2_score
        + fork_3 * fork_3_score
        + open_lines * open_line_score
        - opponent_fork_2 * opponent_fork_2_score
        - opponent_fork_3 * opponent_fork_3_score
        - opponent_open_lines * opponent_open_line_score
    )

    return score


def check_forks(
    board, player, empty=Config.EMPTY, width=Config.WIDTH, height=Config.HEIGHT
):
    """
    A fork is a serie of 2 or 3 consecutive chips of the same player,
    with empty spaces on both the extremities of the consecutive chips
    for example:
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . . . . . . .
    . . .O. . . . .
    . . .X.X. .O. .
    X has a fork of length 4 on the first row
    RETURN: (int, int)
        number of forks of length 4 and number of forks of length 5 in the 
        actual board configuration
    """
    fork_2 = 0
    fork_3 = 0
    # HORIZONTAL FORKS
    # LENGTH 4 FORKS
    for y in range(height):
        for x in range(width - 3):
            if (
                board[x][y] == empty
                and board[x + 1][y] == player
                and board[x + 2][y] == player
                and board[x + 3][y] == empty
            ):
                # We also need to check if the 2 empty side are valid
                # this means that there should be a chip below the empty spots
                if y != 0:
                    if board[x][y - 1] != empty and board[x + 3][y - 1] != empty:
                        fork_2 += 1
                else:
                    fork_2 += 1

    # LEN 5 FORKS
    for y in range(height):
        for x in range(width - 4):
            if (
                board[x][y] == empty
                and board[x + 1][y] == player
                and board[x + 2][y] == player
                and board[x + 3][y] == player
                and board[x + 4][y] == empty
            ):
                if y != 0:
                    if board[x][y - 1] != empty and board[x + 4][y - 1] != empty:
                        fork_3 += 1
                else:
                    fork_3 += 1

    # VERTICAL FORKS DO NOT EXIST

    # POSITIVE SLOPED DIAGONAL FORKS
    # LENGTH 4 FORKS
    for y in range(height - 3):
        for x in range(width - 3):
            if (
                board[x][y] == empty
                and board[x + 1][y + 1] == player
                and board[x + 2][y + 2] == player
                and board[x + 3][y + 3] == empty
            ):
                # We also need to check if the 2 empty side are valid
                # this means that there is a chip below the empty column
                if y != 0:
                    if board[x][y - 1] != empty and board[x + 3][y + 2] != empty:
                        fork_2 += 1
                elif board[x + 3][y + 2] != empty:
                    fork_2 += 1
    # LENGTH 5 FORKS
    for y in range(height - 4):
        for x in range(width - 4):
            if (
                board[x][y] == empty
                and board[x + 1][y + 1] == player
                and board[x + 2][y + 2] == player
                and board[x + 3][y + 3] == player
                and board[x + 4][y + 4] == empty
            ):
                # We also need to check if the 2 empty side are valid
                # this means that there is a chip below the empty column
                if y != 0:
                    if board[x][y - 1] != empty and board[x + 4][y + 3] != empty:
                        fork_3 += 1
                elif board[x + 4][y + 3] != empty:
                    fork_3 += 1

    # NEGATIVE SLOPED DIAGONAL FORKS
    # LENGTH 4 FORKS
    for y in range(height - 3):
        for x in range(3, width):
            if (
                board[x][y] == empty
                and board[x - 1][y + 1] == player
                and board[x - 2][y + 2] == player
                and board[x - 3][y + 3] == empty
            ):
                if y != 0:
                    if board[x - 3][y + 2] != empty and board[x][y - 1] != empty:
                        fork_2 += 1
                elif board[x - 3][y + 2] != empty:
                    fork_2 += 1
    # LENGTH 5 FORKS
    for y in range(height - 4):
        for x in range(4, width):
            if (
                board[x][y] == empty
                and board[x - 1][y + 1] == player
                and board[x - 2][y + 2] == player
                and board[x - 3][y + 3] == player
                and board[x - 4][y + 4] == empty
            ):
                if y != 0:
                    if board[x - 4][y + 3] != empty and board[x][y - 1] != empty:
                        fork_3 += 1
                elif board[x - 4][y + 3] != empty:
                    fork_3 += 1

    return fork_2, fork_3


def check_open_line(
    board, player, empty=Config.EMPTY, width=Config.WIDTH, height=Config.HEIGHT
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
        for x in range(width - 3):
            if (
                board[x][y] == player
                and board[x + 1][y] == player
                and board[x + 2][y] == player
                and board[x + 3][y] == empty
            ):
                open_lines += 1
            elif (
                board[x][y] == empty
                and board[x + 1][y] == player
                and board[x + 2][y] == player
                and board[x + 3][y] == player
            ):
                open_lines += 1
    # VERTICAL OPEN LINES
    for y in range(height - 3):
        for x in range(width):
            if (
                board[x][y] == player
                and board[x][y + 1] == player
                and board[x][y + 2] == player
                and board[x][y + 3] == empty
            ):
                open_lines += 1

    # POSITIVE SLOPED DIAGONAL OPEN LINES
    for y in range(height - 3):
        for x in range(width - 3):
            if (
                board[x][y] == player
                and board[x + 1][y + 1] == player
                and board[x + 2][y + 2] == player
                and board[x + 3][y + 3] == empty
            ):
                if board[x + 3][y + 2] != empty:
                    open_lines += 1
            elif (
                board[x][y] == empty
                and board[x + 1][y + 1] == player
                and board[x + 2][y + 2] == player
                and board[x + 3][y + 3] == player
            ):
                if y != 0:
                    if board[x][y - 1] != empty:
                        open_lines += 1
                else:
                    open_lines += 1

    # NEGATIVE SLOPED DIAGONAL OPEN LINES
    for y in range(height - 3):
        for x in range(3, width):
            if (
                board[x][y] == player
                and board[x - 1][y + 1] == player
                and board[x - 2][y + 2] == player
                and board[x - 3][y + 3] == empty
            ):
                if board[x - 3][y + 2] != empty:
                    open_lines += 1
            elif (
                board[x][y] == empty
                and board[x - 1][y + 1] == player
                and board[x - 2][y + 2] == player
                and board[x - 3][y + 3] == player
            ):
                if y != 0:
                    if board[x][y - 1] != empty:
                        open_lines += 1
                else:
                    open_lines += 1
    return open_lines


def is_winning(board, x, y, connect=Config.CONNECT):
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


def available_actions(board, width=Config.WIDTH, height=Config.HEIGHT):
    """
    check for the current available actions
    """
    return [col for col in range(width) if board[col][height - 1] == -1]


def valid_location(x, y, width=Config.WIDTH, height=Config.HEIGHT):
    return x >= 0 and x < width and y >= 0 and y < height


def get_open_row(board, x, height=Config.HEIGHT):
    """
    return the index (height) of the first position in a given column
    """
    for y in range(height):
        if board[x][y] == -1:
            return y
    return None


def drop_piece(board, current_player, x, y):
    board[x][y] = current_player
