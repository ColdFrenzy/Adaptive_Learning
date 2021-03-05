import numpy as np
from colorama import Fore
from gym.spaces import Box, Dict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from config.custom_config import Config


class Connect4Env(MultiAgentEnv):
    """
    GameState for the Connect 4 game.
    The board is represented as a 2D array (rows and columns).
    Each entry on the array can be:
        -1 = empty    (.)
        0 = player 1 (X)
        1 = player 2 (O)
    from the player 1 point of view, while it's:
        -1 = empty   (.)
        0 = player 2 (0)
        1 = player 1 (X)
    from the player 2 point of view

        Winner can be:
             None = No winner (yet)
            -1 = Draw
             0 = player 1 (X)
             1 = player 2 (O) 
        
        Action Space:
            0 to width-1
    """

    def __init__(
        self,
        env_context,
        width=Config.WIDTH,
        height=Config.HEIGHT,
        n_actions=Config.N_ACTIONS,
        connect=Config.CONNECT,
    ):

        self.width = width
        self.height = height
        self.n_actions = n_actions
        self.connect = connect
        self.player1 = Config.PLAYER1
        self.player2 = Config.PLAYER2
        self.player1_ID = Config.PLAYER1_ID
        self.player2_ID = Config.PLAYER2_ID
        # observation_space needs to include action masking
        self.observation_space = Dict(
            {
                "state": Box(low=-1, high=1, shape=(width, height), dtype=np.float32),
                "action_mask": Box(
                    low=0.0, high=1.0, shape=(n_actions,), dtype=np.float32
                ),
            }
        )
        self.action_space = Discrete(n_actions)

        self.score = {self.player1: 0, self.player2: 0}
        self.num_draws = 0
        self.num_moves = 0
        self.reset()

    def reset(self, starting_player=Config.PLAYER1_ID):
        """
        Initialises the Connect 4 gameboard and return observations
        """
        assert starting_player in [
            self.player1_ID,
            self.player2_ID,
        ], "starting player value is not valid"
        self.board = np.full((self.width, self.height), -1, dtype=np.float32)
        # board seen from player 2 point of view
        self.board_p2 = np.full((self.width, self.height), -1, dtype=np.float32)

        self.current_player = starting_player
        self.num_moves = 0
        self.winner = None
        if self.current_player == self.player1_ID:
            return {self.player1: self.get_player_observations(self.player1)}
        else:
            return {self.player2: self.get_player_observations(self.player2)}

    def get_player_observations(self, player):
        if player == self.player1_ID:
            obs = {"state": self.board, "action_mask": self.get_moves()}
        else:
            obs = {"state": self.board_p2, "action_mask": self.get_moves()}
        return obs

    def clone(self):
        """
        Creates a deep copy of the game state.
        NOTE: it is _really_ important that a copy is used during simulations
              Because otherwise MCTS would be operating on the real game board.
        :returns: deep copy of this GameState
        """
        st = Connect4Env(width=self.width, height=self.height)
        st.current_player = self.current_player
        st.winner = self.winner
        st.board = np.array([self.board[col][:] for col in range(self.width)])
        st.board_p2 = np.array([self.board_p2[col][:] for col in range(self.width)])
        return st

    def step(self, action_dict):
        """
        Changes this GameState by "dropping" a chip in the column
        specified by param movecol.
        :param movecol: column over which a chip will be dropped
        """
        # self.reset_score()
        if self.current_player == self.player1_ID:
            act = action_dict[self.player1]
        else:
            act = action_dict[self.player2]

        if not (
            act >= 0 and act <= self.width and self.board[act][self.height - 1] == -1
        ):
            raise IndexError(
                f"Invalid move. tried to place a chip on column {act} which is already full. Valid moves are: {self.get_moves(mask=False)}"
            )
        row = self.height - 1
        while row >= 0 and self.board[act][row] == -1:
            row -= 1

        row += 1

        self.board[act][row] = self.current_player
        self.board_p2[act][row] = 1 - self.current_player

        self.num_moves += 1
        self.winner, reward_vector = self.check_for_episode_termination(act, row)

        single_info = {}
        done = {"__all__": self.winner is not None}
        p1_obs = self.get_player_observations(self.player1)
        p2_obs = self.get_player_observations(self.player2)

        if done["__all__"] == True:
            obs = {self.player1: p1_obs, self.player2: p2_obs}
            reward = {self.player1: reward_vector[0], self.player2: reward_vector[1]}
            info = {self.player1: single_info, self.player2: single_info}

        elif self.current_player == self.player1_ID:
            obs = {self.player2: p2_obs}
            reward = {self.player2: reward_vector[0]}
            info = {self.player2: single_info}
        elif self.current_player == self.player2_ID:
            obs = {self.player1: p1_obs}
            reward = {self.player1: reward_vector[1]}
            info = {self.player1: single_info}

        self.current_player = 1 - self.current_player

        return obs, reward, done, info

    def check_for_episode_termination(self, movecol, row):
        winner, reward_vector = self.winner, [0, 0]
        if self.does_move_win(movecol, row):
            winner = self.current_player
            if winner == self.player1_ID:
                reward_vector = [1, -1]
                self.score[self.player1] += 1
            elif winner == self.player2_ID:
                reward_vector = [-1, 1]
                self.score[self.player2] += 1
        elif self.get_moves(mask=False) == []:  # A draw has happened
            winner = -1
            reward_vector = [0, 0]
            self.num_draws += 1
        return winner, reward_vector

    def reset_score(self, n_steps=500):
        total_score = self.score[self.player1] + self.score[self.player2]
        if total_score >= n_steps:
            self.score[self.player1] = 0
            self.score[self.player2] = 0
            self.num_draws = 0

    def get_moves(self, mask=True):
        """
        :returns: array with all possible moves, index of columns which aren't full
        """
        if mask == False:
            if self.winner is not None:
                return []
            return [
                col
                for col in range(self.width)
                if self.board[col][self.height - 1] == -1
            ]
        # return an array of 0 if the action is invalid and 1 if it's valid
        if mask == True:
            if self.winner is not None:
                return [0.0] * 7
            act_mask = []
            for col in range(self.width):
                if self.board[col][self.height - 1] == -1:
                    act_mask.append(1.0)
                else:
                    act_mask.append(0.0)
            return act_mask

    def does_move_win(self, x, y):
        """
        Checks whether a newly dropped chip at position param x, param y
        wins the game.
        :param x: column index
        :param y: row index
        :returns: (boolean) True if the previous move has won the game
        """
        me = self.board[x][y]
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1)]:
            p = 1
            while (
                self.is_on_board(x + p * dx, y + p * dy)
                and self.board[x + p * dx][y + p * dy] == me
            ):
                p += 1
            n = 1
            while (
                self.is_on_board(x - n * dx, y - n * dy)
                and self.board[x - n * dx][y - n * dy] == me
            ):
                n += 1

            if p + n >= (
                self.connect + 1
            ):  # want (p-1) + (n-1) + 1 >= 4, or more simply p + n >- 5
                return True

        return False

    def is_on_board(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def get_result(self, player):
        """
        :param player: (int) player which we want to see if he / she is a winner
        :returns: winner from the perspective of the param player
        """
        return +1 if player == self.winner else -1

    def __repr__(self):
        """
        Return current game status as class representation
        """
        s = ""
        s += "\n"
        for x in range(self.height - 1, -1, -1):
            for y in range(self.width):
                s += {-1: ".", 0: "X", 1: "O",}[self.board[y][x]]
                s += " "
            s += "\n"
        return s

    def render(self, mode="human", screen_width=600, screen_height=400):

        if mode == "classic":

            square_len = 70
            circle_radius = 30
            screen_width = square_len * self.width
            screen_height = square_len * (self.height + 1)
            background_height = square_len * (self.height + 1)
            background_width = square_len * self.width
            circle_x = screen_width / 2 - background_width / 2 + square_len / 2
            circle_y = screen_height / 2 - background_height / 2 + square_len / 2
            top_bar_x = screen_width / 2
            top_bar_y = screen_height - square_len / 2

            if self.viewer is None:
                from gym.envs.classic_control import rendering

                self.viewer = rendering.Viewer(screen_width, screen_height)
                l, r, t, b = (
                    -background_width / 2,
                    background_width / 2,
                    background_height / 2,
                    -background_height / 2,
                )

                background = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                backtrans = rendering.Transform(
                    translation=(screen_width / 2, screen_height / 2)
                )
                background.add_attr(backtrans)
                background.set_color(0.0, 0.0, 1.0)
                self.viewer.add_geom(background)

                t_l, t_r, t_t, t_b = (
                    -background_width / 2,
                    background_width / 2,
                    square_len / 2,
                    -square_len / 2,
                )
                top_bar = rendering.FilledPolygon(
                    [(t_l, t_b), (t_l, t_t), (t_r, t_t), (t_r, t_b)]
                )
                toptrans = rendering.Transform(translation=(top_bar_x, top_bar_y))
                top_bar.add_attr(toptrans)
                top_bar.set_color(0.0, 0.0, 0.0)
                self.viewer.add_geom(top_bar)

                self.cells = []
                for row in range(self.height):
                    self.cells.append([])
                    for col in range(self.width):
                        cell = rendering.make_circle(radius=circle_radius, filled=True)
                        celltrans = rendering.Transform(
                            translation=(
                                circle_x + square_len * col,
                                circle_y + square_len * row,
                            )
                        )
                        cell.add_attr(celltrans)
                        cell.set_color(0.0, 0.0, 0.0)
                        self.viewer.add_geom(cell)
                        self.cells[-1].append(cell)

            for row in range(self.height):
                for col in range(self.width):
                    if self.board[col][row] == -1:
                        self.cells[row][col].set_color(0.0, 0.0, 0.0)
                    elif self.board[col][row] == 0:
                        self.cells[row][col].set_color(1.0, 0.0, 0.0)
                    elif self.board[col][row] == 1:
                        self.cells[row][col].set_color(1.0, 1.0, 0.0)
                    else:
                        raise ValueError(
                            "Error: board values is " + str(self.board[row][col])
                        )

            return self.viewer.render(return_rgb_array=mode == "rgb_array")

        else:
            raise NotImplementedError(
                'This mode has not been coded yet, select "human" or "classic"'
            )

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
