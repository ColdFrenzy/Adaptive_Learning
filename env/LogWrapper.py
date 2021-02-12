import logging

from env.connect4_multiagent_env import Connect4Env


class LogsWrapper(Connect4Env):
    """
    Wrapper for Connect4Env
    """

    def __init__(self, width=7, height=6, connect=4):
        super().__init__(width=7, height=6, connect=4)

        self.logger = self.init_logger("match.log")

    @staticmethod
    def init_logger(file_path):
        logger = logging.getLogger("Connect4Logger")
        f_handler = logging.FileHandler('match.log')
        f_handler.setLevel(logging.DEBUG)
        logger.addHandler(f_handler)
        return logger

    def step(self, action_dict):
        self.logger.debug("Player actions: " + str(action_dict))

        obs, reward, done, info = super().step(action_dict)

        if done["__all__"] == True:
            self.logger.debug("PLAYER " + str(self.current_player + 1) + " WON!!!!")
            self.logger.debug(
                "ACTUAL SCORE: P1 = "
                + str(self.score[self.player1])
                + " VS "
                + "P2 = "
                + str(self.score[self.player2])
            )

        self.logger.debug("Player rewards: " + str(reward))
        self.logger.debug(self)

        return obs, reward, done, info
