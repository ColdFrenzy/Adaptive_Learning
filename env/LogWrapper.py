import logging

from env.connect4_multiagent_env import Connect4Env


class LogsWrapper(Connect4Env):
    """
    Wrapper for Connect4Env
    """

    def __init__(self, env_context, width=7, height=6, connect=4):
        super().__init__(env_context, width, height, connect)

        self.logger = self.init_logger("log/match.log")

    @staticmethod
    def init_logger(file_path):
        logger = logging.getLogger("Connect4Logger")

        f_handler = logging.FileHandler(file_path, "w", "utf-8")
        f_handler.setLevel(logging.DEBUG)
        f_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        f_handler.setFormatter(f_format)

        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.WARN)
        c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        c_handler.setFormatter(c_format)

        logger.addHandler(f_handler)
        logger.addHandler(c_handler)

        logger.setLevel(logging.DEBUG)

        return logger

    def step(self, action_dict):
        self.logger.info("Player actions: " + str(action_dict))

        obs, reward, done, info = super().step(action_dict)

        if done["__all__"] == True:
            self.logger.info("PLAYER " + str(self.current_player) + " WON!!!!")
            self.logger.info(
                "ACTUAL SCORE: P1 = "
                + str(self.score[self.player1])
                + " VS "
                + "P2 = "
                + str(self.score[self.player2])
            )

        self.logger.info(f"Player rewards: {reward}\n{self}")

        return obs, reward, done, info
