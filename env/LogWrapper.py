import logging
from env.connect4_multiagent_env import Connect4Env
from config.connect4_config import Connect4Config


class LogsWrapper(Connect4Env):
    """
    Wrapper for Connect4Env
    """

    def __init__(
        self,
        env_context,
        width=Connect4Config.WIDTH,
        height=Connect4Config.HEIGHT,
        n_actions=Connect4Config.N_ACTIONS,
        connect=Connect4Config.CONNECT,
    ):
        self.log_step = Connect4Config.ENV_LOG_STEP
        self.log_idx = 0
        self.logger = self.init_logger("log/match.log")
        super(LogsWrapper, self).__init__(
            env_context, width, height, n_actions, connect
        )

    def reset(
        self,
        current_player=Connect4Config.PLAYER1_ID,
        randomize=Connect4Config.RANDOMIZE_START,
    ):
        self.log_idx += 1

        return super(LogsWrapper, self).reset(current_player, randomize)

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
        obs, reward, done, info = super(LogsWrapper, self).step(action_dict)

        if self.log_idx % self.log_step == 0:
            self.logger.info("GAME NUMBER " + str(self.log_idx))
            self.logger.info("Player actions: " + str(action_dict))
            self.logger.info(self)

            if done["__all__"]:
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
