class Connect4Config:
    # =============================================================================
    # ENV
    # =============================================================================
    WIDTH = 7
    HEIGHT = 6
    N_ACTIONS = 7
    CONNECT = 4
    ENV_LOG_STEP = 100
    PLAYER1 = "player1"
    PLAYER2 = "player2"
    PLAYER1_ID = 0
    PLAYER2_ID = 1
    DRAW_ID = -1
    EMPTY = -1
    # decide randomly which player will start the game
    RANDOMIZE_START = True
    PLAYER_DICT = {"player1": PLAYER1_ID, "player2": PLAYER2_ID}
    
class Connect3Config:
    # =============================================================================
    # ENV
    # =============================================================================
    WIDTH = 5
    HEIGHT = 4
    N_ACTIONS = 5
    CONNECT = 3
    ENV_LOG_STEP = 100
    PLAYER1 = "player1"
    PLAYER2 = "player2"
    PLAYER1_ID = 0
    PLAYER2_ID = 1
    DRAW_ID = -1
    EMPTY = -1
    # decide randomly which player will start the game
    RANDOMIZE_START = True
    PLAYER_DICT = {"player1": PLAYER1_ID, "player2": PLAYER2_ID}
    
