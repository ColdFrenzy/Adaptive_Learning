class Config:

    use_lstm = False
    as_test = True

    # =============================================================================
    # ENV
    # =============================================================================
    WIDTH = 7
    HEIGHT = 6
    N_ACTIONS = 7
    CONNECT = 4
    PLAYER1 = "player1"
    PLAYER2 = "player2"
    PLAYER1_ID = 0
    PLAYER2_ID = 1
    DRAW_ID = -1
    EMPTY = -1
    GAMMA = 0.9
    PLAYER_DICT = {"player1": PLAYER1_ID, "player2": PLAYER2_ID}

    # =============================================================================
    # ROLLOUT WORKERS
    # =============================================================================
    NUM_WORKERS = 0
    NUM_ENVS_PER_WORKER = 1
    ROLLOUT_FRAGMENT_LENGTH = 10
    TRAIN_BATCH_SIZE = 200

    # =============================================================================
    # TRAINING PARAMS
    # =============================================================================
    EPOCHS = 100
    REWARD_DIFFERENCE = 100
    WEIGHT_UPDATE_STEP = 5
    AVAILABLE_POLICIES = ["RANDOM", "MINIMAX", "PG"]

    # =============================================================================
    # MINIMAX PARAMS
    # =============================================================================
    SEARCH_DEPTH = 3
