# =============================================================================
# UTILITY TO SAVE SCREENSHOTS OF THE BOARD IN A GIVEN STATE
# =============================================================================

import os
import sys

# import pyautogui
import cv2
import numpy as np

sys.path.insert(1, os.path.abspath(os.pardir))
from env.connect4_multiagent_env import Connect4Env
from config.connect4_config import Connect4Config,Connect3Config


def make_screenshot(env, board, out_file):
    env.board = board
    env.render()
    viewer = env.viewer
    img = viewer.get_array()
    # img is RGB but cv2 need images in BGR
    img_to_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # data = np.asarray(viewer.get_array(viewer.width, viewer.height, depth=False)[::-1, :, :], dtype=np.uint8)
    if img_to_save is not None:
        cv2.imwrite(out_file, img_to_save)
    viewer.close()


if __name__ == "__main__":
    
    env = Connect4Env(None)
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )
    board = game.board
    # empty_board_connect3 = np.array([
    #     [ -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1.]])
    
    # board_connect3 = np.array([
    #     [ 0., 1., -1., -1.],
    #     [ 1., 0., 0., -1.],
    #     [ 1., 0., -1., -1.],
    #     [ 0., 1., -1., -1.],
    #     [ -1., -1., -1., -1.]])
    open_line_vertical = np.array([
        [ -1., -1., -1., -1., -1., -1.],
        [ -1., -1., -1., -1., -1., -1.],
        [ 1., 1., -1., -1., -1., -1.],
        [ 0., 0., 0., -1., -1., -1.],
        [ 1., -1., -1., -1., -1., -1.],
        [ -1., -1., -1., -1., -1., -1.],
        [ -1., -1., -1., -1., -1., -1.]])
    
    
    # empty_board = np.array([[ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.]])
    # board = np.array(
    #     [
    #         [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    #         [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    #         [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    #         [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    #         [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    #         [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    #         [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    #     ]
    # )
    # p1_board = np.array(
    #     [
    #         [ -1., -1., -1., -1., -1., -1.],
    #         [ 1, -1., -1., -1., -1., -1.],
    #         [ 0., 1., 0., -1., -1., -1.],
    #         [ 1., 0., -1., -1., -1., -1.],
    #         [ 1., 1., 1., 0., -1., -1.],
    #         [ 0., -1., -1., -1., -1., -1.],
    #         [ -1., -1., -1., -1., -1., -1.]
    #     ])
    # board = np.array([[ 1., -1., -1., -1., -1., -1.],
    #    [ 0., -1., -1., -1., -1., -1.],
    #    [ 1.,  1., -1., -1., -1., -1.],
    #    [-1., -1., -1., -1., -1., -1.],
    #    [ 0., -1., -1., -1., -1., -1.],
    #    [-1., -1., -1., -1., -1., -1.],
    #    [ 1.,  0., -1., -1., -1., -1.]])

    # board = env.board
    make_screenshot(env, open_line_vertical, "test.png")
