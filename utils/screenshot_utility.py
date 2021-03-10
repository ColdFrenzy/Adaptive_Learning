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


def make_screenshot(env,board,out_file):
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
    # empty_board = np.array([[ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.],
    #     [ -1., -1., -1., -1., -1., -1.]])
    board = np.array([
        [ 1., 0., 1., 1., 0., 0.],
        [ 1., 1., 0., 0., 1., 0.],
        [ 1., 0., 1., 1., 0., 0.],
        [ 0., 1., 1., 0., 0., 1.],
        [ 1., 0., 1., 1., 0., 0.],
        [ 1., 1., 0., 0., 1., 0.],
        [ 1., 0., 1., 1., 0., 0.]]) 
    # board = np.array([[ 1., -1., -1., -1., -1., -1.],
    #    [ 0., -1., -1., -1., -1., -1.],
    #    [ 1.,  1., -1., -1., -1., -1.],
    #    [-1., -1., -1., -1., -1., -1.],
    #    [ 0., -1., -1., -1., -1., -1.],
    #    [-1., -1., -1., -1., -1., -1.],
    #    [ 1.,  0., -1., -1., -1., -1.]])

    # board = env.board
    make_screenshot(env, board, "test.png")