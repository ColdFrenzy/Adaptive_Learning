# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(1, os.path.abspath(os.path.curdir))
from config.connect4_config import Connect3Config
import tkinter as tk
from tkinter import ttk
import pygame
from env.connect4_multiagent_env import Connect4Env
from numpy import random
import math
import sys
import os
sys.path.insert(1, os.path.abspath(os.path.curdir))

import numpy as np

from models import Connect4ActionMaskModel
from utils.learning_behaviour_utils import LSTM_model,split_train_val,\
    minimax_vs_minimax_connect3_single_game,return_one_hot,\
    minimax_vs_minimax_connect3_single_game_plus_outcome,count_elem_in_dataset
from config.learning_behaviour_config import Config
from config.custom_config import Config as C
from tensorflow import keras
from ray.rllib.agents.ppo import PPOTrainer
from config.trainer_config import TrainerConfig


BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)
GREY = (169,169,169)
WHITE = (255,255,255)

ROW_COUNT = Connect3Config.HEIGHT#6
COLUMN_COUNT = Connect3Config.WIDTH
# EMPTY = Connect3Config.EMPTY
# PLAYER_PIECE = Connect3Config.PLAYER1_ID
# AI_PIECE = Connect3Config.PLAYER2_ID
SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)


# center the window
#os.environ['SDL_VIDEO_CENTERED'] = '1'
#os.environ['SDL_VIDEODRIVER'] = 'windows'
LARGEFONT =("Verdana", 35) 
MEDIUMFONT = ("Verdana", 16)
SMALLFONT = ("Verdana", 10)


agent = None

class tkinterApp(tk.Tk): 
      
    # __init__ function for class tkinterApp  
    def __init__(self,model,lstm_model,weights, *args, **kwargs):  
        tk.Tk.__init__(self, *args, **kwargs)  
        windowWidth = 800 #self.winfo_reqwidth()
        windowHeight = 600 #self.winfo_reqheight()

        # Gets both half the screen width/height and window width/height
        positionRight = int(self.winfo_screenwidth()/2 - windowWidth/2)
        positionDown = int(self.winfo_screenheight()/2 - windowHeight/2)

        # Positions the window in the center of the page.
        self.geometry("800x600+{}+{}".format(positionRight, positionDown))
        pad=3
        self._geom="{0}x{1}+0+0".format(self.winfo_screenwidth()-pad, self.winfo_screenheight()-pad)
        #self.geometry('800x600+0+0')
        self.bind('<Escape>',self.toggle_geom) 
          
        # creating a container 
        container = tk.Frame(self)   
        container.pack(side = "top", fill = "both", expand = True)  
        # weight equal to 1 allow the cell to grow if there is enough space
        # in the following case, it will span all over the root frame
        container.grid_rowconfigure(0, weight = 1) 
        container.grid_columnconfigure(0, weight = 1) 
        
        # initializing frames to an empty array 
        self.frames = {}   
   
        # iterating through a tuple consisting 
        # of the different page layouts 
        for F in (StartPage, SelectModePage): 
   
            frame = F(container, self,model,lstm_model,weights) 
   
            # initializing frame of that object from 
            # startpage, page1, page2 respectively with  
            # for loop 
            self.frames[F] = frame  
   
            frame.grid(row = 0, column = 0, sticky ="nsew") 
   
        self.show_frame(StartPage) 

   
    # to display the current frame passed as 
    # parameter 
    def show_frame(self, cont): 
        frame = self.frames[cont] 
        frame.tkraise() 
        
    def toggle_geom(self,event):
        geom=self.winfo_geometry()
        print(geom,self._geom)
        self.geometry(self._geom)
        self._geom=geom   
        
        
class StartPage(tk.Frame): 
    def __init__(self, parent, controller,model,lstm_model,weights):  
        tk.Frame.__init__(self, parent) 
          
        # label of frame Layout 2 
        label = ttk.Label(self, text ="CONNECT 3", font = LARGEFONT) 
        self.grid_rowconfigure(0, weight = 1) 
        self.grid_rowconfigure(1, weight = 1) 
        self.grid_rowconfigure(2, weight = 1)
        self.grid_columnconfigure(0, weight = 1)
        self.grid_columnconfigure(1, weight = 1) 
        self.grid_columnconfigure(2, weight = 1)

        label.grid(row = 1, column = 1) # padx = 10, pady = 10)  
   
        button1 = tk.Button(self, text ="Start", height = 3, width = 25,\
        command = lambda : game_window(controller,"AIVSAI",model,lstm_model,weights))                 
        #command = lambda : controller.show_frame(SelectModePage))
            
      
        button1.grid(row = 2, column = 1, sticky="N") # padx = 10, pady = 10) 
        
class SelectModePage(tk.Frame): 
      
    def __init__(self, parent, controller,model,lstm_model,weights): 
          
        tk.Frame.__init__(self, parent) 
        self.grid_rowconfigure(0, weight = 1) 
        self.grid_columnconfigure(0, weight = 1) 
        self.grid_rowconfigure(1, weight = 1) 

        Frame1 = tk.Frame(self) # bg = "blue" to see the dimension of the frame
        Frame1.pack(side = "top", fill = "both",expand = True) 
        Frame2 = tk.Frame(self) # bg = "red"
        Frame2.pack(side = "top", fill = "both",expand = True)
        
        Frame1.grid_rowconfigure(0, weight = 1) 
        Frame1.grid_rowconfigure(1, weight = 1) 
        Frame1.grid_rowconfigure(2, weight = 1) 
        Frame1.grid_columnconfigure(0, weight = 1)
        Frame1.grid_columnconfigure(1, weight = 2) 
        Frame1.grid_columnconfigure(2, weight = 1)
        
        Frame2.grid_rowconfigure(0, weight = 1) 
        Frame2.grid_rowconfigure(1, weight = 1) 
        Frame2.grid_rowconfigure(2, weight = 1) 
        Frame2.grid_columnconfigure(0, weight = 1)
        Frame2.grid_columnconfigure(1, weight = 1) 
        Frame2.grid_columnconfigure(2, weight = 1)
   
        label = tk.Label(Frame1, text ="SELECT MODE", font = LARGEFONT) 
        label.grid(row = 2, column = 1)
        button_width = 25
        button_height = 3
            
        button1 = tk.Button(Frame2, text ="PLAYER VS AI",\
                             height= button_height, width = button_width,\
                                 command = lambda : print("hello"))
            # controller.show_frame(SelectAlgorithmPage_PVSAI)
                                 #command = lambda : game_window(controller,"PVSAI")) 
      
       
        button1.grid(row = 0, column = 0)
   
 
        button2 = tk.Button(Frame2, text ="AI VS AI ", \
                             height= button_height, width = button_width,\
                                 command = lambda : print("Yolo")) 
      

        button2.grid(row = 0, column = 2,ipadx = 10)        

def game_window(root,mode,model,lstm_model,weights):
      
    # embed.grid(columnspan = (600), rowspan = 500) # Adds grid
    # self.pack(side = LEFT) #packs window to the left
    # buttonwin = tk.Frame(root, width = 75, height = 500)
    #self.pack(side = "left")
    root.destroy()
    # ROW_COUNT = connect4.ROW_COUNT
    # COLUMN_COUNT = connect4.COLUMN_COUNT
    # SQUARESIZE = connect4.SQUARESIZE
    # width = connect4.width
    # height = connect4.height

    
    pygame.init()
    #pygame.display.update()
    screen = pygame.display.set_mode(size)

    player_vs_AI(screen,model,lstm_model,weights,close_window=False)
    
    restart = restart_window(mode,model,lstm_model,weights)
    restart.mainloop()
    # app = tkinterApp()
    # app.mainloop()       
    
    
    
    
class restart_window(tk.Tk):
    def __init__(self, mode,model,lstm_model,weights, *args, **kwargs):  
        tk.Tk.__init__(self, *args, **kwargs)  
        
        windowWidth = 500 
        windowHeight = 500 
        positionRight = int(self.winfo_screenwidth()/2 - windowWidth/2)
        positionDown = int(self.winfo_screenheight()/2 - windowHeight/2)
        # Positions the window in the center of the page.
        self.geometry("500x400+{}+{}".format(positionRight, positionDown))
        pad=3
        self._geom="{0}x{1}+0+0".format(self.winfo_screenwidth()-pad, self.winfo_screenheight()-pad)
        #self.geometry('800x600+0+0')
          
        # creating a container 
        Frame1 = tk.Frame(self)   
        Frame1.pack(side = "top", fill = "both", expand = True)  
        Frame2 = tk.Frame(self)
        Frame2.pack(side = "top", fill = "both",expand = True) 


        
        Frame1.grid_rowconfigure(0, weight = 1) 
        Frame1.grid_rowconfigure(1, weight = 1) 
        Frame1.grid_rowconfigure(2, weight = 1) 
        Frame1.grid_columnconfigure(0, weight = 1)
        Frame1.grid_columnconfigure(1, weight = 2) 
        Frame1.grid_columnconfigure(2, weight = 1)
        
        
        Frame2.grid_rowconfigure(0, weight = 1) 
        Frame2.grid_rowconfigure(1, weight = 1) 
        Frame2.grid_rowconfigure(2, weight = 1) 
        Frame2.grid_columnconfigure(0, weight = 1)
        Frame2.grid_columnconfigure(1, weight = 1) 
        Frame2.grid_columnconfigure(2, weight = 1)


        restart_label = tk.Label(Frame1, text ="PLAY AGAIN?",font = LARGEFONT)
        restart_label.grid(row = 2, column = 1)

        
        button_width = 25
        button_height = 3

        button1 = tk.Button(Frame2, text ="MENU",\
                             height= button_height, width = button_width,\
                                 command = lambda : restart(self,model,lstm_model,weights) )
                                  
    
       
        button1.grid(row = 1, column = 0, padx = 10, pady = 20) 
   
 
        button2 = tk.Button(Frame2, text ="RESTART", \
                             height= button_height, width = button_width,\
                                 command = lambda : game_window(self,mode,model,lstm_model,weights))
        button2.grid(row = 1, column = 2, padx = 10, pady = 20)     
    

def restart(root,model,lstm_model,weights):
    global agent
    pygame.quit()
    root.destroy()
    agent = None
    app = tkinterApp(model,lstm_model,weights)
    app.mainloop()

def draw_board(board,screen):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):        
            if board[r][c] == player1_ID:
                pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
            elif board[r][c] == player2_ID: 
                pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    pygame.display.update()
    
    
    
def player_vs_AI(screen,model,lstm_model,weights,close_window = True ):
    
    global agent
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )
    board = game.board    
    board_to_print = np.transpose(board)
    draw_board(board_to_print,screen)
    
    myfont = pygame.font.SysFont("monospace", 55)

    full_game = []
    timestep = 0
    game_over = False
    actions = {}
    print("printing agent = " + str(agent))
    if agent is not None:
        print("Agent chosen using the model")
        w2_indx = lvl_to_indx(int(agent))
        w2_key = list(weights.keys())[w2_indx]
        w2 = weights[w2_key]
        lvl = indx_to_lvl(w2_indx)
        # lvl = weights.keys().index(w2)
    else:
        print("Agent chosen randomly...")
        w2_indx = np.random.choice(range(len(weights)))
        w2_key = list(weights.keys())[w2_indx]
        w2 = weights[w2_key]
        lvl = indx_to_lvl(w2_indx)
    
    if randomize:
        starting_player = random.choice([player1_ID, player2_ID])
    else:
        starting_player = player1_ID
    game.reset(starting_player=starting_player,randomize=False)
    
    board_plus_action_total = []
    done = {}
    done["__all__"] = False
    
    
    print("You are now playing against an agent of level " + str(lvl) )
    
    while not game_over:
        timestep += 1
        actual_player = game.current_player
        board = game.board
        board_p2 = game.board_p2
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
                posx = event.pos[0]
                if actual_player == player1_ID:
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
    
            pygame.display.update()
    
            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
                #print(event.pos)
                # Ask for Player 1 Input                
                if actual_player == player1_ID:
                    action_mask = game.get_moves(False) 
                    posx = event.pos[0]
                    act = int(math.floor(posx/SQUARESIZE))
                    print("player 1 move: " + str(act))
                    if act in action_mask:
                        flattened_board = np.ndarray.flatten(board)
                        board_plus_actions = np.append(flattened_board,float(act))
                        board_plus_action_total.append([board_plus_actions])
                
                        actions[player1] = act                
                        _, rew, done, _ = game.step(actions)
    
                        #print_board(board)
                        board = game.board
                        board_to_print = np.transpose(board)
                        draw_board(board_to_print,screen)
    
    
        # # Ask for Player 2 Input
        if actual_player == player2_ID:
            input_dict = {"obs": {}}
            action_mask = game.get_moves(True)
            input_dict["obs"]["state"] = board_p2 #reshaped_board
            input_dict["obs"]["action_mask"] = action_mask
            action_logits, _ = model.forward(input_dict, None, None)
            if timestep > number_of_stochastic_moves:
                act = np.argmax(action_logits[0])
            elif timestep <= number_of_stochastic_moves:
                action_prob = [np.exp(single_log)/sum(np.exp(action_logits[0])) for single_log in action_logits[0]]
                act = np.random.choice([0,1,2,3,4],1,p=action_prob)[0]  
                
            actions[player2] = act
            pygame.time.wait(1000)
            _, rew, done, _ = game.step(actions)
            print(game)
            #game.render()    
            board = game.board
            board_to_print = np.transpose(board)
            draw_board(board_to_print,screen)
                
                
        if done["__all__"]:
            # ADD ENCODED GAME TO THE LISt
            print(rew)
            if rew["player1"] == 1.0:
                print("Player 1 won!!")
                label = myfont.render("Player 1 won!!", 1, RED)
                screen.blit(label, (40, 10))
                pygame.display.update()
                pygame.time.wait(1000)

            elif rew["player1"] == -1.0:
                print("Player 2 won!!")
                label = myfont.render("Player 2 won!!", 1, YELLOW)
                screen.blit(label, (40,10))
                pygame.display.update()
                pygame.time.wait(1000)
                
            elif rew["player1"] == 0.0:
                print("Draw")
                label = myfont.render("Draw!!", 1, WHITE)
                screen.blit(label, (40,10))
                pygame.display.update()
                pygame.time.wait(1000)
                
            if len(board_plus_action_total) < sequence_len:
                print("Game finished too early, restarting...")
                timestep = 0
                game.reset(randomize=True)
                game_over = False
                continue
            
            game_over = True
            board_plus_action_and_outcome = board_plus_action_total
            
            for j in range(len(board_plus_action_and_outcome)-(sequence_len-1)):
                full_game.append([])
                full_game[-1].append(board_plus_action_and_outcome[j:j+sequence_len])
        

        if game_over:
            full_game = np.asarray(full_game)
            full_game = np.squeeze(full_game)
            if len(full_game.shape) == 2:
                full_game = np.expand_dims(full_game, axis=0)
            full_game = full_game.astype("float32")
            y = lstm_model(full_game,training=False)
                    
            predicted_values = tf.math.reduce_mean(y,axis=0)
            predicted_indx = tf.math.argmax(predicted_values) 
            agent = predicted_indx
        
            print("Model output probability: " + str(predicted_values.numpy()))
            pygame.time.wait(3000)
            
    if close_window:
        pygame.display.quit()
        
    
  
        
  
def indx_to_lvl(indx):
    if indx < 4:
        return 1
    elif indx < 8:
        return 4
    elif indx <12: 
        return 6
    
def lvl_to_indx(lvl):
    if lvl == 0:
        return random.choice(range(4))
    elif lvl == 1:
        return random.choice(range(4,8))
    elif lvl == 2:
        return random.choice(range(8,12))


if __name__ == "__main__":
    _ = Connect4ActionMaskModel
    data_dir = C.DATA_DIR
    lstm_hidden = Config.LSTM_HIDDEN[-2] # best weights
    batch_size = Config.BATCH_SIZE
    lstm_timesteps = Config.LSTM_TIMESTEPS
    outcome_as_feature = Config.OUTCOME_AS_FEATURE
    if outcome_as_feature:
        features_len = Config.FEATURES_LEN_2
    else:
        features_len = Config.FEATURES_LEN
    output_len = Config.OUTPUT_LEN
    best_weights_npy = os.path.join(data_dir,"lstm_best_weights.npy")
    lstm_weights = np.load(best_weights_npy,allow_pickle=True)
    number_of_evaluation_games = Config.NUMBER_OF_EVALUATION_GAMES #  100 
    number_of_games_to_test = Config.NUMBER_OF_GAMES_TO_TEST #[1,2,3,4,5]
    depth_list = Config.DEPTH_LIST # [1,4,6]
    number_of_stochastic_moves = 6
    sequence_len = lstm_timesteps
    
    npy_weights_file = os.path.join(data_dir,"weights.npy")
    weights = np.load(npy_weights_file,allow_pickle=True)[()]  
    
    play = True
    
    trainer_obj = PPOTrainer(
        config=TrainerConfig.PPO_TRAINER_CONNECT3,
    )
    model = trainer_obj.get_policy("player1").model

    # =============================================================================
    # TEST THE MODEL 
    # =============================================================================
    import tensorflow as tf 
    lstm_model = LSTM_model(batch_size,(lstm_timesteps,features_len),output_len,lstm_hidden,False)  
    
    # generate a fake input to define the model stucture and then load the weights 
    # [batch,timestep,features]
    # random_input = np.random.rand(1,lstm_timesteps,features_len)
    random_input = np.random.rand(1,lstm_timesteps,features_len)
    random_input = random_input.astype('float32')
    lstm_model(random_input)
    lstm_model.set_weights(lstm_weights[()])
    
    randomize = True
    player1_ID = Connect3Config.PLAYER1_ID
    player2_ID = Connect3Config.PLAYER2_ID
    player1 = Connect3Config.PLAYER1
    player2 = Connect3Config.PLAYER2
    game = Connect4Env(None,width=Connect3Config.WIDTH,
        height=Connect3Config.HEIGHT,
        n_actions=Connect3Config.N_ACTIONS,
        connect=Connect3Config.CONNECT,
    )
    
    
    # =============================================================================
    # PYGAME
    # =============================================================================
    # pygame.init()
    # #pygame.display.update()
    # screen = pygame.display.set_mode(size)
    # pygame.display.set_caption("Connect 3")
    # player_vs_AI(screen,model,lstm_model) 
    
    app = tkinterApp(model,lstm_model,weights) 
    app.mainloop() 