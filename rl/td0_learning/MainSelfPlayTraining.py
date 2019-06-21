import matplotlib.pyplot as plt
import torch
import random
import time
import logging

from utils import utils

from game.globals import CONST
from game.globals import Globals
import td0_learning


# The logger
utils.init_logger(logging.DEBUG, file_name="log/tic_tac_toe.log")
logger = logging.getLogger('TD0_Training')
livePlots = True


# set the random seed
random.seed(a=None, version=2)


# start to train the neural network
epoch_count = 480               # the number of epochs to train the neural network 962 ~ 100'000 episodes ~ 1h
episode_count = 104             # the number of games that are self-played in one epoch
update_count = 9*episode_count  # the number the neural net is updated  in one epoch with the experience data
test_interval = 10              # epoch intervals at which the network plays against a random player
test_game_count = 1000          # the number of games that are played in the test against the random opponent
epsilon = 0.1                   # the exploration constant
disc = 0.99                     # the discount factor
learning_rate = 0.005           # the learning rate of the neural network
batch_size = 32                 # the batch size of the experience buffer for the neural network training
exp_buffer_size = 10000         # the size of the experience replay buffer

# define the devices for the training and the target networks     cpu or cuda, here cpu is way faster for small nets
Globals.device = torch.device('cpu')


# create the agent
agent = td0_learning.Agent(learning_rate, epsilon, disc, batch_size, exp_buffer_size)


# to plot the fitness
episodes = []
fitness_white = []
fitness_black = []


start_training = time.time()
for i in range(epoch_count):
    
    ###### evaluation: let the agent play against a random test opponent
    if i % test_interval == 0:
        # let both q learners play against a random opponent
        logger.info("start evaluating the network in epoch {}".format(i))
        white_score = agent.play_against_random(CONST.WHITE, test_game_count)
        logger.info("white score rate after {} epochs: {}".format(i, white_score))
        
        black_score = agent.play_against_random(CONST.BLACK, test_game_count)
        logger.info("black score rate after {} epochs: {}".format(i, black_score))

        episodes.append(i*episode_count)
        fitness_white.append(white_score)
        fitness_black.append(black_score)
        
        
    ###### self play: create some game data through self play
    # logger.info("start playing games in epoch {}".format(i))
    for _ in range(episode_count):
        # start a fresh game 
        agent.reset_board()
            
        while not agent.game_terminal():
            # play the epsilon greedy move and save the state transition in the experience buffer
            agent.epsilon_greedy_move()

        
    ###### training, train the training network and use the target network for predictions
    # logger.info("start updates in epoch {}".format(i)) 
    for _ in range(update_count):
        agent.td_update()
    
    
    ###### synchronize the target network with the training network
    # logger.info("sync neural networks in epoch {}".format(i)) 
    agent.sync_networks()


end_training = time.time()
trainingTime = end_training - start_training
logger.info("elapsed time whole training process {} for {} episodes".format(trainingTime, episode_count))

# save the currently trained neural network
torch.save(agent.training_network, "ticTacToeSelfPlay.pt")


# plot the results
fig2 = plt.figure(2)
axes = plt.gca()
axes.set_ylim([0, 1])

plt.plot(episodes, fitness_white, label="white")
plt.plot(episodes, fitness_black, label="black")
plt.legend(loc='best')


plt.title("Self-Play Training Progress")       
plt.xlabel("Episode")
plt.ylabel("Average score against random player")
fig2.show()
plt.show()


