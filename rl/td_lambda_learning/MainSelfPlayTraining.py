import matplotlib.pyplot as plt
import torch
import random
import time
import logging

from utils import utils

from game.globals import CONST
from game.globals import Globals
import td_lambda_learning


# The logger
utils.init_logger(logging.DEBUG, file_name="log/tic_tac_toe.log")
logger = logging.getLogger('TicTacToeTrain')


# set the random seed
random.seed(a=None, version=2)


# start to train the neural network
epoch_count = 100        # the number of epochs to train the neural network 1000 ~ 100'000 episodes ~ 25min
episode_count = 100       # the number of games that are self-played in one epoch
test_interval = 10        # epoch intervals at which the network plays against a random player
test_game_count = 1000    # the number of games that are played in the test against the random opponent
epsilon = 0.1             # the exploration constant
lambda_param = 0.6        # the lambda parameter in TD(lambda)
disc = 0.9                # the discount factor
learning_rate = 0.01      # the learning rate of the neural network
batch_size = 32           # the batch size of the experience buffer for the neural network training
exp_buffer_size = 10000   # the size of the experience replay buffer

# define the devices for the training and the target networks     cpu or cuda, here cpu is way faster for small nets
Globals.device = torch.device('cpu')


# create the agent
agent = td_lambda_learning.Agent(learning_rate, epsilon, disc, lambda_param, batch_size, exp_buffer_size)


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
        
        
    ###### self play and update: create some game data through self play
    # logger.info("start playing games in epoch {}".format(i))
    for _ in range(episode_count):
        # play one self-game
        agent.play_self_play_game()

        # update the neural network
        agent.td_update()
                
    
    
    ###### refresh the eligibility traces
    # logger.info("sync neural networks in epoch {}".format(i)) 
    agent.update_eligibilities(lambda_param, disc)


end_training = time.time()
training_time = end_training - start_training
logger.info("elapsed time whole training process {} for {} episodes".format(training_time, epoch_count*episode_count))

# save the currently trained neural network
torch.save(agent.network, "ticTacToeSelfPlay.pt")


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


