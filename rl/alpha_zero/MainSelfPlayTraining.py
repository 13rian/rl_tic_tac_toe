import matplotlib.pyplot as plt
import torch
import random
import time
import logging

from utils import utils

from game.globals import CONST
from game.globals import Globals
import alpha_zero


# The logger
utils.init_logger(logging.DEBUG, file_name="log/tic_tac_toe.log")
logger = logging.getLogger('TicTacToeTrain')


# set the random seed
random.seed(a=None, version=2)


# define the parameters
epoch_count = 400               # the number of epochs to train the neural network 100'000 episodes ~ 1h
episode_count = 100             # the number of games that are self-played in one epoch
test_interval = 10              # epoch intervals at which the network plays against a random player
test_game_count = 1000          # the number of games that are played in the test against the random opponent
mcts_sim_count = 80             # the number of simulations for the monte-carlo tree search
c_puct = 1                      # the higher this constant the more the mcts explores
temp = 1                        # the temperature, controls the policy value distribution
new_net_win_rate = 0.55         # win rate of the new network in order to replace the old one
learning_rate = 0.005           # the learning rate of the neural network
batch_size = 32                 # the batch size of the experience buffer for the neural network training
exp_buffer_size = episode_count * 9         # the size of the experience replay buffer

# define the devices for the training and the target networks     cpu or cuda, here cpu is way faster for small nets
Globals.device = torch.device('cpu')


# create the agent
agent = alpha_zero.Agent(learning_rate, mcts_sim_count, c_puct, temp, batch_size, exp_buffer_size)


# to plot the fitness
episodes = []
fitness_white = []
fitness_black = []


start_training = time.time()
for i in range(epoch_count):
    
    ###### evaluation: let the agent play against a random test opponent
    if i % test_interval == 0:
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



    ###### training, train the training network and use the target network for predictions
    # logger.info("start updates in epoch {}".format(i))
    agent.nn_update()
                

    ###### let the previous network play against the new network
    # logger.info("sync neural networks in epoch {}".format(i))
    win_rate = agent.play_against_old_net(test_game_count)
    if win_rate


end_training = time.time()
training_time = end_training - start_training
logger.info("elapsed time whole training process {} for {} episodes".format(training_time, epoch_count*episode_count))

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


