import matplotlib.pyplot as plt
import torch
import random
import time
import logging

from utils import utils

from game.globals import Globals
from rl.alpha_zero import alpha_zero_learning


# The logger
utils.init_logger(logging.DEBUG, file_name="log/tic_tac_toe.log")
logger = logging.getLogger('Alpha Tic')


# set the random seed
random.seed(a=None, version=2)


# define the parameters
epoch_count = 100                   # the number of epochs to train the neural network
episode_count = 100                 # the number of games that are self-played in one epoch
update_count = 10                   # the number the neural net is updated  in one epoch with the experience data
network_duel_game_count = 40        # number of games that are played between the old and the new network
mcts_sim_count = 15                 # the number of simulations for the monte-carlo tree search
c_puct = 1                          # the higher this constant the more the mcts explores
temp = 1                            # the temperature, controls the policy value distribution
temp_threshold = 5                  # up to this move the temp will be temp, otherwise 0 (deterministic play)
new_net_win_rate = 0.55             # win rate of the new network in order to replace the old one
learning_rate = 0.005                 # the learning rate of the neural network
batch_size = 128                    # the batch size of the experience buffer for the neural network training
exp_buffer_size = 9*episode_count   # the size of the experience replay buffer

# define the devices for the training and the target networks     cpu or cuda, here cpu is way faster for small nets
Globals.device = torch.device('cpu')


# create the agent
agent = alpha_zero_learning.Agent(learning_rate, mcts_sim_count, c_puct, temp, batch_size, exp_buffer_size)


# to plot the fitness
policy_loss = []
value_loss = []


start_training = time.time()
for i in range(epoch_count):
    ###### self play and update: create some game data through self play
    # logger.info("start playing games in epoch {}".format(i))
    for _ in range(episode_count):
        # play one self-game
        agent.play_self_play_game(temp_threshold)



    ###### training, train the training network and use the target network for predictions
    # logger.info("start updates in epoch {}".format(i))
    loss_p, loss_v = agent.nn_update(update_count)
    policy_loss.append(loss_p)
    value_loss.append(loss_v)
    print("policy loss: ", loss_p)
    print("value loss: ", loss_v)
    agent.clear_exp_buffer()            # clear the experience buffer

    ###### let the previous network play against the new network
    logger.info("start neural networks duel in epoch {}".format(i))
    network_improved = agent.network_duel(new_net_win_rate, network_duel_game_count)
    if network_improved:
        logger.info("new generation network has improved")
    else:
        logger.info("new generation network has not improved")


end_training = time.time()
training_time = end_training - start_training
logger.info("elapsed time whole training process {} for {} episodes".format(training_time, epoch_count*episode_count))

# save the currently trained neural network
torch.save(agent.old_network, "ticTacToeSelfPlay_old.pt")
torch.save(agent.new_network, "ticTacToeSelfPlay.pt")


# plot the value training loss
fig1 = plt.figure(1)
plt.plot(value_loss)
plt.legend(loc='best')
plt.title("Average Value Training Loss")
plt.xlabel("Episode")
plt.ylabel("Value Loss")
fig1.show()

# plot the training policy loss
fig2 = plt.figure(2)
plt.plot(policy_loss)
plt.legend(loc='best')
plt.title("Average Policy Training Loss")
plt.xlabel("Episode")
plt.ylabel("Policy Loss")
fig2.show()

plt.show()


