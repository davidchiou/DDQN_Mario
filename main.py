import gym
import torch.optim as optim

from DQN_model import DQN
from DQN_learn import OptimizerSpec, dqn_learing
from utils.schedule import LinearSchedule

import ppaquette_gym_super_mario

from gym import wrappers

import torch
import numpy as np
import random
from utils.atari_wrapper import wrap_deepmind
SEED = 1
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 10000
#LEARNING_STARTS = 32 #debug for back_prop
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 3000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

def main(env):

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

if __name__ == '__main__':
    env = gym.make("ppaquette/SuperMarioBros-1-1-v0")
    
    
    # set global seeds
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    
    # monitor & wrap the game
    env = wrap_deepmind(env)
    
    expt_dir = 'Game_video'
    #env = wrappers.Monitor(env, expt_dir, force=True, video_callable=lambda episode_id: episode_id % 10 == 0)
    env = wrappers.Monitor(env, expt_dir, force=True, video_callable=lambda episode_id: True)

    # main
main(env)


