# import dependencies
import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import os

from gym_utils import SMBRamWrapper, load_smb_env, SMB

# Select a pre-trained model
MODEL_DIR = './models'

# obs = 4 frames
#crop_dim = [0, 16, 0, 13]
#n_stack = 4
#n_skip = 4
#MODEL_NAME = 'pre-trained-1'

# obs = 1 frames
#crop_dim = [0, 16, 0, 13]
#n_stack = 1
#n_skip = 1
#MODEL_NAME = 'pre-trained-2'


# obs = 2 frames
crop_dim = [0, 16, 0, 13]
n_stack = 2
n_skip = 4
MODEL_NAME = 'pre-trained-3'

# load env and model
env_wrap = load_smb_env('SuperMarioBros-1-1-v0', crop_dim, n_stack, n_skip)
model = PPO.load(os.path.join(MODEL_DIR, MODEL_NAME), env=env_wrap)
smb = SMB(env_wrap, model)

smb.play(episodes=1, deterministic=True, render=True, return_eval=True)