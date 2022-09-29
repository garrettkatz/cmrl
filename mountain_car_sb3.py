"""
This didn't work, but from zoo repo and python train.py:
- 5k, 10k steps almost always successful
- 1k steps often not successful, and about 6 seconds to run
"""

from collections import OrderedDict

# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
# according to sb3 baselines, ppo solves mountain car in fewest timesteps

# https://huggingface.co/sb3/ppo-MountainCarContinuous-v0
# according to hugging face, with these hyperparameters
hparams = OrderedDict([
             ('batch_size', 256),
             ('clip_range', 0.1),
             ('ent_coef', 0.00429),
             ('gae_lambda', 0.9),
             ('gamma', 0.9999),
             ('learning_rate', 7.77e-05),
             ('max_grad_norm', 5),
             ('n_envs', 1),
             ('n_epochs', 10),
             ('n_steps', 8),
             ('n_timesteps', 20000.0),
             ('normalize', True),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs', 'dict(log_std_init=-3.29, ortho_init=False)'),
             ('use_sde', True),
             ('vf_coef', 0.19),
             ('normalize_kwargs', {'norm_obs': True, 'norm_reward': False})])

# sanitize kwargs for PPO
hparams.pop('n_envs')
hparams.pop('normalize_kwargs')
n_timesteps = hparams.pop('n_timesteps')
hparams['normalize_advantage'] = hparams.pop('normalize')
hparams['policy_kwargs'] = eval(hparams['policy_kwargs'])

# https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html
# basic sb3 training setup
import gym
from stable_baselines3 import PPO

env = gym.make('MountainCarContinuous-v0')
model = PPO(env=env, verbose=1, **hparams)
model.learn(total_timesteps=n_timesteps)

