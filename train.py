from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch,trpo_tf1,vpg_pytorch,ddpg_pytorch,td3_pytorch,sac_pytorch
import torch
import gym

import tensorflow as tf
env_fn=lambda: gym.make('LunarLander-v2')
if __name__ == '__main__':
     import argparse
     parser = argparse.ArgumentParser()
     parser.add_argument('--cpu', type=int, default=4)
     parser.add_argument('--num_runs', type=int, default=3)
     args = parser.parse_args()
     '''Experimenting with PPO
     eg = ExperimentGrid(name='ppo-pyt-bench')
     eg.add('env_name', 'LunarLander-v2', '', True)
     eg.add('seed', [10*i for i in range(args.num_runs)])
     eg.add('epochs', 10)
     eg.add('steps_per_epoch', 4000)
     eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
     eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
     eg.run(ppo_pytorch, num_cpu=-1)
	#Experimenting with TRPO
     ega = ExperimentGrid(name='trpo-tf1-bench')
     ega.add('env_name', 'LunarLander-v2', '', True)
     ega.add('seed', [10*i for i in range(args.num_runs)])
     ega.add('epochs', 10)
     ega.add('steps_per_epoch', 4000)
     ega.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
     ega.add('ac_kwargs:activation', [tf.tanh, tf.nn.relu], '')
     ega.run(trpo_tf1, num_cpu=-1)
	#Experimenting with VPG
     eg1 = ExperimentGrid(name='vpg-pyt-bench')
     eg1.add('env_name', 'LunarLander-v2', '', True)
     eg1.add('seed', [10*i for i in range(args.num_runs)])
     eg1.add('epochs', 10)
     eg1.add('steps_per_epoch', 4000)
     eg1.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
     eg1.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
     eg1.run(vpg_pytorch, num_cpu=-1)
	#Experimenting with DDPG'''
     eg2 = ExperimentGrid(name='ddpg-pyt-bench')
     eg2.add('env_name', 'Pendulum-v0', '', True)
     eg2.add('seed', [10*i for i in range(args.num_runs)])
     eg2.add('epochs', 10)
     eg2.add('steps_per_epoch', 4000)
     eg2.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
     eg2.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
     eg2.run(ddpg_pytorch, num_cpu=-1)
	#Experimenting with td3
     eg3 = ExperimentGrid(name='td3-pyt-bench')
     eg3.add('env_name', 'Pendulum-v0', '', True)
     eg3.add('seed', [10*i for i in range(args.num_runs)])
     eg3.add('epochs', 10)
     eg3.add('steps_per_epoch', 4000)
     eg3.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
     eg3.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
     eg3.run(td3_pytorch, num_cpu=-1)
	#Experimenting with sac
     eg4 = ExperimentGrid(name='sac-pyt-bench')
     eg4.add('env_name', 'Pendulum-v0', '', True)
     eg4.add('seed', [10*i for i in range(args.num_runs)])
     eg4.add('epochs', 10)
     eg4.add('steps_per_epoch', 4000)
     eg4.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
     eg4.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
     eg4.run(sac_pytorch, num_cpu=-1)



