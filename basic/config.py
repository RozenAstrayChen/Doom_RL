# -*- coding: utf-8 -*-
"""
This file is config all need hyperparameters
"""
# scenarios path
map_basic = "./scenarios/simpler_basic.cfg"
map_corridor = "./scenarios/deadly_corridor.cfg"
map_d_line = "./scenarios/defend_the_line.cfg"
map_d_center = "./scenarios/defend_the_center.cfg"
map_health = "./scenarios/health_gathering.cfg"
map_xhealth = "./scenarios/health_gathering_supreme.cfg"
map_match = "./scenarios/deathmatch.cfg"
map_cover = "./scenarios/take_cover.cfg"
map_oblige = "./scenarios/oblige.cfg"
map_d3 = "./scenarios/D3_battle.cfg"
map_match_shotgun = "./scenarios/deathmatch_shotgun.wad"
map_match_full = "./scenarios/full_deathmatch.wad"
map_my_way_home = "./scenarios/my_way_home.cfg"

'''
in first basic wad i using 30,45
in second map, i using origin pixel size
'''
resolution = (84, 84)
#resolution = (64, 64)
#resolution = (160 ,120)
#resolution = (60, 108)
#resolution = (30, 45)

resolution_dim = 3

# memory size
replay_memory_size = 100000
# learning hyperparameters
learning_rate = 0.00025
actor_lr = 0.00025
critic_lr = 0.0025
coeff_entropy = 0.01
coeff_value = 0.5
gamma = 0.99
horizon = 2048
nupdates = 5
clip_value = 0.2

epsilon = 1.0
dec_eps = 100000
min_eps = 0.01

train_episodes = 500
enjoy_episodes = 10
# Typical Range (Continuous): 512 - 5120
# Typical Range (Discrete): 32 - 512
batch_size = 64
# frame
frame_skip = 4
frame_repeat = 12
stack_size = 4
max_steps = 100


# parameters
deep_q_netowrk = 'dqn'
policy_gradient = 'pg'
actor_cirtic = 'ac'
double_dqn = 'ddqn'
ddpg = 'ddpg'
ddpg_p = 'ddqg_p'
ddpg_v = 'ddqg_v'
