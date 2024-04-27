import gym
import numpy as np

from DQN import DQN
from replay_buffer import ReplayBuffer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from Region_env import region_env
import time
# GPU运算
device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")

# ------------------------------- #
# 全局变量
# ------------------------------- #

capacity = 100000  # 经验池容量
lr = 2e-3  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.9  # 贪心系数
target_update = 100  # 目标网络的参数的更新频率
batch_size = 1
n_hidden = 128  # 隐含层神经元个数
min_size = 100000  # 经验池超过200后再训练
return_list = []  # 记录每个回合的回报

# 加载环境
env = region_env()
state, clusters_node_feature,graph_size,K = env.reset()
# 实例化经验池
replay_buffer = ReplayBuffer(capacity)
# 实例化DQN
agent = DQN(
            num_cluster=K,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            target_update=target_update,
            device=device,
            graph_size=graph_size
            )

# 训练模型
num_transition=0
'''
以下是采样步骤
'''
while num_transition<min_size:
    state, clusters_node_feature, graph_size, K = env.reset()
    total_return = 0
    while True:

        for actions in range(len(state[1])):
            next_state, reward, prob, done = env.explore_step(actions, state[1])
            replay_buffer.store_transition(mu=clusters_node_feature,
                                       edge_index=state[0],
                                       B=state[1],
                                       reward=reward,
                                       action=actions,
                                       new_edge_index=next_state[0],
                                       cluster=state[2],
                                       new_cluster=next_state[2],
                                       new_B=next_state[1],
                                       prob=prob,
                                       done=done)
        # 更新当前状态
            num_transition+=1
        max_action = env.greedy()
        next_state, reward, prob, done = env.step(max_action, state[1])
        state=next_state
        total_return+=reward

        if reward<0:
            print(total_return)#采样到delta_ACF小于0时停止，输出总奖励
            break
"""
训练步骤
"""
for i in range(500):  # 100回合
    # 每个回合开始前重置环境
    print(i)
    state, clusters_node_feature,graph_size,K = env.reset() # len=4
    # 记录每个回合的回报
    episode_return = 0
    done = False

    count=1
    while True:
        # 获取当前状态下需要采取的动作

        action = agent.take_action(state,clusters_node_feature)
        # 更新环境

        next_state, reward, prob, done= env.step(action,state[1])

        # 添加经验池
        # replay_buffer.store_transition(mu=clusters_node_feature,
        #                                edge_index=state[0],
        #                                B=state[1],
        #                                reward=reward,
        #                                action=action,
        #                                new_edge_index=next_state[0],
        #                                cluster=state[2],
        #                                new_cluster=next_state[2],
        #                                new_B=next_state[1],
        #                                prob=prob,
        #                                done=done)
        # 更新当前状态
        state = next_state
        # 更新回合回报
        episode_return += reward

        # 当经验池超过一定数量后，训练网络
        if len(replay_buffer) > 0:
            # 从经验池中随机抽样作为训练集


            # 网络更新
            agent.update(replay_buffer,batch_size,device)
        # 找到目标就结束
        count+=1
        if count>50: break

    # 记录每个回合的回报
    return_list.append(episode_return)
    print()
    print(episode_return)

# 绘图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN Returns')
plt.show()
