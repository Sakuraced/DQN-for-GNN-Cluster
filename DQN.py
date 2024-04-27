import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer



# -------------------------------------- #
# 构造深度学习网络，输入状态s，得到各个动作的reward
# -------------------------------------- #

from torch_scatter import scatter_add
from functools import partial
import torch.optim as optim

embed_dim = 128
import torch.nn.functional as F

import torch
from torch_geometric.nn import GCNConv
class S2V_GCN(torch.nn.Module):
    def __init__(self,num_features,out_features,device):
        super(S2V_GCN, self).__init__()
        self.device=device
        self.conv1 = GCNConv(num_features, 96)  #输入=节点特征维度，16是中间隐藏神经元个数
        self.conv2 = GCNConv(96,out_features)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
class S2V(nn.Module):
    def __init__(self, in_dim, out_dim,device,vertex_dim=24):
        super(S2V, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device=device
        Linear = partial(nn.Linear, bias=False)
        self.lin1 = Linear(vertex_dim, out_dim)
        self.lin2 = Linear(in_dim, out_dim)
    def forward(self, mu, x, edge_index):
        #first part of eq. 3

        x = self.lin1(x)
        if mu!=None:
            edge_index=edge_index.long()
            mu_j = mu[edge_index[1, :].long(), :]
            mu_aggr = scatter_add(mu_j, edge_index[0, :].long(), dim=0)
            mu_aggr = self.lin2(mu_aggr)
            return F.relu(x + mu_aggr)
        else: return F.relu(x)



# Q function
class Net(nn.Module):
    def __init__(self, hid_dim, ALPHA,device,graph_size,num_cluster):
        super(Net, self).__init__()
        self.hid_dim = hid_dim
        self.num_cluster=num_cluster
        Linear = partial(nn.Linear, bias=False)
        self.lin1 = Linear(2*hid_dim, 1)
        self.lin2=Linear(hid_dim, hid_dim)
        self.lin3 = Linear(4*hid_dim, hid_dim)
        self.lin4 = Linear(hid_dim, hid_dim)
        self.lin5 = Linear(hid_dim, hid_dim)
        self.lin6 = Linear(hid_dim, hid_dim)
        self.graph_size=graph_size
        self.S2V_GCN=S2V_GCN(num_features=24, out_features=hid_dim,device=device)
        self.loss = nn.MSELoss
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.graph_size=graph_size
        self.device = device
        self.to(self.device)

    def forward(self, x, clusters_edge_index, B, clusters):
        # mu has shape [batch_size*N, in_dim]
        # x has shape [batch_size*N, 1]
        # edge_index has shape [Es, 2]
        # edge_w has shape [Es, 1]
        x=self.S2V_GCN(x.float(), clusters_edge_index.long())
        nodes_vec=x
        summed_features = torch.zeros((self.num_cluster, x.shape[1]), dtype=x.dtype,device=x.device)
        # 遍历每个组
        temp=[[] for i in range(self.num_cluster)]
        for i,j in clusters:
            temp[j].append(i.item())
        clusters=temp
        for i, group in enumerate(clusters):
            summed_features[i] = torch.sum(nodes_vec[group], dim=0)
        cluster_vec=summed_features
        Q_tensor = torch.empty(0, 2*self.hid_dim)
        Q_tensor=Q_tensor.to(self.device)
        graph_vec = torch.sum(nodes_vec, dim=0)
        graph_vec = self.lin2(graph_vec)

        for i,j,k,l in B:
            temp=torch.cat((self.lin5(cluster_vec[i]),self.lin4(nodes_vec[j]),self.lin5(cluster_vec[k]),self.lin4(nodes_vec[l])),dim=0)
            temp=self.lin3(torch.relu(temp))

            temp=torch.cat((graph_vec,temp),dim=0)
            temp=temp.unsqueeze(0)
            Q_tensor=torch.cat((Q_tensor,temp),dim=0)
        Q_tensor = self.lin1(torch.relu(Q_tensor))



        return Q_tensor


# -------------------------------------- #
# 构造深度强化学习模型
# -------------------------------------- #

class DQN:
    # （1）初始化
    def __init__(self,
                 learning_rate, gamma, epsilon,
                 target_update, device,graph_size,num_cluster):
        # 属性分配
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device  # 在GPU计算
        # 计数器，记录迭代次数
        self.count = 0

        # 构建2个神经网络，相同的结构，不同的参数
        # 实例化训练网络  [b,4]-->[b,2]  输出动作对应的奖励
        self.q_net = Net(ALPHA=0.1,hid_dim=64, device=device,graph_size=graph_size,num_cluster=num_cluster)
        # 实例化目标网络
        self.target_q_net = Net(ALPHA=0.1,hid_dim=64, device=device,graph_size=graph_size,num_cluster=num_cluster)

        # 优化器，更新训练网络的参数


    # （3）网络训练
    def update(self, replay_buffer,batch_size,device):  # 传入经验池中的batch个样本
        self.q_net.optimizer.zero_grad()
        dqn_loss = 0
        for _ in range(32):

            graph_batch = replay_buffer.sample_buffer(1)
            graph_batch = graph_batch.to(device)
            x = graph_batch.x_attr.squeeze(0)
            cluster=graph_batch.cluster.squeeze(0)

            B = graph_batch.B.squeeze(0)
            clusters_edge_index = graph_batch.edge_index.squeeze(0)

            action = graph_batch.action.squeeze(0)
            new_cluster=graph_batch.new_cluster.squeeze(0)
            new_clusters_edge_index=graph_batch.new_edge_index.squeeze(0)
            new_B=graph_batch.new_B.squeeze(0)

            done = graph_batch.done.squeeze(0)
            reward = graph_batch.reward.squeeze(0)

            q_values = self.q_net(x,clusters_edge_index,B,cluster).T.squeeze(0)[action.item()]# [b,1]
            double_Q=self.q_net(x,new_clusters_edge_index,new_B,new_cluster).T.squeeze(0)
            double_q_value, double_q_action = torch.sort(double_Q, descending=True)
            action=double_q_action[0]
            Q=self.target_q_net(x,new_clusters_edge_index,new_B,new_cluster).T.squeeze(0)
            max_next_q_values=Q[action]
            # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报

            q_targets = reward + self.gamma * max_next_q_values * (1 - done)
            dqn_loss += torch.mean(F.mse_loss(q_values.float(), q_targets.float()))

        print(dqn_loss.item(),end=' ')
        dqn_loss.backward()
            # 对训练网络更新
        self.q_net.optimizer.step()

        # 在一段时间后更新目标网络的参数
        if self.count % self.target_update == 0:
            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())

        self.count += 1
        if self.epsilon<0.90:
            self.epsilon=self.epsilon+0.0004
    def take_action(self, state, x):
        """
        edge_index: [E, 2]
        state: [N, 1]
        """


        clusters_edge_index, B, cluster= state
        clusters_edge_index = torch.tensor(clusters_edge_index).to(self.q_net.device)
        clusters_edge_index = clusters_edge_index.transpose(0,1)
        x = torch.tensor(np.array(x)).to(self.q_net.device)

        cluster=torch.tensor(np.array(cluster)).to(self.q_net.device)
        Q = self.q_net(x, clusters_edge_index, B, cluster)
        # make sure select new nodes
        if np.random.rand() > self.epsilon:
            action = np.random.choice([i for i in range(len(B))], size=1)[0]
        else:
            q_value, q_action = torch.sort((Q.T).squeeze(0), descending=True)
            action = q_action[0]
            action = action.item()
        print(B[action],end=' ')
        return action
