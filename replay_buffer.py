import numpy as np
import torch
from torch_geometric.data import Data, Batch
"""
Store trajectories using Data class
mini-Batch using Batch
"""
class ReplayBuffer:
    def __init__(self, max_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.graph_memory = [None] * self.mem_size
    print(1)
    def store_transition(self,  mu, edge_index, B, action, reward, new_edge_index,new_B, cluster,new_cluster,prob,done):
        graph = Data(edge_index=torch.tensor(edge_index).transpose(0,1))
        graph.edge_index=torch.tensor(edge_index).transpose(0,1)
        graph.x_attr = torch.tensor(mu)
        graph.B = torch.tensor(B)
        graph.prob=torch.tensor(prob)
        graph.cluster=torch.tensor(cluster)
        graph.action = torch.tensor(action)
        graph.reward = torch.tensor(reward)
        graph.new_cluster=torch.tensor(new_cluster)
        graph.new_edge_index = torch.tensor(new_edge_index).transpose(0,1)
        graph.new_B=torch.tensor(new_B)
        if done:
            done=1
        else:
            done=0
        graph.done = torch.tensor(done)

        index = self.mem_cntr % self.mem_size
        self.graph_memory[index] = graph
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        probabilities = [abs(self.graph_memory[i].prob) for i in range(max_mem)]

        # 确保概率总和为1
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        # 根据提供的概率进行采样
        batch = np.random.choice(max_mem, batch_size, replace=False, p=probabilities)

        graph_list = [self.graph_memory[b] for b in batch]
        keys = graph_list[0].keys
        return Batch.from_data_list(graph_list)

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)

