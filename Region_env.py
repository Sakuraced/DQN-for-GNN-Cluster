import numpy as np
import networkx as nx
import copy
import random
import pandas as pd
from dateutil.parser import parse
from statsmodels.tsa.stattools import acf
from metrics import mape, avgACF, avgCoverage, Recall, calculate_acf
from copy import deepcopy
import folium
import time
import pickle as pkl
from sklearn.cluster import SpectralClustering


def reindex_graph_with_features(graph, stmatrix, component):
    """
    生成子图，和对应的时序信息，且子图编号从0,1,2...开始重新编码
    """
    # Extract the subgraph based on the specified component
    stmatrix=stmatrix.T
    subgraph = nx.subgraph(graph, component)

    # Reindex the graph nodes
    mapping = {node: new_id for new_id, node in enumerate(subgraph.nodes())}
    reindexed_graph = nx.relabel_nodes(subgraph, mapping)

    # Reorder the features according to the new node indices
    num_nodes = len(mapping)
    reindexed_features = [[] for _ in range(num_nodes)]
    for old_id, new_id in mapping.items():
        reindexed_features[new_id] = stmatrix[old_id]
    reindexed_features=np.array(reindexed_features).T
    return reindexed_graph, reindexed_features

def st_to_daily_pattern(matrix):
    """
    将矩阵的每列分割成24维的向量，并计算平均值。

    参数:
    - matrix: 输入的矩阵，一个NumPy数组。

    返回值:
    - averages: 每列转换成的24维向量的平均值，一个NumPy数组。
    """
    # 检查矩阵行数是否是24的倍数
    if matrix.shape[0] % 24 != 0:
        raise ValueError("矩阵的行数必须是24的倍数")

    # 初始化结果列表
    averages = []

    for col in range(matrix.shape[1]):
        column = matrix[:, col]
        vectors = np.split(column, matrix.shape[0] / 24)
        vector_averages = np.mean(vectors, axis=0)
        averages.append(vector_averages)

    # 将结果列表转换成NumPy数组并返回
    return np.array(averages)

class region_env:
    '''
    成员信息说明
    self.stmatrix:所有节点的时序信息
    self.daily_pattern:经过处理后的stmatrix，即每个节点的特征
    self.cluster:聚类信息，一个大列表中存储若干个小列表，每个小列表表示一类，每个小列表内部存储节点编号
    self.cluster_index:聚类信息，一个字典，其中键是节点编号，值是该节点所在的cluster的索引
    self.cluster_edge_index:列表，记录每个类中的边，即如果边的两个节点属于同一类，那么则在该列表中
    self.B:列表，记录记录两个类之间的边，即如果边的两个节点不属于同一类，那么则在该列表中
    '''
    def __init__(self):
        self.stmatrix_ = np.load('processed_data/processed_stmatrix.npy')[:-96 * 2 - 2016, :]
        self.daily_pattern = st_to_daily_pattern(self.stmatrix_)
        self.graph=nx.Graph(np.load('processed_data/processed_Chicago_am.npy'))
        self.edge_index = []
        for edge in self.graph.edges():
            self.edge_index.append(list(edge))
        self.area_array = np.load('processed_data/processed_area_info.npy')
        with open('initial_result/partition.pkl', 'rb') as fp:
            self.partition = pkl.load(fp)

    def reset(self):
        self.solver_index = 3 #选择编号为3的子图（最大子图）
        K, component = self.partition[self.solver_index]
        self.K,self.component=K, component

        self.subgraph, self.stmatrix= reindex_graph_with_features(self.graph, self.stmatrix_, component)
        partitions=[[] for i in range(int(self.K))]
        partition=nx.algorithms.community.asyn_fluid.asyn_fluidc(self.subgraph,k=int(self.K),max_iter=100)

        for i, c in enumerate(partition):
            partitions[i]=list(c)
        self.cluster = partitions
        self.cluster_index = self.create_lookup_table()
        self.acf_vec = [calculate_acf(self.stmatrix, i, 96) for i in self.cluster]
        self.initial_acf_vec = sum(self.acf_vec)
        print(sum(self.acf_vec),end=' ')

        self.clusters_edge_index = []
        for class_id in set(self.cluster_index.values()):
            for edge in self.subgraph.edges():
                node1, node2 = edge
                if self.cluster_index[node1] == class_id and self.cluster_index[node2] == class_id:
                    self.clusters_edge_index.append(list(edge))
        self.B=[]

        for edge in self.subgraph.edges():
            node1, node2 = edge
            if self.cluster_index[node1] != self.cluster_index[node2]:
                self.B.append([self.cluster_index[node1],node1,self.cluster_index[node2],node2])
        self.done = False

        sorted_items = sorted(self.cluster_index.items())

        sorted_list = [list(item) for item in sorted_items]
        return (self.clusters_edge_index, self.B,sorted_list),self.daily_pattern,len(self.subgraph.nodes()),self.K

    def step(self,action,B):
        _,node1,_,node2=B[action]
        p, q = self.cluster_index[node1], self.cluster_index[node2]
        self.cluster[p].remove(node1)
        self.cluster[q].append(node1)
        new_p=calculate_acf(self.stmatrix, self.cluster[p], 96)
        new_q=calculate_acf(self.stmatrix, self.cluster[q], 96)
        self.initial_acf_vec=sum(self.acf_vec)
        reward = (new_p + new_q - self.acf_vec[p] - self.acf_vec[q])*1000
        prob= new_p + new_q - self.acf_vec[p] - self.acf_vec[q]
        print(prob, end=' ')

        self.acf_vec[p] , self.acf_vec[q]= new_p,new_q


        self.cluster_index = self.create_lookup_table()
        self.clusters_edge_index = []
        for class_id in set(self.cluster_index.values()):
            # 创建一个空列表，用于存储当前类别的 edge_index
            class_edge_index = []

            # 遍历图中的所有边
            for edge in self.subgraph.edges():
                node1, node2 = edge
                if self.cluster_index[node1] == class_id and self.cluster_index[node2] == class_id:
                    class_edge_index.append(list(edge))

            self.clusters_edge_index.append(class_edge_index)

        self.B = []
        for edge in self.subgraph.edges():
            node1, node2 = edge
            if self.cluster_index[node1] != self.cluster_index[node2]:
                self.B.append([self.cluster_index[node1],node1,self.cluster_index[node2],node2])
        self.done = False

        merge = []
        for i in self.clusters_edge_index:
            merge += i
        sorted_items = sorted(self.cluster_index.items())
        sorted_list = [list(item) for item in sorted_items]
        return (merge,self.B,sorted_list), reward, prob, self.done

    def create_lookup_table(self):
        """
        创建一个查找表，用于快速找到元素所在的cluster的索引。

        参数:
        - clusters: 一个包含若干个小列表（cluster）的列表。

        返回值:
        - 一个字典，其中键是元素，值是该元素所在的cluster的索引。
        """
        lookup_table = {}
        for index, cluster in enumerate(self.cluster):
            for element in cluster:
                lookup_table[element] = index
        return lookup_table
    def create_lookup_temp_table(self,cluster):
        """
        创建一个查找表，用于快速找到元素所在的cluster的索引。

        参数:
        - clusters: 一个包含若干个小列表（cluster）的列表。

        返回值:
        - 一个字典，其中键是元素，值是该元素所在的cluster的索引。
        """
        lookup_table = {}
        for index, cluste in enumerate(cluster):
            for element in cluste:
                lookup_table[element] = index
        return lookup_table

    def greedy(self):
        '''
        找到当前状态下奖励最高的动作编号

        '''
        max_reward=-1000000
        num_act=-1
        self.B = []
        for edge in self.subgraph.edges():
            node1, node2 = edge
            if self.cluster_index[node1] != self.cluster_index[node2]:
                self.B.append([self.cluster_index[node1], node1, self.cluster_index[node2], node2])
        for index,(cluster1,node1,cluster2,node2) in enumerate(self.B):
            p, q = self.cluster_index[node1], self.cluster_index[node2]

            cluster1 = copy.deepcopy(self.cluster[p])
            cluster2 = copy.deepcopy(self.cluster[q])
            cluster1.remove(node1)
            cluster2.append(node1)
            new_p=calculate_acf(self.stmatrix, cluster1, 96)
            new_q=calculate_acf(self.stmatrix, cluster2, 96)

            reward = (new_p + new_q - self.acf_vec[p] - self.acf_vec[q])*1000
            #print(reward,end=' ')
            if reward>max_reward:
                max_reward=reward
                num_act=index
        return num_act

    def explore_step(self,action,B):
        _, node1, _, node2 = B[action]
        p, q = self.cluster_index[node1], self.cluster_index[node2]
        clus=copy.deepcopy(self.cluster)
        clus[p].remove(node1)
        clus[q].append(node1)
        new_p = calculate_acf(self.stmatrix, clus[p], 96)
        new_q = calculate_acf(self.stmatrix, clus[q], 96)
        reward = (new_p + new_q - self.acf_vec[p] - self.acf_vec[q]) * 1000
        prob = new_p + new_q - self.acf_vec[p] - self.acf_vec[q]
        cluster_index = self.create_lookup_temp_table(clus)
        self.clusters_edge_index = []
        for class_id in set(self.cluster_index.values()):

            for edge in self.subgraph.edges():
                node1, node2 = edge
                if cluster_index[node1] == class_id and cluster_index[node2] == class_id:
                    self.clusters_edge_index.append(list(edge))
        self.B = []
        for edge in self.subgraph.edges():
            node1, node2 = edge
            if cluster_index[node1] != cluster_index[node2]:
                self.B.append([cluster_index[node1], node1, cluster_index[node2], node2])
        self.done = False

        sorted_items = sorted(self.cluster_index.items())
        sorted_list = [list(item) for item in sorted_items]
        return (self.clusters_edge_index, self.B, sorted_list), reward, prob, self.done
if __name__ == '__main__':
    stmatrix = np.load('processed_data/processed_stmatrix.npy')
    stmatrix = stmatrix[:-96 * 2 - 2016, :]
    daily_pattern = st_to_daily_pattern(stmatrix)
    print(daily_pattern.shape)
    am = np.load('processed_data/processed_Chicago_am.npy')
    area_array = np.load('processed_data/processed_area_info.npy')
    graph = nx.Graph(am)
    best_area = 0
    best_demand = 0
    best_acf = 0
    with open('initial_result/partition.pkl', 'rb') as fp:
        partition = pkl.load(fp)
    acf_list = []
    for index in range(1, 51):
        flag = True
        final_result = []
        all_acf_vec = []
        for K, component in partition:
            subgraph = nx.subgraph(graph, component)
            greedyexp = GreedyExp(subgraph, stmatrix, K)
            greedyexp.setupExp()
            output_result = greedyexp.MainAlgorithm()
            all_acf_vec.append(sum(greedyexp.acf_vec))
            final_result.append(output_result)
        print(sum(all_acf_vec))

        node_which_cluster = [-1 for i in range(161)]
        for component in final_result:
            for ind, cluster in enumerate(component):
                subgraph = nx.subgraph(graph, cluster)
                if not nx.is_connected(subgraph):
                    flag = False
                for element in cluster:
                    node_which_cluster[element] = ind
        for label in node_which_cluster:
            if label == -1:
                flag = False
        for i in final_result:
            print(i, end=' ')
        print()

        if flag:
            with open('initial_result/greedy/{}.pkl'.format(index), 'wb') as fp:
                pkl.dump(final_result, fp)

    # print('平均coverage：{}'.format(avgcoverage))