import numpy as np
from setting import SetParameter
import random
import torch
from torch_geometric.data import Data
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging
import os.path as osp
from sklearn.neighbors import KDTree, BallTree
from scipy.interpolate import interp1d
random.seed(1933)
np.random.seed(1933)
config = SetParameter()


# %=======================================================================================


class TrainSTData(Dataset):
    def __init__(self, node, coor, d2vec, traj_idx,  traj_dis):
        self.node = node
        self.coor = coor
        self.d2vec = d2vec
        self.traj_idx = traj_idx
        self.traj_dis = traj_dis

    def __len__(self):
        return len(self.node)

    def __getitem__(self, idx):
        return self.node[idx], self.coor[idx], self.d2vec[idx], self.traj_idx[idx], self.traj_dis[idx], idx


def load_st_traindata(train_file, node_coor):
    data = np.load(train_file, allow_pickle=True)
    node = data["train_node"]  # [30000,traj]
    coor = data["train_coor"]  # [30000,traj]
    d2vec = data["train_d2vec"]
    # coe = data["coe"]

    logging.info("use all node mean and std")
    # logging.info(f'coe:{coe}')
    meanx, meany, stdx, stdy = node_coor['mean_lng'], node_coor['mean_lat'], node_coor['std_lng'], node_coor['std_lat']
    coor = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
             for r in t] for t in coor]

    traj_idx = data["traj_idx"]  # [30000,2,10,traj]
    traj_dis = data["traj_dis"]

    return node, coor, d2vec, traj_idx, traj_dis


def train_st_data_loader(train_file, batchsize, node_coor):
    def collate_fn_neg(data_tuple):

        # data, label, neg_label, dis, neg_dis, idx_list = data_tuple
        node_aco = []
        coor_aco = []
        d2vec_aco = []
        node_pos = []
        coor_pos = []
        d2vec_pos = []
        node_neg = []
        coor_neg = []
        d2vec_neg = []
        pos_dis_list = []
        neg_dis_list = []
        # for idx, d in enumerate(data):
        for i, (node, coor, d2vec, label, dis, idx) in (enumerate(data_tuple)):
            # aco_idx = idx_list[idx]

            # pos_sampled_indices = np.random.randint(1, 11, size=config.pos_num)
            sam_num = min(20, len(label))
            pos_sampled_indices = np.random.choice(np.arange(1, sam_num//2), config.pos_num, replace=True)
            pos_sampled_indices = np.sort(pos_sampled_indices)
            # neg_sampled_indices = np.random.randint(11, len(label), size=config.pos_num)
            neg_sampled_indices = np.random.choice(np.arange(sam_num//2, len(label)), config.pos_num, replace=True)
            neg_sampled_indices = np.sort(neg_sampled_indices)
            pos_idx = label[pos_sampled_indices]
            neg_idx = label[neg_sampled_indices]
            pos_dis = dis[pos_sampled_indices]
            neg_dis = dis[neg_sampled_indices]

            for j in range(config.pos_num):
                node_aco.append(torch.LongTensor(node))
                coor_aco.append(torch.tensor(coor, dtype=torch.float32))
                d2vec_aco.append(torch.tensor(d2vec, dtype=torch.float32))

                node_pos.append(torch.LongTensor(node_list[pos_idx[j]]))
                coor_pos.append(torch.tensor(coor_list[pos_idx[j]], dtype=torch.float32))
                d2vec_pos.append(torch.tensor(d2vec_list[pos_idx[j]], dtype=torch.float32))
                pos_dis_list.append(pos_dis[j])

                node_neg.append(torch.LongTensor(node_list[neg_idx[j]]))
                coor_neg.append(torch.tensor(coor_list[neg_idx[j]], dtype=torch.float32))
                d2vec_neg.append(torch.tensor(d2vec_list[neg_idx[j]], dtype=torch.float32))
                neg_dis_list.append(neg_dis[j])

        pos_dis_list = torch.tensor(pos_dis_list)
        neg_dis_list = torch.tensor(neg_dis_list)
        aco_length = torch.tensor(list(map(len, node_aco)))
        pos_length = torch.tensor(list(map(len, node_pos)))
        neg_length = torch.tensor(list(map(len, node_neg)))

        node_aco = rnn_utils.pad_sequence(node_aco, batch_first=True, padding_value=0)
        node_pos = rnn_utils.pad_sequence(node_pos, batch_first=True, padding_value=0)
        node_neg = rnn_utils.pad_sequence(node_neg, batch_first=True, padding_value=0)
        coor_aco = rnn_utils.pad_sequence(coor_aco, batch_first=True, padding_value=0)
        coor_pos = rnn_utils.pad_sequence(coor_pos, batch_first=True, padding_value=0)
        coor_neg = rnn_utils.pad_sequence(coor_neg, batch_first=True, padding_value=0)
        d2vec_aco = rnn_utils.pad_sequence(d2vec_aco, batch_first=True, padding_value=0)
        d2vec_pos = rnn_utils.pad_sequence(d2vec_pos, batch_first=True, padding_value=0)
        d2vec_neg = rnn_utils.pad_sequence(d2vec_neg, batch_first=True, padding_value=0)
        return (node_aco, coor_aco, d2vec_aco,
                node_pos, coor_pos, d2vec_pos,
                node_neg, coor_neg, d2vec_neg,
                pos_dis_list, neg_dis_list,
                aco_length, pos_length, neg_length)

    node_list, coor_list, d2vec_list, traj_idx, traj_dis = load_st_traindata(train_file, node_coor)
    logging.info(f'train size:{len(node_list)}')
    data_ = TrainSTData(node_list, coor_list, d2vec_list, traj_idx, traj_dis)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn_neg)

    return dataset

# %=======================================================================================


def load_st_validata(vali_file, node_coor):
    data = np.load(vali_file, allow_pickle=True)
    node = data["vali_node"]
    coor = data["vali_coor"]
    d2vec = data["vali_d2vec"]
    y_idx = data["vali_y"]

    logging.info("use all node mean and std")
    meanx, meany, stdx, stdy = node_coor['mean_lng'], node_coor['mean_lat'], node_coor['std_lng'], node_coor['std_lat']

    coor = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
             for r in t] for t in coor]
    return node, coor, d2vec, y_idx


class ValiSTData(Dataset):
    def __init__(self, data, coor, d2vec, label):
        self.data = data
        self.coor = coor
        self.d2vec = d2vec
        self.label = label
        self.index = list(range(len(data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tuple_ = (self.data[idx], self.coor[idx], self.d2vec[idx], self.label[idx], self.index[idx])
        return tuple_


def vali_st_data_loader(vali_file, batchsize, node_coor):
    def collate_fn_neg(data_tuple):
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        coor = [torch.tensor(sq[1], dtype=torch.float32) for sq in data_tuple]
        d2vec = [torch.tensor(sq[2], dtype=torch.float32) for sq in data_tuple]
        label = [sq[3] for sq in data_tuple]
        idx = [sq[4] for sq in data_tuple]
        data_length = torch.tensor(list(map(len, data)))
        label = torch.tensor(np.array(label))
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        coor = rnn_utils.pad_sequence(coor, batch_first=True, padding_value=0)
        d2vec = rnn_utils.pad_sequence(d2vec, batch_first=True, padding_value=0)
        return data, coor, d2vec, label, data_length, idx

    val_x, coor_x, d2vec_x, val_y = load_st_validata(vali_file, node_coor)
    data_ = ValiSTData(val_x, coor_x, d2vec_x, val_y)
    dataset = DataLoader(
        data_,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate_fn_neg,
        drop_last=False,
    )

    return dataset, len(val_x)
# %==========================================================================


def load_region_node(dataset):
    file = osp.join('ground_truth', dataset, 'distance_to_anchor_node.pt')
    region_node = torch.load(file)
    logging.info(f"distance to anchor node: {file}")
    return region_node


def load_neighbor(dataset, knn):

    k5_neighbor = np.load(f'dataset/{dataset}/{str(knn)}/k{str(knn)}_neighbor.npy')
    k5_distance = np.load(f'dataset/{dataset}/{str(knn)}/k{str(knn)}_distance.npy')
    k10_neighbor = np.load(f'dataset/{dataset}/{str(knn)}/k{str(knn*2)}_neighbor.npy')
    k10_distance = np.load(f'dataset/{dataset}/{str(knn)}/k{str(knn*2)}_distance.npy')
    k15_neighbor = np.load(f'dataset/{dataset}/{str(knn)}/k{str(knn*3)}_neighbor.npy')
    k15_distance = np.load(f'dataset/{dataset}/{str(knn)}/k{str(knn*3)}_distance.npy')

    node_embeddings_file = f"data/{dataset}/node_features.npy"
    logging.info(f"use Node2Vec: node embeddings file: {node_embeddings_file}")
    node_embeddings = np.load(node_embeddings_file)

    node = pd.read_csv(config.node_file)
    lng = node['lng'].values
    lat = node['lat'].values

    mean_lng, mean_lat = np.mean(lng), np.mean(lat)
    std_lng, std_lat = np.std(lng), np.std(lat)

    node_coor = {'mean_lng': mean_lng, 'mean_lat': mean_lat,
                 'std_lng': std_lng, 'std_lat': std_lat, }

    edge_index_l0 = torch.LongTensor(k5_neighbor).t().contiguous()  # [2,556704]
    edge_attr_l0 = torch.tensor(k5_distance, dtype=torch.float)  # [556704]
    edge_index_l1 = torch.LongTensor(k10_neighbor).t().contiguous()
    edge_attr_l1 = torch.tensor(k10_distance, dtype=torch.float)
    edge_index_l2 = torch.LongTensor(k15_neighbor).t().contiguous()
    edge_attr_l2 = torch.tensor(k15_distance, dtype=torch.float)
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)

    logging.info(f"node embeddings shape: {node_embeddings.shape}")
    logging.info(f"edge_index_l0 shape: {edge_index_l0.shape}")
    logging.info(f"edge_attr_l0 shape: {edge_attr_l0.shape}")
    logging.info(f"edge_index_l1 shape: {edge_index_l1.shape}")
    logging.info(f"edge_attr_l1 shape: {edge_attr_l1.shape}")
    logging.info(f"edge_index_l2 shape: {edge_index_l2.shape}")
    logging.info(f"edge_attr_l2 shape: {edge_attr_l2.shape}")

    edge_index_l0 = edge_index_l0[[1, 0], :]
    edge_index_l1 = edge_index_l1[[1, 0], :]
    edge_index_l2 = edge_index_l2[[1, 0], :]

    road_network_l0 = Data(x=node_embeddings, edge_index=edge_index_l0, edge_attr=edge_attr_l0)
    road_network_l1 = Data(x=[], edge_index=edge_index_l1, edge_attr=edge_attr_l1)
    road_network_l2 = Data(x=[], edge_index=edge_index_l2, edge_attr=edge_attr_l2)

    return [road_network_l0, road_network_l1, road_network_l2], node_coor


# %=====================================================================================
# 预训练
class GraphSTTrainData(Dataset):
    def __init__(self, knn_neighbor, knn_distance):
        self.knn_neighbor = knn_neighbor
        self.knn_distance = knn_distance
        self.node_idx = list(range(len(knn_neighbor)))

    def __len__(self):
        return len(self.node_idx)

    def __getitem__(self, idx):
        return self.node_idx[idx], self.knn_neighbor[idx], self.knn_distance[idx]


def graph_node_dist_loader(knn_neighbor_file, knn_distance_file, batch_size, knn_neighbor_num, knn_distance_coe):
    def collate_fn_neg(data_tuple):

        node_aco = []
        node_pos = []
        node_neg = []
        node_pos_dis = []
        node_neg_dis = []

        node_coor_idx = []
        for i, (node_aco_idx, node_idx, node_dis) in (enumerate(data_tuple)):

            if len(node_idx) >= 2:

                max_samples = min(20, len(node_idx))
                num_samples = max_samples // 2 * 2  

                sampled_indices = np.random.choice(len(node_idx), num_samples, replace=False)
                sampled_indices = np.sort(sampled_indices)
                sampled_node_idx = node_idx[sampled_indices]
                sampled_node_dis = node_dis[sampled_indices]
           
                mid_point = num_samples // 2

  
                pos_indices = sampled_node_idx[:mid_point]
                pos_distances = sampled_node_dis[:mid_point]


                neg_indices = sampled_node_idx[mid_point:]
                neg_distances = sampled_node_dis[mid_point:]
                # =====
                node_aco.extend([node_aco_idx] * mid_point)
                node_pos.extend(pos_indices)
                node_pos_dis.extend(pos_distances)
                node_neg.extend(neg_indices)
                node_neg_dis.extend(neg_distances)

                node_coor_idx.append(node_aco_idx)
        node_aco = torch.LongTensor(node_aco)
        node_pos = torch.LongTensor(node_pos)
        node_neg = torch.LongTensor(node_neg)
        node_pos_dis = torch.tensor(node_pos_dis, dtype=torch.float)
        node_neg_dis = torch.tensor(node_neg_dis, dtype=torch.float)

        node_coor_idx = torch.LongTensor(node_coor_idx)
        return (node_aco, node_pos, node_neg,
                node_pos_dis, node_neg_dis, node_coor_idx)

    knn_neighbor = np.load(knn_neighbor_file, allow_pickle=True)
    knn_distance = np.load(knn_distance_file, allow_pickle=True)

    for i in range(len(knn_distance)):
        if knn_distance[i] is not None and len(knn_distance[i]) > 0:
            knn_neighbor[i] = knn_neighbor[i][:knn_neighbor_num]
            knn_distance[i] = knn_distance[i][:knn_neighbor_num]
            knn_distance[i] = np.array(knn_distance[i]) / knn_distance_coe

    data_ = GraphSTTrainData(knn_neighbor, knn_distance)
    dataset = DataLoader(data_, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_neg)

    return dataset

# %=====================================================================================


def filter_trajectories(node_list, coor_list, kseg_list, d2vec_list, max_trajectories=1500, max_length=500, min_length=100):
    logging.info(f"filter trajectories: max_trajectories={max_trajectories}, max_length={max_length}, min_length={min_length}")
    filtered_node_list = []
    filtered_coor_list = []
    filtered_kseg_list = []
    filtered_d2vec_list = []
    for traj in zip(node_list, coor_list, kseg_list, d2vec_list):
        if len(filtered_node_list) >= max_trajectories:
            break
        if len(traj[0]) <= max_length and len(traj[0]) >= min_length:
            filtered_node_list.append(traj[0])
            filtered_coor_list.append(traj[1])
            filtered_kseg_list.append(traj[2])
            filtered_d2vec_list.append(traj[3])
    filtered_node_list = np.array(filtered_node_list, dtype=object)
    filtered_coor_list = np.array(filtered_coor_list, dtype=object)
    filtered_kseg_list = np.array(filtered_kseg_list, dtype=np.float32)
    filtered_d2vec_list = np.array(filtered_d2vec_list, dtype=object)
    return filtered_node_list, filtered_coor_list, filtered_kseg_list, filtered_d2vec_list


class EmbDistSTTrainData(Dataset):
    def __init__(self, node, coor, d2vec, tree_dist, tree_idx):
        self.node = node
        self.coor = coor
        self.d2vec = d2vec
        self.tree_dist = tree_dist
        self.tree_idx = tree_idx
        # self.idx = list(range(len(data)))

    def __len__(self):
        return len(self.node)

    def __getitem__(self, idx):
        return self.node[idx], self.coor[idx], self.d2vec[idx], self.tree_dist[idx], self.tree_idx[idx], idx


def load_st_traj_emb_dist_data(shuffle_node_file,
                               shuffle_coor_file,
                               shuffle_kseg_file,
                               shuffle_d2vec_file, node_coor):

    node_list = np.load(shuffle_node_file, allow_pickle=True)[:config.train_set_size]
    coor_list = np.load(shuffle_coor_file, allow_pickle=True)[:config.train_set_size]
    kseg_list = np.load(shuffle_kseg_file, allow_pickle=True)[:config.train_set_size]
    d2vec_list = np.load(shuffle_d2vec_file, allow_pickle=True)[:config.train_set_size]
    node_list, coor_list, kseg_list, d2vec_list = filter_trajectories(
        node_list, coor_list, kseg_list, d2vec_list,
        max_trajectories=config.pretrain['TrajNum'],
        max_length=config.pretrain['TrajMaxLen'],
        min_length=config.pretrain['TrajMinLen'])

    kseg_list_out = kseg_list.reshape((kseg_list.shape[0], -1))  # shape: [1500, 10]
    kdtree = KDTree(kseg_list_out, leaf_size=40)
    tree_sample_dist, tree_sample_idx = kdtree.query(kseg_list_out, k=len(kseg_list_out))
    norm_dis = np.max(tree_sample_dist)
    tree_sample_dist = tree_sample_dist / norm_dis * config.pretrain_coe  # 归一化距离
    meanx, meany, stdx, stdy = node_coor['mean_lng'], node_coor['mean_lat'], node_coor['std_lng'], node_coor['std_lat']
    coor_list = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                  for r in t] for t in coor_list]

    return node_list, coor_list, d2vec_list, tree_sample_dist, tree_sample_idx


def traj_st_emb_dist_loader(node_list_file,
                            coor_list_file,
                            kseg_list_file,
                            d2vec_list_file,
                            batch_size, num_nodes, node_coor):
    def collate_fn_fun(data_tuple):

        node_aco = []
        coor_aco = []
        d2vec_aco = []

        node_pos = []
        coor_pos = []
        d2vec_pos = []
        dist_pos = []

        node_neg = []
        coor_neg = []
        d2vec_neg = []
        dist_neg = []

        node_tgt = []
        coor_tgt = []
        node_lbl = []
        coor_lbl = []
        for i, (node, coor, d2vec, sample_dist, sample_idx, idx) in (enumerate(data_tuple)):
            # %==========


            num_samples = config.pretrain['TrajSamNum']
   
            sampled_indices = np.random.choice(len(train_node), num_samples, replace=False)

            sampled_indices = np.sort(sampled_indices)
            sampled_node_idx = sample_idx[sampled_indices]
            sampled_node_dis = sample_dist[sampled_indices]

            pos_node_idx = sampled_node_idx[:num_samples // 2]
            neg_node_idx = sampled_node_idx[num_samples // 2:]
            pos_node_dis = sampled_node_dis[:len(sampled_node_dis)//2]
            neg_node_dis = sampled_node_dis[len(sampled_node_dis)//2:]
            # %==========
            for j in range(len(pos_node_idx)):
                node_aco.append(torch.LongTensor(node))
                coor_aco.append(torch.tensor(coor, dtype=torch.float32))
                d2vec_aco.append(torch.tensor(d2vec, dtype=torch.float32))

                pos_node_seq = train_node[pos_node_idx[j]]
                pos_coor_seq = train_coor[pos_node_idx[j]]
                pos_d2vec_seq = train_d2vec[pos_node_idx[j]]
                node_pos.append(torch.LongTensor(pos_node_seq))
                coor_pos.append(torch.tensor(pos_coor_seq, dtype=torch.float32))
                d2vec_pos.append(torch.tensor(pos_d2vec_seq, dtype=torch.float32))
                dist_pos.append(pos_node_dis[j])

                neg_node_seq = train_node[neg_node_idx[j]]
                neg_coor_seq = train_coor[neg_node_idx[j]]
                neg_d2vec_seq = train_d2vec[neg_node_idx[j]]
                node_neg.append(torch.LongTensor(neg_node_seq))
                coor_neg.append(torch.tensor(neg_coor_seq, dtype=torch.float32))
                d2vec_neg.append(torch.tensor(neg_d2vec_seq, dtype=torch.float32))
                dist_neg.append(neg_node_dis[j])

            start_node = torch.LongTensor([num_nodes])
            traj_node_tgt_idx = torch.cat([start_node, torch.LongTensor(node[:-1])])
            start_coor = torch.tensor([[0, 0]], dtype=torch.float32)
            traj_coor_tgt_idx = torch.cat([start_coor, torch.tensor(coor[:-1], dtype=torch.float32)])

            node_tgt.append(traj_node_tgt_idx)
            coor_tgt.append(traj_coor_tgt_idx)
            node_lbl.append(torch.LongTensor(node))
            coor_lbl.append(torch.tensor(coor, dtype=torch.float32))

        dist_pos = torch.tensor(dist_pos, dtype=torch.float)
        dist_neg = torch.tensor(dist_neg, dtype=torch.float)
        aco_length = torch.tensor(list(map(len, node_aco)))
        pos_length = torch.tensor(list(map(len, node_pos)))
        neg_length = torch.tensor(list(map(len, node_neg)))
        node_aco = rnn_utils.pad_sequence(node_aco, batch_first=True, padding_value=0)
        coor_aco = rnn_utils.pad_sequence(coor_aco, batch_first=True, padding_value=0)
        d2vec_aco = rnn_utils.pad_sequence(d2vec_aco, batch_first=True, padding_value=0)
        node_pos = rnn_utils.pad_sequence(node_pos, batch_first=True, padding_value=0)
        coor_pos = rnn_utils.pad_sequence(coor_pos, batch_first=True, padding_value=0)
        d2vec_pos = rnn_utils.pad_sequence(d2vec_pos, batch_first=True, padding_value=0)
        node_neg = rnn_utils.pad_sequence(node_neg, batch_first=True, padding_value=0)
        coor_neg = rnn_utils.pad_sequence(coor_neg, batch_first=True, padding_value=0)
        d2vec_neg = rnn_utils.pad_sequence(d2vec_neg, batch_first=True, padding_value=0)
        # %====
        node_tgt = rnn_utils.pad_sequence(node_tgt, batch_first=True, padding_value=0)
        coor_tgt = rnn_utils.pad_sequence(coor_tgt, batch_first=True, padding_value=0)
        node_lbl = rnn_utils.pad_sequence(node_lbl, batch_first=True, padding_value=0)
        coor_lbl = rnn_utils.pad_sequence(coor_lbl, batch_first=True, padding_value=0)
        return (node_aco, coor_aco, d2vec_aco,
                node_pos, coor_pos, d2vec_pos, dist_pos,
                node_neg, coor_neg, d2vec_neg, dist_neg,
                aco_length, pos_length, neg_length,
                node_tgt, coor_tgt, node_lbl, coor_lbl)

    train_node, train_coor, train_d2vec, tree_dist, tree_idx = load_st_traj_emb_dist_data(
        shuffle_node_file=node_list_file,
        shuffle_coor_file=coor_list_file,
        shuffle_kseg_file=kseg_list_file,
        shuffle_d2vec_file=d2vec_list_file, node_coor=node_coor)
    logging.info(f'traj emb dist train size:{len(train_node)}')
    data_ = EmbDistSTTrainData(train_node, train_coor, train_d2vec, tree_dist, tree_idx)
    dataset = DataLoader(data_, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_fun)
    return dataset
