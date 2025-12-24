import pandas as pd
from setting import SetParameter
from model_network import AttnPooling, GraphEncoder, GraphPretrainEncoder,  GraphTrajSimEncoder, TrajCoorEncoder, TrajNodeEncoder
from spatial_data_utils import traj_emb_dist_loader, load_region_node, load_neighbor, load_network, graph_node_dist_loader, train_data_loader, vali_data_loader
import torch
import spatial_data_utils
from lossfun import GraphSimLossFun, SpaLossFun, TrajLossFun, TrajSimLossFun
import time
from tqdm import tqdm
import numpy as np
import logging
import os.path as osp
import os
from typing import Optional
import test_method
import torch.nn as nn
from itertools import cycle


config = SetParameter()


class GraphPretrain_Trainer(object):
    def __init__(self, save_folder: Optional[str] = None):
        self.save_folder = save_folder if save_folder else config.save_folder
        self.epochs = config.pretrain_epochs
        self.dataset = config.dataset
        self.device = config.device
        self.num_knn = config.num_knn
        self.learning_rate = config.pretrain_learning_rate

        self.usePI = config.gtraj["usePI"]
        self.useSI = config.gtraj["useSI"]

        self.train_batch = config.train_batch
        self.test_batch = config.test_batch
        self.pretrain_node_batch = config.pretrain_node_batch
        self.pretrain_traj_batch = config.pretrain_traj_batch
        self.num_nodes = config.truenum[config.dataset]

        self.useGraphNodeDist = config.pretrain["useGraphNodeDist"]
        self.useGraphCoorPred = config.pretrain["useGraphCoorPred"]

        self.useTrajEmbDist = config.pretrain["useTrajEmbDist"]
        self.useTrajAutoRegNode = config.pretrain["useTrajAutoRegNode"]
        self.useTrajAutoRegCoor = config.pretrain["useTrajAutoRegCoor"]
        self.net = GraphPretrainEncoder(useGraphNodeDist=self.useGraphNodeDist,
                                        useGraphCoorPred=self.useGraphCoorPred,

                                        useTrajAutoRegNode=self.useTrajAutoRegNode,
                                        useTrajAutoRegCoor=self.useTrajAutoRegCoor,

                                        useTrajEmbDist=self.useTrajEmbDist,

                                        feature_size=config.feature_size,
                                        embedding_size=config.embedding_size,
                                        num_layers=config.num_layers,
                                        dropout_rate=config.dropout_rate,
                                        device=config.device,
                                        usePI=config.gtraj["usePI"],
                                        useSI=config.gtraj["useSI"],
                                        useTransPI=config.gtraj["useTransPI"],
                                        useTransGI=config.gtraj["useTransGI"],
                                        dataset=config.dataset, alpha1=config.alpha1, alpha2=config.alpha2,
                                        num_nodes=self.num_nodes).to(self.device)
        if self.useGraphNodeDist:
            self.graph_node_dist_loss_fun = GraphSimLossFun(device=self.device)
        if self.useGraphCoorPred:
            self.graph_coor_pred_loss_fun = nn.MSELoss()
        if self.useTrajEmbDist:
            self.traj_emb_dist_loss_fun = TrajSimLossFun(device=self.device)

        if self.useTrajAutoRegNode:
            logging.info("Using TrajAutoRegNode")
            self.traj_auto_reg_node_loss_fun = nn.CrossEntropyLoss(ignore_index=self.num_nodes)
        if self.useTrajAutoRegCoor:
            logging.info("Using TrajAutoRegCoor")
            self.traj_auto_reg_coor_loss_fun = nn.MSELoss()

    def Spa_train(self):

        optimizer = torch.optim.AdamW([p for p in self.net.parameters() if p.requires_grad], lr=self.learning_rate,
                                      weight_decay=0.0001)

        milestones_list = [150]
        logging.info(milestones_list)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_list, gamma=0.2)

        # %===============
        distance_to_anchor_node = load_region_node(self.dataset).to(self.device)

        if self.useSI:
            road_network, node_coor = load_neighbor(self.dataset, self.num_knn)
            for item in road_network:
                item = item.to(self.device)
        else:
            road_network, node_coor = load_network(self.dataset)
            road_network = road_network.to(self.device)
        # %===============
        if self.useGraphNodeDist or self.useGraphCoorPred:
            graph_node_dist_train = graph_node_dist_loader(knn_neighbor_file=config.knn_neighbor_file,
                                                           knn_distance_file=config.knn_distance_file,
                                                           batch_size=self.pretrain_node_batch,
                                                           knn_neighbor_num=config.knn_neighbor_num,
                                                           knn_distance_coe=config.knn_distance_coe)
            logging.info(f'GraphNodeDist train data size: {len(graph_node_dist_train)}')
        if self.useGraphCoorPred:
            node = pd.read_csv(config.node_file)  # [node_id,lng,lat]
            node = node.sort_values('node').reset_index(drop=True)
            lng = node['lng'].values
            lat = node['lat'].values
            if self.dataset == 'beijing':
                abnormal_nodes = [102012, 106747]
                replacement_nodes = [0, 1]
                for i, abnormal_node in enumerate(abnormal_nodes):
                    if abnormal_node in node['node'].values:
                        abnormal_idx = node[node['node'] == abnormal_node].index[0]
                        replacement_idx = node[node['node'] == replacement_nodes[i]].index[0]
                        lng[abnormal_idx] = lng[replacement_idx]
                        lat[abnormal_idx] = lat[replacement_idx]

            meanx, meany, stdx, stdy = node_coor['mean_lng'], node_coor['mean_lat'], node_coor['std_lng'], node_coor['std_lat']
            graph_coor_label = np.stack([(lng - meanx) / stdx, (lat - meany) / stdy], axis=1)
            graph_coor_label = torch.tensor(graph_coor_label, dtype=torch.float32)
        if self.useTrajEmbDist or self.useTrajAutoRegNode or self.useTrajAutoRegCoor:
            traj_emb_dist_train = traj_emb_dist_loader(node_list_file=config.shuffle_node_file,
                                                       coor_list_file=config.shuffle_coor_file,
                                                       kseg_list_file=config.shuffle_kseg_file,
                                                       batch_size=self.pretrain_traj_batch,
                                                       num_nodes=self.num_nodes,
                                                       node_coor=node_coor)
            logging.info(f'TrajEmbDist train data size: {len(traj_emb_dist_train)}')

        # %==============================================================

        for epoch in range(0, self.epochs):

            graph_node_dist_losses = []
            graph_coor_pred_losses = []

            traj_emb_dist_losses = []
            traj_auto_reg_node_losses = []
            traj_auto_reg_coor_losses = []

            losses = []
            start_train = time.time()

            graph_node_dist_weights = []
            graph_coor_pred_weights = []
            traj_emb_dist_weights = []
            traj_auto_reg_weights = []
            traj_auto_reg_node_weights = []
            traj_auto_reg_coor_weights = []
            use_graph = self.useGraphNodeDist or self.useGraphCoorPred
            use_traj = self.useTrajEmbDist or self.useTrajAutoRegNode or self.useTrajAutoRegCoor
            logging.info(f"GraphNodeDist:{self.useGraphNodeDist}, "
                         f"GraphCoorPred:{self.useGraphCoorPred}, "
                         f"useTrajEmbDist:{self.useTrajEmbDist}, "
                         f"useTrajAutoReg:{self.useTrajAutoRegNode},{self.useTrajAutoRegCoor}")
            # %====================
            if use_graph and not use_traj:
                logging.info(f"use Graph {use_graph} training")
                self.net.train()
                for g_node_dist_batch in tqdm(graph_node_dist_train, total=len(graph_node_dist_train)):
                    # %====
                    (graph_node_aco, graph_node_pos, graph_node_neg,
                        node_pos_dis, node_neg_dis,
                        graph_coor_idx) = g_node_dist_batch
                    # %====
                    (graph_node_dis_emb,
                     graph_coor_pred, _, _) = self.net(network_data=[road_network, distance_to_anchor_node],
                                                       graph_node_idx=[graph_node_aco, graph_node_pos, graph_node_neg],
                                                       graph_coor_idx=graph_coor_idx)
                    # %=====graph_node_dist_loss=====
                    if self.useGraphNodeDist:
                        graph_node_dist_loss = self.graph_node_dist_loss_fun(graph_node_dis_emb, node_pos_dis, node_neg_dis)
                        graph_node_dist_losses.append(graph_node_dist_loss.item())
                    # %=====graph_coor_pred_loss=====
                    if self.useGraphCoorPred:
                        out_graph_coor_label = graph_coor_label[graph_coor_idx].to(self.device)
                        graph_coor_pred_loss = self.graph_coor_pred_loss_fun(graph_coor_pred, out_graph_coor_label)
                        graph_coor_pred_losses.append(graph_coor_pred_loss.item())
                    # %=====

                    optimizer.zero_grad()

                    if self.useGraphNodeDist and self.useGraphCoorPred:
                        weights = torch.stack([
                            torch.exp(self.net.graph_node_dist_weight),
                            torch.exp(self.net.graph_coor_pred_weight),
                        ])
                        weights = 2 * weights / weights.sum()
                        graph_node_dist_weight, graph_coor_pred_weight = weights
                        loss = (graph_node_dist_weight * graph_node_dist_loss +
                                graph_coor_pred_weight * graph_coor_pred_loss)

                        graph_node_dist_weights.append(graph_node_dist_weight.item())
                        graph_coor_pred_weights.append(graph_coor_pred_weight.item())
                    elif self.useGraphNodeDist and not self.useGraphCoorPred:
                        loss = graph_node_dist_loss
                    elif not self.useGraphNodeDist and self.useGraphCoorPred:
                        loss = graph_coor_pred_loss

                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
            elif not use_graph and use_traj:
                logging.info(f"use Traj {use_traj} training")
                self.net.train()
                for t_emb_dist_batch in tqdm(traj_emb_dist_train, total=len(traj_emb_dist_train)):

                    # %====
                    (traj_node_aco, traj_coor_aco,
                        traj_node_pos, traj_coor_pos, traj_dist_pos,
                        traj_node_neg, traj_coor_neg, traj_dist_neg,
                        aco_length, pos_length, neg_length,
                        traj_node_tgt, traj_coor_tgt, traj_node_lbl, traj_coor_lbl) = t_emb_dist_batch
                    # %====
                    (_, _,
                     out_traj_emb_dist,
                     out_traj_auto_reg_pred
                     ) = self.net(network_data=[road_network, distance_to_anchor_node],
                                  traj_aco=[traj_node_aco, traj_coor_aco, aco_length],
                                  traj_pos=[traj_node_pos, traj_coor_pos, pos_length],
                                  traj_neg=[traj_node_neg, traj_coor_neg, neg_length],
                                  traj_node_tgt=traj_node_tgt,
                                  traj_coor_tgt=traj_coor_tgt,)
                    # %=====traj_emb_dist_loss=====
                    if self.useTrajEmbDist:
                        traj_emb_dist_loss = self.traj_emb_dist_loss_fun(out_traj_emb_dist, traj_dist_pos, traj_dist_neg)
                        traj_emb_dist_losses.append(traj_emb_dist_loss.item())
                    # %=====traj_auto_reg_node_loss=====
                    if self.useTrajAutoRegNode and self.useTrajAutoRegCoor:
                        traj_auto_reg_node_pred, traj_auto_reg_coor_pred, tgt_padding_mask = out_traj_auto_reg_pred

                        traj_node_lbl = traj_node_lbl.to(self.device)
                        traj_node_lbl_idx = traj_node_lbl[~tgt_padding_mask].to(self.device)
                        traj_auto_reg_node_loss = self.traj_auto_reg_node_loss_fun(traj_auto_reg_node_pred, traj_node_lbl_idx)
                        traj_auto_reg_node_losses.append(traj_auto_reg_node_loss.item())
                        # %=====traj_auto_reg_coor_loss=====

                        traj_coor_lbl = traj_coor_lbl.to(self.device)
                        traj_coor_lbl_idx = traj_coor_lbl[~tgt_padding_mask].to(self.device)
                        traj_auto_reg_coor_loss = self.traj_auto_reg_coor_loss_fun(traj_auto_reg_coor_pred, traj_coor_lbl_idx)
                        traj_auto_reg_coor_losses.append(traj_auto_reg_coor_loss.item())
                    # %=====
                    optimizer.zero_grad()
                    # =====
                    if self.useTrajEmbDist and (self.useTrajAutoRegNode and self.useTrajAutoRegCoor):
                        weights = torch.stack([
                            torch.exp(self.net.traj_emb_dist_weight),
                            torch.exp(self.net.traj_auto_reg_weight),
                        ])
                        weights = 2 * weights / weights.sum()
                        traj_emb_dist_weight, traj_auto_reg_weight = weights

                        subweights = torch.stack([
                            torch.exp(self.net.traj_auto_reg_node_weight),
                            torch.exp(self.net.traj_auto_reg_coor_weight)
                        ])
                        subweights = 2 * subweights / subweights.sum()
                        traj_auto_reg_node_weight, traj_auto_reg_coor_weight = subweights

                        loss = (traj_emb_dist_weight * traj_emb_dist_loss +
                                traj_auto_reg_weight * (traj_auto_reg_node_weight * traj_auto_reg_node_loss +
                                                        traj_auto_reg_coor_weight * traj_auto_reg_coor_loss))
                        traj_emb_dist_weights.append(traj_emb_dist_weight.item())
                        traj_auto_reg_weights.append(traj_auto_reg_weight.item())
                        traj_auto_reg_node_weights.append(traj_auto_reg_node_weight.item())
                        traj_auto_reg_coor_weights.append(traj_auto_reg_coor_weight.item())
                    elif self.useTrajEmbDist and not (self.useTrajAutoRegNode and self.useTrajAutoRegCoor):
                        loss = traj_emb_dist_loss
                    elif not self.useTrajEmbDist and (self.useTrajAutoRegNode and self.useTrajAutoRegCoor):
                        subweights = torch.stack([
                            torch.exp(self.net.traj_auto_reg_node_weight),
                            torch.exp(self.net.traj_auto_reg_coor_weight)
                        ])
                        subweights = 2 * subweights / subweights.sum()
                        traj_auto_reg_node_weight, traj_auto_reg_coor_weight = subweights

                        loss = (traj_auto_reg_node_weight * traj_auto_reg_node_loss +
                                traj_auto_reg_coor_weight * traj_auto_reg_coor_loss)
                        traj_auto_reg_node_weights.append(traj_auto_reg_node_weight.item())
                        traj_auto_reg_coor_weights.append(traj_auto_reg_coor_weight.item())
                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
            elif use_graph and use_traj:
                logging.info(f"use Graph {use_graph}, use Traj {use_traj} training")
                self.net.train()
                for g_node_dist_batch, t_emb_dist_batch in tqdm(zip(cycle(graph_node_dist_train), traj_emb_dist_train), total=len(traj_emb_dist_train)):

                    # %====
                    (graph_node_aco, graph_node_pos, graph_node_neg,
                        node_pos_dis, node_neg_dis,
                        graph_coor_idx) = g_node_dist_batch

                    (traj_node_aco, traj_coor_aco,
                        traj_node_pos, traj_coor_pos, traj_dist_pos,
                        traj_node_neg, traj_coor_neg, traj_dist_neg,
                        aco_length, pos_length, neg_length,
                        traj_node_tgt, traj_coor_tgt, traj_node_lbl, traj_coor_lbl) = t_emb_dist_batch
                    # %====
                    (graph_node_dis_emb, graph_coor_pred,
                        out_traj_emb_dist, out_traj_auto_reg_pred
                     ) = self.net(network_data=[road_network, distance_to_anchor_node],
                                  graph_node_idx=[graph_node_aco, graph_node_pos, graph_node_neg],
                                  graph_coor_idx=graph_coor_idx,
                                  traj_aco=[traj_node_aco, traj_coor_aco, aco_length],
                                  traj_pos=[traj_node_pos, traj_coor_pos, pos_length],
                                  traj_neg=[traj_node_neg, traj_coor_neg, neg_length],
                                  traj_node_tgt=traj_node_tgt,
                                  traj_coor_tgt=traj_coor_tgt,)
                    # %=====graph_node_dist_loss=====
                    if self.useGraphNodeDist:
                        graph_node_dist_loss = self.graph_node_dist_loss_fun(graph_node_dis_emb, node_pos_dis, node_neg_dis)
                        graph_node_dist_losses.append(graph_node_dist_loss.item())
                    # %=====graph_coor_pred_loss=====
                    if self.useGraphCoorPred:
                        out_graph_coor_label = graph_coor_label[graph_coor_idx].to(self.device)
                        graph_coor_pred_loss = self.graph_coor_pred_loss_fun(graph_coor_pred, out_graph_coor_label)
                        graph_coor_pred_losses.append(graph_coor_pred_loss.item())
                    # %=====traj_emb_dist_loss=====
                    if self.useTrajEmbDist:
                        traj_emb_dist_loss = self.traj_emb_dist_loss_fun(out_traj_emb_dist, traj_dist_pos, traj_dist_neg)
                        traj_emb_dist_losses.append(traj_emb_dist_loss.item())
                    # %=====traj_auto_reg_node_loss=====
                    if self.useTrajAutoRegNode and self.useTrajAutoRegCoor:
                        traj_auto_reg_node_pred, traj_auto_reg_coor_pred, tgt_padding_mask = out_traj_auto_reg_pred

                        traj_node_lbl = traj_node_lbl.to(self.device)
                        traj_node_lbl_idx = traj_node_lbl[~tgt_padding_mask].to(self.device)
                        traj_auto_reg_node_loss = self.traj_auto_reg_node_loss_fun(traj_auto_reg_node_pred, traj_node_lbl_idx)
                        traj_auto_reg_node_losses.append(traj_auto_reg_node_loss.item())
                        # %=====traj_auto_reg_coor_loss=====

                        traj_coor_lbl = traj_coor_lbl.to(self.device)
                        traj_coor_lbl_idx = traj_coor_lbl[~tgt_padding_mask].to(self.device)
                        traj_auto_reg_coor_loss = self.traj_auto_reg_coor_loss_fun(traj_auto_reg_coor_pred, traj_coor_lbl_idx)
                        traj_auto_reg_coor_losses.append(traj_auto_reg_coor_loss.item())
                    # %=====
                    optimizer.zero_grad()
                    # =====
                    if self.useGraphNodeDist and self.useGraphCoorPred and self.useTrajEmbDist and (self.useTrajAutoRegNode and self.useTrajAutoRegCoor):
                        weights = torch.stack([
                            torch.exp(self.net.graph_node_dist_weight),
                            torch.exp(self.net.graph_coor_pred_weight),
                            torch.exp(self.net.traj_emb_dist_weight),
                            torch.exp(self.net.traj_auto_reg_weight)
                        ])
                        weights = 4 * weights / weights.sum()
                        graph_node_dist_weight, graph_coor_pred_weight, traj_emb_dist_weight, traj_auto_reg_weight = weights

                        subweights = torch.stack([
                            torch.exp(self.net.traj_auto_reg_node_weight),
                            torch.exp(self.net.traj_auto_reg_coor_weight)
                        ])
                        subweights = 2 * subweights / subweights.sum()
                        traj_auto_reg_node_weight, traj_auto_reg_coor_weight = subweights

                        # =====
                        loss = (graph_node_dist_weight * graph_node_dist_loss +
                                graph_coor_pred_weight * graph_coor_pred_loss +
                                traj_emb_dist_weight * traj_emb_dist_loss +
                                traj_auto_reg_weight * (traj_auto_reg_node_weight * traj_auto_reg_node_loss +
                                                        traj_auto_reg_coor_weight * traj_auto_reg_coor_loss))

                        graph_node_dist_weights.append(graph_node_dist_weight.item())
                        graph_coor_pred_weights.append(graph_coor_pred_weight.item())
                        traj_emb_dist_weights.append(traj_emb_dist_weight.item())
                        traj_auto_reg_weights.append(traj_auto_reg_weight.item())
                        traj_auto_reg_node_weights.append(traj_auto_reg_node_weight.item())
                        traj_auto_reg_coor_weights.append(traj_auto_reg_coor_weight.item())
                    elif self.useGraphNodeDist and self.useGraphCoorPred and self.useTrajEmbDist and not (self.useTrajAutoRegNode and self.useTrajAutoRegCoor):
                        weights = torch.stack([
                            torch.exp(self.net.graph_node_dist_weight),
                            torch.exp(self.net.graph_coor_pred_weight),
                            torch.exp(self.net.traj_emb_dist_weight),
                        ])
                        weights = 3 * weights / weights.sum()
                        graph_node_dist_weight, graph_coor_pred_weight, traj_emb_dist_weight = weights

                        loss = (graph_node_dist_weight * graph_node_dist_loss +
                                graph_coor_pred_weight * graph_coor_pred_loss +
                                traj_emb_dist_weight * traj_emb_dist_loss)
                        graph_node_dist_weights.append(graph_node_dist_weight.item())
                        graph_coor_pred_weights.append(graph_coor_pred_weight.item())
                        traj_emb_dist_weights.append(traj_emb_dist_weight.item())
                    elif self.useGraphNodeDist and self.useGraphCoorPred and not self.useTrajEmbDist and (self.useTrajAutoRegNode and self.useTrajAutoRegCoor):
                        weights = torch.stack([
                            torch.exp(self.net.graph_node_dist_weight),
                            torch.exp(self.net.graph_coor_pred_weight),
                            torch.exp(self.net.traj_auto_reg_weight)
                        ])
                        weights = 3 * weights / weights.sum()
                        graph_node_dist_weight, graph_coor_pred_weight, traj_auto_reg_weight = weights

                        subweights = torch.stack([
                            torch.exp(self.net.traj_auto_reg_node_weight),
                            torch.exp(self.net.traj_auto_reg_coor_weight)
                        ])
                        subweights = 2 * subweights / subweights.sum()
                        traj_auto_reg_node_weight, traj_auto_reg_coor_weight = subweights

                        loss = (graph_node_dist_weight * graph_node_dist_loss +
                                graph_coor_pred_weight * graph_coor_pred_loss +
                                traj_auto_reg_weight * (traj_auto_reg_node_weight * traj_auto_reg_node_loss +
                                                        traj_auto_reg_coor_weight * traj_auto_reg_coor_loss))
                        graph_node_dist_weights.append(graph_node_dist_weight.item())
                        graph_coor_pred_weights.append(graph_coor_pred_weight.item())
                        traj_auto_reg_weights.append(traj_auto_reg_weight.item())
                        traj_auto_reg_node_weights.append(traj_auto_reg_node_weight.item())
                        traj_auto_reg_coor_weights.append(traj_auto_reg_coor_weight.item())
                    elif not self.useGraphNodeDist and self.useGraphCoorPred and self.useTrajEmbDist and (self.useTrajAutoRegNode and self.useTrajAutoRegCoor):
                        weights = torch.stack([
                            torch.exp(self.net.graph_coor_pred_weight),
                            torch.exp(self.net.traj_emb_dist_weight),
                            torch.exp(self.net.traj_auto_reg_weight)
                        ])
                        weights = 3 * weights / weights.sum()
                        graph_coor_pred_weight, traj_emb_dist_weight, traj_auto_reg_weight = weights

                        subweights = torch.stack([
                            torch.exp(self.net.traj_auto_reg_node_weight),
                            torch.exp(self.net.traj_auto_reg_coor_weight)
                        ])
                        subweights = 2 * subweights / subweights.sum()
                        traj_auto_reg_node_weight, traj_auto_reg_coor_weight = subweights

                        loss = (graph_coor_pred_weight * graph_coor_pred_loss +
                                traj_emb_dist_weight * traj_emb_dist_loss +
                                traj_auto_reg_weight * (traj_auto_reg_node_weight * traj_auto_reg_node_loss +
                                                        traj_auto_reg_coor_weight * traj_auto_reg_coor_loss))
                        graph_coor_pred_weights.append(graph_coor_pred_weight.item())
                        traj_emb_dist_weights.append(traj_emb_dist_weight.item())
                        traj_auto_reg_weights.append(traj_auto_reg_weight.item())
                        traj_auto_reg_node_weights.append(traj_auto_reg_node_weight.item())
                        traj_auto_reg_coor_weights.append(traj_auto_reg_coor_weight.item())
                    elif self.useGraphNodeDist and not self.useGraphCoorPred and self.useTrajEmbDist and (self.useTrajAutoRegNode and self.useTrajAutoRegCoor):
                        weights = torch.stack([
                            torch.exp(self.net.graph_node_dist_weight),
                            torch.exp(self.net.traj_emb_dist_weight),
                            torch.exp(self.net.traj_auto_reg_weight)
                        ])
                        weights = 3 * weights / weights.sum()
                        graph_node_dist_weight, traj_emb_dist_weight, traj_auto_reg_weight = weights

                        subweights = torch.stack([
                            torch.exp(self.net.traj_auto_reg_node_weight),
                            torch.exp(self.net.traj_auto_reg_coor_weight)
                        ])
                        subweights = 2 * subweights / subweights.sum()
                        traj_auto_reg_node_weight, traj_auto_reg_coor_weight = subweights

                        loss = (graph_node_dist_weight * graph_node_dist_loss +
                                traj_emb_dist_weight * traj_emb_dist_loss +
                                traj_auto_reg_weight * (traj_auto_reg_node_weight * traj_auto_reg_node_loss +
                                                        traj_auto_reg_coor_weight * traj_auto_reg_coor_loss))
                        graph_node_dist_weights.append(graph_node_dist_weight.item())
                        traj_emb_dist_weights.append(traj_emb_dist_weight.item())
                        traj_auto_reg_weights.append(traj_auto_reg_weight.item())
                        traj_auto_reg_node_weights.append(traj_auto_reg_node_weight.item())
                        traj_auto_reg_coor_weights.append(traj_auto_reg_coor_weight.item())

                    losses.append(loss.item())
                    loss.backward()
                    optimizer.step()
            # %====================

            end_train = time.time()
            scheduler.step()
            logging.info(f'Used learning rate:{scheduler.get_last_lr()[0]}')
            logging.info(f'Epoch [{epoch}/{self.epochs}]: Training time is {end_train - start_train}')
            logging.info(f'GraphNodeDist Loss: {np.mean(graph_node_dist_losses) if graph_node_dist_losses else "N/A"}')
            logging.info(f'GraphCoorPred Loss: {np.mean(graph_coor_pred_losses) if graph_coor_pred_losses else "N/A"}')
            logging.info(f'TrajEmbDist Loss: {np.mean(traj_emb_dist_losses) if traj_emb_dist_losses else "N/A"}')
            logging.info(f'TrajAutoRegNode Loss: {np.mean(traj_auto_reg_node_losses) if traj_auto_reg_node_losses else "N/A"}')
            logging.info(f'TrajAutoRegCoor Loss: {np.mean(traj_auto_reg_coor_losses) if traj_auto_reg_coor_losses else "N/A"}')
            logging.info(f'Total Loss: {np.mean(losses)}')

            logging.info(f'GraphNodeDist Weight: {np.mean(graph_node_dist_weights) if graph_node_dist_weights else "N/A"}')
            logging.info(f'GraphCoorPred Weight: {np.mean(graph_coor_pred_weights) if graph_coor_pred_weights else "N/A"}')
            logging.info(f'TrajEmbDist Weight: {np.mean(traj_emb_dist_weights) if traj_emb_dist_weights else "N/A"}')
            logging.info(f'TrajAutoReg Weight: {np.mean(traj_auto_reg_weights) if traj_auto_reg_weights else "N/A"}')
            logging.info(f'TrajAutoRegNode Weight: {np.mean(traj_auto_reg_node_weights) if traj_auto_reg_node_weights else "N/A"}')
            logging.info(f'TrajAutoRegCoor Weight: {np.mean(traj_auto_reg_coor_weights) if traj_auto_reg_coor_weights else "N/A"}')
            if epoch % 1 == 0:

                model_save_path = osp.join(self.save_folder, f"GraphNodeSimEncoder_{self.dataset}_epochs{epoch}.pth")
                torch.save({
                    'graph_encoder_state_dict': self.net.graph_encoder.state_dict() if self.net.graph_encoder is not None else None,
                    'node_encoder_state_dict': self.net.node_encoder.state_dict() if self.net.node_encoder is not None else None,
                    'coor_encoder_state_dict': self.net.coor_encoder.state_dict() if self.net.coor_encoder is not None else None,
                    'node_pooling_state_dict': self.net.node_pooling.state_dict() if self.net.node_pooling is not None else None,
                    'epoch': self.epochs,
                    'loss': np.mean(losses),
                    'pretrain': config.pretrain
                }, model_save_path)
                logging.info(f'Model saved to: {model_save_path}')
# # %=======================================================================================


class GTrajSim_Trainer(object):

    def __init__(self, save_folder=None, pretrain_folder=None):
        self.save_folder = save_folder if save_folder else config.save_folder
        self.feature_size = config.feature_size
        self.embedding_size = config.embedding_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate
        self.concat = config.concat
        self.device = config.device
        self.learning_rate = config.train_learning_rate
        self.epochs = config.train_epochs

        self.train_batch = config.train_batch
        self.test_batch = config.test_batch
        self.usePI = config.gtraj["usePI"]
        self.useSI = config.gtraj["useSI"]
        self.useTransPI = config.gtraj["useTransPI"]
        self.useTransGI = config.gtraj["useTransGI"]

        self.usePreGraphEncoder = config.use_pretrain_model["usePreGraphEncoder"]
        self.usePreTrajEncoder = config.use_pretrain_model["usePreTrajEncoder"]

        self.dataset = str(config.dataset)
        self.distance_type = str(config.distance_type)
        self.early_stop = config.early_stop
        self.alpha1 = config.alpha1
        self.alpha2 = config.alpha2
        self.num_knn = config.num_knn
        self.config = config
        self.use_AdamW = False
        graph_encoder, node_encoder, node_pooling, coor_encoder, = self.load_trained_model(model_path=pretrain_folder)
        self.net = GraphTrajSimEncoder(graph_encoder=graph_encoder,
                                       node_encoder=node_encoder,
                                       node_pooling=node_pooling,
                                       coor_encoder=coor_encoder,

                                       feature_size=self.feature_size,
                                       embedding_size=self.embedding_size,
                                       num_layers=self.num_layers,
                                       dropout_rate=self.dropout_rate,
                                       concat=self.concat,
                                       device=self.device,
                                       usePI=self.usePI,
                                       useSI=self.useSI,
                                       useTransPI=self.useTransPI,
                                       useTransGI=self.useTransGI,
                                       dataset=self.dataset,
                                       alpha1=self.alpha1,
                                       alpha2=self.alpha2).to(self.device)

    def Spa_eval(self, load_model=None, is_long_traj=False):
        if load_model != None:
            logging.info(load_model)
            self.net.load_state_dict(torch.load(load_model, map_location=self.device))
            self.net.to(self.device)
        else:
            logging.info('No pre-trained model is loaded!')

        if self.useSI:
            road_network, node_coor = load_neighbor(self.dataset, self.num_knn)
            for item in road_network:
                item = item.to(self.device)
        else:
            road_network, node_coor = load_network(self.dataset)
            road_network = road_network.to(self.device)


        distance_to_anchor_node = load_region_node(self.dataset).to(self.device)

        if is_long_traj:
            test_data_loader, test_lenth = spatial_data_utils.vali_data_loader(
                config.spatial_long_test_set, self.test_batch, node_coor)
        else:
            test_data_loader, test_lenth = spatial_data_utils.vali_data_loader(
                config.spatial_test_set, self.test_batch, node_coor)

        logging.info(f'test size:{test_lenth}')

        self.net.eval()
        with torch.no_grad():
            start_test_epoch = time.time()
            if self.useTransGI:
                test_embedding = torch.zeros((test_lenth, self.embedding_size*2),
                                                device=self.device, requires_grad=False)
            else:
                test_embedding = torch.zeros((test_lenth, self.embedding_size),
                                                device=self.device, requires_grad=False)
            test_label = torch.zeros((test_lenth, 50), requires_grad=False, dtype=torch.long)

            for batch in tqdm(test_data_loader):
                (data, coor, label, data_length, idx) = batch
                a_embedding = self.net([road_network, distance_to_anchor_node], data, coor, data_length)
                test_embedding[idx] = a_embedding
                test_label[idx] = label
            end_test_epoch = time.time()
            acc = test_method.test_spa_model(test_embedding, test_label, self.device)
            logging.info('Dataset: {}, Distance type: {}, f_num is {}'.format(
                self.dataset, self.distance_type, test_lenth))
            end_test = time.time()
            logging.info(f"epoch test time: {end_test_epoch - start_test_epoch}")
            logging.info(f"all test time: {end_test - start_test_epoch}")
            logging.info('HR-10 HR-50 R10@50 R1@1 R1@10 R1@50')
            logging.info(' & '.join([f"{x.item():.4f}" for x in acc]))

    def Spa_train(self, load_model_path=None):
        if load_model_path != None:
            logging.info(load_model_path)
            self.net.load_state_dict(torch.load(load_model_path, map_location=self.device))
            self.net.to(self.device)


        optimizer = torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad], lr=self.learning_rate,
                                    weight_decay=0.0001)
        milestones_list = [20, 40]
        logging.info(milestones_list)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_list, gamma=0.2)

        lossfunction = SpaLossFun(self.train_batch, self.distance_type).to(self.device)
        distance_to_anchor_node = load_region_node(self.dataset).to(self.device)

        if self.useSI:
            road_network, node_coor = load_neighbor(self.dataset, self.num_knn)
            for item in road_network:
                item = item.to(self.device)
        else:
            road_network, node_coor = load_network(self.dataset)
            road_network = road_network.to(self.device)

        train_loader = train_data_loader(config.spatial_train_set, self.train_batch,  node_coor)
        logging.info(f'train batch number:{len(train_loader)}')
        vali_loader,  vali_lenth = vali_data_loader(config.spatial_vali_set, self.test_batch, node_coor)
        logging.info(f'vali batch number:{len(vali_loader)}')
        best_epoch = 0
        best_hr10 = 0
        for epoch in range(0, self.epochs):
            self.net.train()
            losses = []
            start_train = time.time()
            for batch in tqdm(train_loader):
                optimizer.zero_grad()
                (node_aco, coor_aco, node_pos, coor_pos, node_neg, coor_neg,
                pos_dis, neg_dis,
                aco_length, pos_length, neg_length) = batch
                a_embedding = self.net([road_network, distance_to_anchor_node], node_aco, coor_aco, aco_length)
                p_embedding = self.net([road_network, distance_to_anchor_node], node_pos, coor_pos, pos_length)
                n_embedding = self.net([road_network, distance_to_anchor_node], node_neg, coor_neg, neg_length)

                loss = lossfunction(a_embedding, p_embedding, n_embedding, pos_dis, neg_dis, self.device)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            end_train = time.time()
            scheduler.step()
            logging.info(f'Used learning rate:{scheduler.get_last_lr()[0]}')
            logging.info('Epoch [{}/{}]: Training time is {}, loss is {}'.format(
                epoch, self.epochs, (end_train - start_train), np.mean(losses)))

            if epoch % 1 == 0:
                self.net.eval()
                with torch.no_grad():
                    start_vali = time.time()
                    if self.useTransGI:
                        vali_embedding = torch.zeros((vali_lenth, self.embedding_size*2),
                                                    device=self.device, requires_grad=False)
                    else:
                        vali_embedding = torch.zeros((vali_lenth, self.embedding_size),
                                                    device=self.device, requires_grad=False)
                    vali_label = torch.zeros((vali_lenth, 50), requires_grad=False, dtype=torch.long)

                    for batch in tqdm(vali_loader):
                        (node_aco, coor_aco, label, aco_length, idx) = batch
                        a_embedding = self.net([road_network, distance_to_anchor_node], node_aco, coor_aco, aco_length)
                        vali_embedding[idx] = a_embedding
                        vali_label[idx] = label

                    acc = test_method.test_spa_model(vali_embedding, vali_label, self.device)

                    logging.info('Dataset: {}, Distance type: {}, f_num is {}'.format(
                        self.dataset, self.distance_type, vali_lenth))
                    end_vali = time.time()
                    logging.info(f"vali time: {end_vali - start_vali}")
                    # logging.info(acc)
                    logging.info(" ")
                    logging.info(' & '.join([f"{x.item():.4f}" for x in acc]))

                    if acc[0] > best_hr10 or epoch % 5 == 0 or epoch == 1:
                        if acc[0] > best_hr10:
                            best_hr10 = acc[0]
                            best_epoch = epoch
                            logging.info(f'best epoch: {best_epoch}')

                        save_modelname = self.save_folder + "/epoch_%d.pt" % epoch
                        if not os.path.exists(self.save_folder):
                            os.makedirs(self.save_folder)
                        torch.save(self.net.state_dict(), save_modelname)
                        logging.info(save_modelname)

                    if epoch - best_epoch >= self.early_stop:
                        logging.info(f"Early stopping at epoch {epoch}")
                        logging.info(f"Best HR-10: {best_hr10} at epoch {best_epoch}")
                        break

    def load_trained_model(self, model_path=None):

        graph_encoder = GraphEncoder(feature_size=self.feature_size,
                                     embedding_size=self.embedding_size,
                                     usePI=self.usePI,
                                     useSI=self.useSI,
                                     dataset=self.dataset,
                                     alpha1=self.alpha1,
                                     alpha2=self.alpha2).to(self.device)
        node_encoder = TrajNodeEncoder(embedding_size=self.embedding_size,
                                       nhead=4, d_hid=512,
                                       num_layers=self.num_layers,
                                       dropout_rate=self.dropout_rate, device=self.device)
        node_pooling = AttnPooling(self.embedding_size)
        coor_encoder = TrajCoorEncoder(embedding_size=self.embedding_size, device=self.device)

        if model_path is not None:
            logging.info("=" * 30 + "Loading GraphNodeSimEncoder Model" + "=" * 30)
            checkpoint = torch.load(model_path, map_location=self.device)
            logging.info(f"model path: {model_path}")
            logging.info(checkpoint['pretrain'])

            if self.usePreGraphEncoder and checkpoint['graph_encoder_state_dict'] is not None:
                logging.info("Using GraphSim pretraining")
                graph_encoder.load_state_dict(checkpoint['graph_encoder_state_dict'])
            else:
                logging.info("Not using pretraining for GraphEncoder")

            if self.usePreTrajEncoder:
                if checkpoint['node_encoder_state_dict'] is not None and checkpoint['node_pooling_state_dict'] is not None:
                    logging.info("Using Node pretraining")
                    node_encoder.load_state_dict(checkpoint['node_encoder_state_dict'])
                    node_pooling.load_state_dict(checkpoint['node_pooling_state_dict'])
                else:
                    logging.info("Node pretraining not found or not used")
                if checkpoint['coor_encoder_state_dict'] is not None:
                    logging.info("Using Coor pretraining")
                    coor_encoder.load_state_dict(checkpoint['coor_encoder_state_dict'])
                else:
                    logging.info("Coor pretraining not found or not used")
            else:
                logging.info("Not using pretraining for TrajEncoder")
            logging.info("=" * 60)
            return graph_encoder, node_encoder, node_pooling, coor_encoder

        elif model_path is None or not os.path.exists(model_path):
            logging.info("Creating a new GraphNodeSimEncoder model")
            logging.info("=" * 60)
            return graph_encoder, node_encoder, node_pooling, coor_encoder
