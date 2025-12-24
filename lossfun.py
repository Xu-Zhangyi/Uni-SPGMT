import logging
import torch
from torch.nn import Module
from setting import SetParameter
import torch.nn.functional as F
import torch.nn as nn
config = SetParameter()


class GraphSimLossFun(Module):
    def __init__(self, device):
        super(GraphSimLossFun, self).__init__()
        self.device = device

    def forward(self, graph_node_dis_emb, node_pos_dis, node_neg_dis):

        a_embedding = graph_node_dis_emb[0]
        p_embedding = graph_node_dis_emb[1]
        n_embedding = graph_node_dis_emb[2]

        D_ap = torch.exp(-node_pos_dis).to(self.device)
        D_an = torch.exp(-node_neg_dis).to(self.device)

        v_ap = torch.exp(-(torch.norm(a_embedding-p_embedding, p=2, dim=-1)))
        v_an = torch.exp(-(torch.norm(a_embedding-n_embedding, p=2, dim=-1)))

        loss_entire_ap = (D_ap - v_ap) ** 2
        loss_entire_an = (D_an - v_an) ** 2
        loss = loss_entire_ap + loss_entire_an + (F.relu(v_an - v_ap)) ** 2
        loss_mean = loss.mean(dim=-1)

        return loss_mean


class TrajSimLossFun(Module):
    def __init__(self, device):
        super(TrajSimLossFun, self).__init__()
        self.device = device

    def forward(self, traj_embedding, dist_pos, dist_neg):
        (a_embedding, p_embedding, n_embedding) = traj_embedding
        D_ap = torch.exp(-dist_pos).to(self.device)  # -700
        D_an = torch.exp(-dist_neg).to(self.device)
        v_ap = torch.exp(-(torch.norm(a_embedding-p_embedding, p=2, dim=-1)))
        v_an = torch.exp(-(torch.norm(a_embedding-n_embedding, p=2, dim=-1)))

        loss_entire_ap = (D_ap - v_ap) ** 2
        loss_entire_an = (D_an - v_an) ** 2
        loss = loss_entire_ap + loss_entire_an + (D_ap > D_an)*(F.relu(v_an - v_ap)) ** 2

        loss_mean = loss.mean(dim=-1)

        return loss_mean


class SpaLossFun(Module):
    def __init__(self, train_batch, distance_type):
        super(SpaLossFun, self).__init__()
        self.train_batch = train_batch
        self.distance_type = distance_type
        self.flag = True

    def forward(self, embedding_a, embedding_p, embedding_n, pos_dis, neg_dis, device):

        D_ap = torch.exp(-pos_dis).to(device)  # -700
        D_an = torch.exp(-neg_dis).to(device)

        v_ap = torch.exp(-(torch.norm(embedding_a-embedding_p, p=2, dim=-1)))
        v_an = torch.exp(-(torch.norm(embedding_a-embedding_n, p=2, dim=-1)))
        loss_entire_ap = (D_ap - v_ap) ** 2
        loss_entire_an = (D_an - v_an) ** 2
        loss = loss_entire_ap + loss_entire_an + (D_ap > D_an)*(F.relu(v_an - v_ap)) ** 2
        loss_mean = loss.mean(dim=-1)
        return loss_mean
