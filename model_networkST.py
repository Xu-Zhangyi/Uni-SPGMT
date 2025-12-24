import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
from setting import SetParameter
config = SetParameter()


class SMNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size,  device=0):
        super(SMNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.nonLeaky = torch.nn.LeakyReLU(0.1)
        self.nonTanh = torch.nn.Tanh()
        self.point_pooling = torch.nn.AvgPool1d(10)

        self.seq_model_layer = 1
        self.device = device
        self.t2s_model = torch.nn.LSTM(self.input_size, hidden_size, num_layers=self.seq_model_layer)

        self.res_linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.res_linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.res_linear3 = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs_a):
        input_a, input_len_a = inputs_a  # porto inputs:220x149x4 inputs_len:list

        outputs_a, (hn_a, cn_a) = self.t2s_model(input_a.permute(1, 0, 2))
        outputs_ca = F.sigmoid(self.res_linear1(outputs_a)) * F.tanh(self.res_linear2(outputs_a))
        outputs_hata = F.sigmoid(self.res_linear3(outputs_a)) * F.tanh(outputs_ca)
        outputs_fa = outputs_a + outputs_hata
        mask_out_a = []
        for b, v in enumerate(input_len_a):
            mask_out_a.append(outputs_fa[v - 1][b, :].view(1, -1))
        fa_outputs = torch.cat(mask_out_a, dim=0)
        return fa_outputs, outputs_fa

# %===========================================================================================================


class PosKnnGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, usePI, useSI, dataset):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.usePI = usePI
        self.useSI = useSI
        logging.info('usePI: ' + str(self.usePI) + ', useSI: ' + str(self.useSI))
        self.dataset = dataset

        if dataset == 'beijing':
            self.nodeLin = torch.nn.Linear(in_channels + 98, in_channels, bias=False)
        elif dataset == 'tdrive':
            self.nodeLin = torch.nn.Linear(in_channels + 112, in_channels, bias=False)
        elif dataset == 'porto':
            self.nodeLin = torch.nn.Linear(in_channels + 68, in_channels, bias=False)
        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=False)  # [64,64]
        self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.nodeLin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        with torch.no_grad():
            self.bias.zero_()

        # self.bias.data.zero_()

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        if self.useSI:
            return norm[0].view(-1, 1) * (self.lin1(x_j)) + norm[1].view(-1, 1) * (self.lin2(x_j))
        else:
            return norm.view(-1, 1) * (self.lin1(x_j))

    def forward(self, x, input_edge_index, input_edge_attr, d2an, firstLayer):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_attr = add_self_loops(input_edge_index, input_edge_attr, num_nodes=x.size(0), fill_value=1.0)

        # Step 2: Linearly transform node feature matrix.

        if firstLayer and self.usePI:
            combined_input = torch.cat((x, d2an), dim=1)
            x = self.nodeLin(combined_input)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        edge_inv_sqrt = edge_attr.pow(-0.5)
        edge_inv_sqrt[edge_inv_sqrt == float('inf')] = 0
        edge_inv_sqrt[edge_inv_sqrt > 1.0] = 1.0
        edge_norm = edge_inv_sqrt

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=[deg_norm, edge_norm])

        return out

    def gcn_forward(self, x, input_edge_index, input_edge_attr, d2an, firstLayer):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_attr = add_self_loops(input_edge_index, input_edge_attr, num_nodes=x.size(0), fill_value=1.0)

        # Step 2: Linearly transform node feature matrix.
        if firstLayer and self.usePI:
            combined_input = torch.cat((x, d2an), dim=1)
            x = self.nodeLin(combined_input)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        edge_inv_sqrt = edge_attr.pow(-0.5)
        edge_inv_sqrt[edge_inv_sqrt == float('inf')] = 0
        edge_inv_sqrt[edge_inv_sqrt > 1.0] = 1.0
        edge_norm = edge_inv_sqrt

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=deg_norm)

        return out


class KnnGNN(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePI, useSI, dataset, alpha1, alpha2):
        super(KnnGNN, self).__init__()
        self.usePI = usePI
        self.useSI = useSI
        self.dataset = dataset
        self.posconv1 = PosKnnGNNLayer(encoding_size, embedding_size, self.usePI, self.useSI, self.dataset)
        self.posconv2 = PosKnnGNNLayer(encoding_size, embedding_size, self.usePI, self.useSI, self.dataset)
        self.posconv3 = PosKnnGNNLayer(embedding_size, embedding_size, self.usePI, self.useSI, self.dataset)
        self.posconv4 = PosKnnGNNLayer(embedding_size, embedding_size, self.usePI, self.useSI, self.dataset)
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index_l0, edge_weight_l0 = data[0].x, data[0].edge_index, data[0].edge_attr
        _, edge_index_l1, edge_weight_l1 = data[1].x, data[1].edge_index, data[1].edge_attr
        _, edge_index_l2, edge_weight_l2 = data[2].x, data[2].edge_index, data[2].edge_attr

        x0 = F.relu(self.posconv1(x, edge_index_l0, edge_weight_l0, d2an, True))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x1 = F.relu(self.posconv2(x, edge_index_l1, edge_weight_l1, d2an, True))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x = self.alpha1 * x0 + self.alpha2 * x1

        x0 = F.relu(self.posconv3(x, edge_index_l0, edge_weight_l0, d2an, False))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x1 = F.relu(self.posconv4(x, edge_index_l1, edge_weight_l1, d2an, False))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x = self.alpha1 * x0 + self.alpha2 * x1
        return x

    def noSI_forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index_l0, edge_weight_l0 = data.x, data.edge_index, data.edge_attr

        x0 = F.relu(self.posconv1.gcn_forward(x, edge_index_l0, edge_weight_l0, d2an, True))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x = x0

        x0 = F.relu(self.posconv3.gcn_forward(x, edge_index_l0, edge_weight_l0, d2an, False))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x = x0

        return x
# %==============================================================================================


class GraphEncoder(nn.Module):
    def __init__(self, feature_size, embedding_size, usePI, useSI, dataset, alpha1, alpha2):
        super(GraphEncoder, self).__init__()
        self.usePI = usePI
        self.useSI = useSI
        self.graph_embedding = KnnGNN(feature_size,
                                      embedding_size,
                                      self.usePI,
                                      self.useSI,
                                      dataset,
                                      alpha1,
                                      alpha2)

    def forward(self, network_data):
        if self.useSI:
            graph_node_embeddings = self.graph_embedding(network_data)
        else:
            graph_node_embeddings = self.graph_embedding.noSI_forward(network_data)
        return graph_node_embeddings


class AttnPooling(nn.Module):
    def __init__(self, embedding_size):
        super(AttnPooling, self).__init__()
        self.w_omega = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.u_omega = nn.Parameter(torch.Tensor(embedding_size, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, traj_node_outputs, padding_mask):
        u = torch.tanh(torch.matmul(traj_node_outputs, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        att = att.masked_fill(padding_mask == True, float('-inf'))
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = traj_node_outputs * att_score
        traj_outputs = torch.sum(scored_outputs, dim=1)
        return traj_outputs





class TrajCoorEncoder(nn.Module):
    def __init__(self, embedding_size, device=None):
        super(TrajCoorEncoder, self).__init__()
        self.linear_gi = nn.Linear(2, embedding_size)
        logging.info('SMNEncoder for TrajCoorEncoder')
        self.smn = SMNEncoder(embedding_size,
                              embedding_size,
                              device=device).to(device)

    def forward(self, traj_coor, seq_lengths):
        traj_coor = self.linear_gi(traj_coor)
        traj_emb_coor_outputs, traj_coor_outputs = self.smn([traj_coor, seq_lengths])
        return traj_emb_coor_outputs, traj_coor_outputs.transpose(1, 0)

    def get_coor_emb(self, traj_coor):
        traj_coor = self.linear_gi(traj_coor)
        return traj_coor


class Co_Att(nn.Module):
    def __init__(self, dim):
        super(Co_Att, self).__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.temperature = dim ** 0.5
        self.FFN = nn.Sequential(
            nn.Linear(dim, int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5), dim),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, seq_s, seq_t):
        h = torch.stack([seq_s, seq_t], 2)  # [n, 2, dim]
        # print('shape of h is: ', h.shape)
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # print('shape of attn is: ', attn.shape)
        attn = F.softmax(attn, dim=-1)
        attn_h = torch.matmul(attn, v)

        attn_o = self.FFN(attn_h) + attn_h
        attn_o = self.layer_norm(attn_o)

        att_s = attn_o[:, :, 0, :]
        att_t = attn_o[:, :, 1, :]

        return att_s,  att_t


class ST_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout_rate, device):
        super(ST_LSTM, self).__init__()
        self.device = device
        self.bi_lstm = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout_rate,
                               bidirectional=True)
        # self-attention weights
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def getMask(self, seq_lengths):
        """
        create mask based on the sentence lengths
        :param seq_lengths: sequence length after `pad_packed_sequence`
        :return: mask (batch_size, max_seq_len)
        """
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0

        return mask

    def forward(self, packed_input):
        # output features (h_t) from the last layer of the LSTM, for each t
        # (batch_size, seq_len, 2 * num_hiddens)
        packed_output, _ = self.bi_lstm(packed_input)  # output, (h, c)
        outputs, seq_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # get sequence mask
        mask = self.getMask(seq_lengths)

        # Attention...
        # (batch_size, seq_len, 2 * num_hiddens)
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        # (batch_size, seq_len)
        att = torch.matmul(u, self.u_omega).squeeze()

        # add mask
        att = att.masked_fill(mask == 0, -1e10)

        # (batch_size, seq_len,1)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        # normalization attention weight
        # (batch_size, seq_len, 2 * num_hiddens)
        scored_outputs = outputs * att_score

        # weighted sum as output
        # (batch_size, 2 * num_hiddens)
        out = torch.sum(scored_outputs, dim=1)
        return out


class GraphPretrainSTEncoder(nn.Module):
    def __init__(self,
                 useGraphNodeDist, useGraphCoorPred,
                 useTrajEmbDist,
                 useTrajAutoRegNode,
                 useTrajAutoRegCoor,

                 feature_size, embedding_size,
                 num_layers, dropout_rate, device,
                 usePI, useSI, useTransPI, useTransGI, dataset, alpha1, alpha2,
                 num_nodes):
        super(GraphPretrainSTEncoder, self).__init__()
        self.useGraphNodeDist = useGraphNodeDist
        self.useGraphCoorPred = useGraphCoorPred

        self.useTrajEmbDist = useTrajEmbDist
        self.useTrajAutoRegNode = useTrajAutoRegNode
        self.useTrajAutoRegCoor = useTrajAutoRegCoor

        self.device = device

        self.usePI = usePI
        self.useSI = useSI
        self.useTransPI = useTransPI
        self.useTransGI = useTransGI

        self.num_nodes = num_nodes
        # %=========================================================================================

        self.graph_encoder = GraphEncoder(feature_size, embedding_size,
                                          usePI, useSI, dataset, alpha1, alpha2)
        self.padding = nn.Embedding(1, embedding_size)

        self.coor_encoder = TrajCoorEncoder(embedding_size*2,
                                            nhead=4, d_hid=512, num_layers=num_layers,
                                            dropout_rate=dropout_rate, device=device)

        self.co_attention = Co_Att(embedding_size).to(device)
        self.encoder_ST = ST_LSTM(embedding_size*2, embedding_size*2, num_layers, dropout_rate, device)

        # %=========================================================================================
        if self.useGraphNodeDist:
            logging.info('useGraphNodeDist: ' + str(self.useGraphNodeDist))
            self.graph_node_dist_weight = nn.Parameter(torch.tensor(1., device=device))
            self.graph_node_dist = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, embedding_size))

        if self.useGraphCoorPred:
            logging.info('useGraphCoorPred: ' + str(self.useGraphCoorPred))
            self.graph_coor_pred_weight = nn.Parameter(torch.tensor(0., device=device))
            self.graph_coor_pred = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, 2))
        # %=========================================================================================

        if self.useTrajEmbDist:
            logging.info('useTrajEmbDist: ' + str(self.useTrajEmbDist))
            self.traj_emb_dist_weight = nn.Parameter(torch.tensor(1., device=device))
            self.traj_emb_dist = nn.Sequential(
                nn.Linear(embedding_size*6, embedding_size*6),
                nn.ReLU(),
                nn.Linear(embedding_size*6, embedding_size*6))

        if self.useTrajAutoRegNode and self.useTrajAutoRegCoor:
            logging.info('useTrajAutoRegNode: ' + str(self.useTrajAutoRegNode))
            logging.info('useTrajAutoRegCoor: ' + str(self.useTrajAutoRegCoor))
            logging.info('use encoder as input')
            self.traj_auto_reg_weight = nn.Parameter(torch.tensor(1., device=device))
            self.traj_auto_reg_node_weight = nn.Parameter(torch.tensor(0., device=device))
            self.traj_auto_reg_coor_weight = nn.Parameter(torch.tensor(1., device=device))
            self.node_decoder = nn.RNN(embedding_size*3, embedding_size*3, batch_first=True)

            self.traj_auto_reg_node_pred = nn.Linear(embedding_size*3, num_nodes)
            self.traj_auto_reg_coor_pred = nn.Linear(embedding_size*3, 2)

        # %=========================================================================================

    def get_traj_emb(self, graph_node_emb, node_seqs, coor_seqs, d2vec_seqs, seq_lengths):

        seq_lengths = seq_lengths.to(self.device)
        node_seqs = node_seqs.to(self.device)
        coor_seqs = coor_seqs.to(self.device)
        d2vec_seqs = d2vec_seqs.to(self.device)

        # node seqs
        node_seqs_emb = graph_node_emb[node_seqs].to(self.device)

        # time+node
        att_s, att_t = self.co_attention(node_seqs_emb, d2vec_seqs)
        st_input = torch.cat((att_s, att_t), dim=2)
        seq_lengths = seq_lengths.to('cpu')
        packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)
        node_out = self.encoder_ST(packed_input)

        # coor seqs
        gi_out, coor_outputs = self.coor_encoder(coor_seqs, seq_lengths)

        out = torch.cat((node_out, gi_out), dim=-1)
        return out

    def obtain_trm_src_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)
        for i, l in enumerate(seq_lengths):
            mask[i, :l] = 0
        return mask.bool()

    def obtain_trm_tgt_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())-1
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)
        for i, l in enumerate(seq_lengths):
            mask[i, :l-1] = 0
        return mask.bool()

    def obtain_trm_tgt_attn_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())
        attn_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool().to(self.device)
        return attn_mask

    def forward(self, network_data,
                graph_node_idx=None,
                graph_coor_idx=None,
                traj_aco=None, traj_pos=None, traj_neg=None,
                traj_node_tgt=None, traj_coor_tgt=None):

        graph_node_emb = self.graph_encoder(network_data)
        padding_indices = torch.LongTensor([0]).to(self.device)
        padding_embs = self.padding(padding_indices)  # shape: (2, embedding_size)
        graph_node_emb = torch.cat([graph_node_emb, padding_embs], dim=0)

        if self.useGraphNodeDist:
            aco_node_idx = graph_node_idx[0].to(self.device)
            pos_node_idx = graph_node_idx[1].to(self.device)
            neg_node_idx = graph_node_idx[2].to(self.device)
            aco_node_emb = self.graph_node_dist(graph_node_emb[aco_node_idx])
            pos_node_emb = self.graph_node_dist(graph_node_emb[pos_node_idx])
            neg_node_emb = self.graph_node_dist(graph_node_emb[neg_node_idx])
        else:
            aco_node_emb = None
            pos_node_emb = None
            neg_node_emb = None

        if self.useGraphCoorPred:
            graph_coor_idx = graph_coor_idx.to(self.device)
            graph_coor_pred = self.graph_coor_pred(graph_node_emb[graph_coor_idx])
        else:
            graph_coor_pred = None
        if self.useTrajEmbDist:
            traj_aco_emb = self.get_traj_emb(graph_node_emb,
                                             traj_aco[0], traj_aco[1], traj_aco[2], traj_aco[3])
            traj_pos_emb = self.get_traj_emb(graph_node_emb,
                                             traj_pos[0], traj_pos[1], traj_pos[2], traj_pos[3])
            traj_neg_emb = self.get_traj_emb(graph_node_emb,
                                             traj_neg[0], traj_neg[1], traj_neg[2], traj_neg[3])

            out_traj_aco_emb = self.traj_emb_dist(traj_aco_emb)
            out_traj_pos_emb = self.traj_emb_dist(traj_pos_emb)
            out_traj_neg_emb = self.traj_emb_dist(traj_neg_emb)

        else:
            out_traj_aco_emb = None
            out_traj_pos_emb = None
            out_traj_neg_emb = None

        traj_auto_reg_node_pred = None
        traj_auto_reg_coor_pred = None
        tgt_padding_mask = None
        if self.useTrajAutoRegNode or self.useTrajAutoRegCoor:

            # traj_aco[0] shape: [B, S]
            # traj_aco[1] shape: [B, S, 2]
            sample_num = config.pretrain['TrajSamNum']//2
            seq_len = traj_aco[3][::sample_num]  # 采样数量10
            tgt_padding_mask = self.obtain_trm_src_mask(seq_len).to(self.device)

            traj_node_tgt = traj_node_tgt.to(self.device)
            traj_coor_tgt = traj_coor_tgt.to(self.device)
            traj_node_tgt_emb = graph_node_emb[traj_node_tgt]
            traj_coor_tgt_emb = self.coor_encoder.get_coor_emb(traj_coor_tgt)


            traj_tgt_emb = torch.cat((traj_node_tgt_emb, traj_coor_tgt_emb), dim=-1)

            out_decoder_node_emb, hn = self.node_decoder(traj_tgt_emb)

            out_decoder_node_emb = out_decoder_node_emb[~tgt_padding_mask]
            traj_auto_reg_node_pred = self.traj_auto_reg_node_pred(out_decoder_node_emb)
            traj_auto_reg_coor_pred = self.traj_auto_reg_coor_pred(out_decoder_node_emb)

        return ((aco_node_emb, pos_node_emb, neg_node_emb),
                graph_coor_pred,
                (out_traj_aco_emb, out_traj_pos_emb, out_traj_neg_emb),
                (traj_auto_reg_node_pred, traj_auto_reg_coor_pred, tgt_padding_mask))
# %=========================================================================================


class GraphTrajSTEncoder(nn.Module):
    def __init__(self, graph_encoder, coor_encoder,
                 co_attention, encoder_ST,
                 feature_size, embedding_size, num_layers, dropout_rate, concat, device,
                 usePI, useSI, useTransPI, useTransGI, dataset, alpha1, alpha2):
        super(GraphTrajSTEncoder, self).__init__()
        self.device = device
        self.usePI = usePI
        self.useSI = useSI
        self.useTransPI = useTransPI
        self.useTransGI = useTransGI
        self.graph_encoder = graph_encoder

        self.coor_encoder = coor_encoder

        self.co_attention = co_attention
        self.encoder_ST = encoder_ST
        self.out_linear = nn.Linear(384, 384)

    def obtain_trm_src_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            mask[i, :l] = 0

        return mask.bool()

    def forward(self, network_data, node_seqs, coor_seqs, d2vec_seqs, seq_lengths):

        seq_lengths = seq_lengths.to(self.device)
        node_seqs = node_seqs.to(self.device)
        coor_seqs = coor_seqs.to(self.device)
        d2vec_seqs = d2vec_seqs.to(self.device)

        graph_node_embeddings = self.graph_encoder(network_data)

        # node seqs
        node_seqs_emb = graph_node_embeddings[node_seqs].to(self.device)

        # time+node
        att_s, att_t = self.co_attention(node_seqs_emb, d2vec_seqs)
        st_input = torch.cat((att_s, att_t), dim=2)
        seq_lengths = seq_lengths.to('cpu')
        packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)
        node_out = self.encoder_ST(packed_input)

        # coor seqs
        gi_out, coor_outputs = self.coor_encoder(coor_seqs, seq_lengths)

        out = torch.cat((node_out, gi_out), dim=-1)
        out = self.out_linear(out)

        return out
# %=================================================================================
