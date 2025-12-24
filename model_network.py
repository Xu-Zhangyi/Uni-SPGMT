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
        input_a, input_len_a = inputs_a 

        outputs_a, (hn_a, cn_a) = self.t2s_model(input_a.permute(1, 0, 2))
        outputs_ca = F.sigmoid(self.res_linear1(outputs_a)) * F.tanh(self.res_linear2(outputs_a))
        outputs_hata = F.sigmoid(self.res_linear3(outputs_a)) * F.tanh(outputs_ca)
        outputs_fa = outputs_a + outputs_hata
        mask_out_a = []
        for b, v in enumerate(input_len_a):
            mask_out_a.append(outputs_fa[v - 1][b, :].view(1, -1))
        fa_outputs = torch.cat(mask_out_a, dim=0)
        return fa_outputs, outputs_fa


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, device, dropout: float = 0.1, max_len: int = 3700):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)  # (3700,1,64)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoderModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float, device):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, device, dropout)  # 64
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        

    def forward(self, src, attn_mask, padding_mask):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size] (S, N, E)
            attn_mask: `(N\cdot\text{num\_heads}, S, S)`.
            src_mask: Tensor, shape [seq_len, seq_len] (N, S)

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """

        src = src * math.sqrt(self.d_model)  # (367,32,64)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=padding_mask)

        return output


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
        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=False)  
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


class TrajNodeEncoder(nn.Module):
    def __init__(self,  embedding_size,
                 nhead=4, d_hid=512, num_layers=1, dropout_rate=0.3, device=None):
        super(TrajNodeEncoder, self).__init__()
        self.trm_encoder = TransformerEncoderModel(embedding_size,
                                                   nhead=nhead,
                                                   d_hid=d_hid,
                                                   nlayers=num_layers,
                                                   dropout=dropout_rate,
                                                   device=device)

    def forward(self, traj_node, attn_mask, padding_mask):
        traj_node_outputs = self.trm_encoder(src=traj_node.transpose(1, 0),
                                             attn_mask=attn_mask,
                                             padding_mask=padding_mask).transpose(1, 0)

        return traj_node_outputs


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


class GraphPretrainEncoder(nn.Module):
    def __init__(self,
                 useGraphNodeDist, useGraphCoorPred,
                 useTrajEmbDist,
                 useTrajAutoRegNode,
                 useTrajAutoRegCoor,

                 feature_size, embedding_size,
                 num_layers, dropout_rate, device,
                 usePI, useSI, useTransPI, useTransGI, dataset, alpha1, alpha2,
                 num_nodes):
        super(GraphPretrainEncoder, self).__init__()
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

        self.node_encoder = TrajNodeEncoder(embedding_size,
                                            nhead=4, d_hid=512, num_layers=num_layers,
                                            dropout_rate=dropout_rate, device=device)
        self.node_pooling = AttnPooling(embedding_size)
        self.coor_encoder = TrajCoorEncoder(embedding_size, device=device)

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
                nn.Linear(embedding_size*2, embedding_size*2),
                nn.ReLU(),
                nn.Linear(embedding_size*2, embedding_size*2))

        if self.useTrajAutoRegNode and self.useTrajAutoRegCoor:
            logging.info('useTrajAutoRegNode: ' + str(self.useTrajAutoRegNode))
            logging.info('useTrajAutoRegCoor: ' + str(self.useTrajAutoRegCoor))
            logging.info('use encoder as input')
            self.traj_auto_reg_weight = nn.Parameter(torch.tensor(1., device=device))
            self.traj_auto_reg_node_weight = nn.Parameter(torch.tensor(0., device=device))
            self.traj_auto_reg_coor_weight = nn.Parameter(torch.tensor(1., device=device))
            self.node_decoder = nn.RNN(embedding_size*2, embedding_size*2, batch_first=True)

            self.traj_auto_reg_node_pred = nn.Linear(embedding_size*2, num_nodes)
            self.traj_auto_reg_coor_pred = nn.Linear(embedding_size*2, 2)

        # %=========================================================================================

    def get_traj_emb(self, graph_node_emb, node, coor, length):

        node = node.to(self.device)
        coor = coor.to(self.device)
        length = length.to(self.device)

        node_emb = graph_node_emb[node].to(self.device)
        padding_mask = self.obtain_trm_src_mask(length)

        node_outputs = self.node_encoder(node_emb, attn_mask=None, padding_mask=padding_mask)
        traj_node_emb = self.node_pooling(node_outputs, padding_mask)

        traj_coor_emb, coor_outputs = self.coor_encoder(coor, length)

        out_traj_emb = torch.cat((traj_node_emb, traj_coor_emb), dim=-1)

        return node_outputs, coor_outputs, out_traj_emb

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
            node_aco, coor_aco, traj_aco_emb = self.get_traj_emb(graph_node_emb,
                                                                 traj_aco[0], traj_aco[1], traj_aco[2])
            node_pos, coor_pos, traj_pos_emb = self.get_traj_emb(graph_node_emb,
                                                                 traj_pos[0], traj_pos[1], traj_pos[2])
            node_neg, coor_neg, traj_neg_emb = self.get_traj_emb(graph_node_emb,
                                                                 traj_neg[0], traj_neg[1], traj_neg[2])

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

            sample_num = config.pretrain['TrajSamNum']//2
            seq_len = traj_aco[2][::sample_num]
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

    def eval_forward(self, network_data, node_seqs, coor_seqs, seq_lengths):

        graph_node_embeddings = self.graph_encoder(network_data)
        node_aco, coor_aco, traj_aco_emb = self.get_traj_emb(graph_node_embeddings,
                                                             node_seqs,
                                                             coor_seqs,
                                                             seq_lengths)
        return traj_aco_emb
# %=========================================================================================


class GraphTrajSimEncoder(nn.Module):
    def __init__(self, graph_encoder,
                 node_encoder, node_pooling, coor_encoder,
                 feature_size, embedding_size, num_layers, dropout_rate, concat, device,
                 usePI, useSI, useTransPI, useTransGI, dataset, alpha1, alpha2):
        super(GraphTrajSimEncoder, self).__init__()
        self.device = device
        self.usePI = usePI
        self.useSI = useSI
        self.useTransPI = useTransPI
        self.useTransGI = useTransGI
        self.graph_encoder = graph_encoder
        if self.useTransPI:
            logging.info('useTransPI: ' + str(self.useTransPI))
            self.node_encoder = node_encoder
            self.node_pooling = node_pooling
        else:
            self.lstm_model = torch.nn.LSTM(embedding_size, embedding_size, num_layers=num_layers)

        if self.useTransGI:
            logging.info('useTransGI: ' + str(self.useTransGI))
            self.coor_encoder = coor_encoder

    def obtain_trm_src_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())

        # (batch_size, max_seq_len)
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            mask[i, :l] = 0

        return mask.bool()

    def forward(self, network_data, node_seqs, coor_seqs, seq_lengths):

        seq_lengths = seq_lengths.to(self.device)
        node_seqs = node_seqs.to(self.device)
        coor_seqs = coor_seqs.to(self.device)

        graph_node_embeddings = self.graph_encoder(network_data)
        traj_padding_mask = self.obtain_trm_src_mask(seq_lengths)

        # (batch_size, max_len, embedding_size), trm
        if self.useTransPI:
            node_seqs_emb = graph_node_embeddings[node_seqs].to(self.device)
            node_outputs = self.node_encoder(node_seqs_emb, None, traj_padding_mask)
            pi_out = self.node_pooling(node_outputs, traj_padding_mask)
        else:
            # [S,B,d]
            outputs, (hn, cn) = self.lstm_model(node_seqs_emb.permute(1, 0, 2))
            mask_out_a = []
            for b, v in enumerate(seq_lengths):
                mask_out_a.append(outputs[v - 1][b, :].view(1, -1))
            pi_out = torch.cat(mask_out_a, dim=0)

        if self.useTransGI:
            gi_out, coor_outputs = self.coor_encoder(coor_seqs, seq_lengths)

        if self.useTransPI and self.useTransGI:
            out = torch.cat((pi_out, gi_out), dim=-1)
        elif self.useTransPI and not self.useTransGI:
            out = pi_out
        elif self.useTransGI and not self.useTransPI:
            out = gi_out

        return out
# %=================================================================================
