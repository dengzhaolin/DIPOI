import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp

import os
from DIPOI.Utils import *
from DIPOI.gMLP.gmlp import gMLP
from DIPOI.hGCN.hGCN import hGCNEncoder
from DIPOI.transformer.Layers import EncoderLayer
from DIPOI.transformerls.lstransformer import TransformerLS
from djhModel import InterGraphRep, SDE_Diffusion
from preprocess.cal_poi_pairwise import read_interaction


class Encoder(nn.Module):
    def __init__(
            self,
            num_types, d_model, n_layers, n_head, dropout):
        super().__init__()
        self.d_model = d_model

        if C.ENCODER == 'gMLP':
            self.gmlp = gMLP(d_model, 512, 700, n_layers)

        if C.ENCODER == 'hGCN':
            directory_path = './data/{dataset}/'.format(dataset=C.DATASET)
            poi_matrix_file = 'poi_matrix.npy'
            if not os.path.exists(directory_path + poi_matrix_file):
                print('Poi_matrix is not found, generating ...')
                read_interaction()
            print('Loading ', directory_path + poi_matrix_file, '...')
            self.ui_adj = np.load(directory_path + poi_matrix_file)
            self.ui_adj = sp.csr_matrix(self.ui_adj)
            print('Computing adj matrix ...')
            self.ui_adj = torch.tensor(self.normalize_graph_mat(self.ui_adj).toarray(), device=C.DEVICE)

            self.layer_stack = nn.ModuleList([
                hGCNEncoder(d_model, n_head)
                for _ in range(n_layers)])

        if C.ENCODER == 'Transformer':
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_model, 512, n_head, 512, 512, dropout=dropout)  # 512 1024 4 512 512 M
                for _ in range(n_layers)])

        if C.ENCODER == 'LSTransformer':
            self.layer_stack = nn.ModuleList([
                TransformerLS(d_model, n_head, dropout)
                for _ in range(n_layers)])

    def forward(self, event_type, enc_output, slf_attn_mask, non_pad_mask):
        """ Encode event sequences via masked self-attention. """
        if C.ENCODER == 'Transformer':
            for enc_layer in self.layer_stack:
                residual = enc_output
                enc_output, _ = enc_layer(
                    enc_output,
                    non_pad_mask=non_pad_mask,  # non_pad_mask
                    slf_attn_mask=slf_attn_mask,  # slf_attn_mask
                )
                enc_output += residual

        elif C.ENCODER == 'gMLP':
            enc_output = self.gmlp(enc_output)

        elif C.ENCODER == 'hGCN':
            # get individual adj
            adj = torch.zeros((event_type.size(0), event_type.size(1), event_type.size(1)), device=C.DEVICE)
            for i, e in enumerate(event_type):
                # Thanks to Lin Fang for reminding me to correct a mistake here.
                adj[i] = self.ui_adj[e-1, :][:, e-1]
                # performance can be enhanced by adding the element in the diagonal of the normalized adjacency matrix.
                adj[i] += self.ui_adj[e-1, e-1]

            for enc_layer in self.layer_stack:
                enc_output = enc_layer(enc_output, adj, event_type)

        if C.ENCODER == 'LSTransformer':
            for enc_layer in self.layer_stack:
                residual = enc_output
                enc_output = enc_layer(
                    enc_output,
                    mask=non_pad_mask
                )
                enc_output += residual

        return enc_output.mean(1)

    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        rowsum[rowsum==0] = 1e-9
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat


class Decoder(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.dropout = nn.Dropout(0.5)
        self.temperature = 512 ** 0.5

        self.conv = torch.nn.Conv2d(1, 1, (3, 3), padding=1, padding_mode='zeros')
        self.conv3 = torch.nn.Conv2d(1, 1, (700, 1))

        self.implicit_graph_features = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.implicit_graph_features.weight)

        self.implicit_conv_features = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.implicit_conv_features.weight)

        self.implicit_att_features = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.implicit_att_features.weight)



    def forward(self, user_embeddings, embeddings, enc_output, slf_attn_mask):
        outputs = []
        if C.ABLATION != 'w/oMatcher':# 是否包含Neural Matrix Factorization
            out = user_embeddings#tensor[32,]
            out = F.normalize(out, p=2, dim=-1, eps=1e-05)
            outputs.append(out)

        if C.ABLATION != 'w/oImFe':# 是否包含隐含特征学习

            if C.ABLATION != "w/oAtt":
                # seq1 implicit
                attn = torch.matmul(enc_output / self.temperature, enc_output.transpose(1, 2))
                attn = self.dropout(torch.tanh(attn)) * slf_attn_mask
                seq1_implicit = torch.matmul(attn, enc_output)
                seq1_implicit = self.implicit_att_features(seq1_implicit.mean(1))
                seq1_implicit = F.normalize(seq1_implicit, p=2, dim=-1, eps=1e-05)
                outputs.append(seq1_implicit/2)

            if C.ABLATION != "w/oConv":
                # seq2 implicit
                seq2_implicit = self.conv(enc_output.unsqueeze(1))
                seq2_implicit = self.conv3(seq2_implicit)
                seq2_implicit = self.implicit_conv_features(seq2_implicit.squeeze(1).squeeze(1))
                seq2_implicit = F.normalize(seq2_implicit, p=2, dim=-1, eps=1e-05)
                outputs.append(seq2_implicit*2)

        outputs = torch.stack(outputs, dim=0).sum(0)
        return out


class Model(nn.Module):
    def __init__(
            self,  num_types, d_model=256, n_layers=4, n_head=4, dropout=0.1, device=C.DEVICE,gdata=None):
        super(Model, self).__init__()

        self.event_emb = nn.Embedding(num_types, d_model, padding_idx=C.PAD)  # Embeding[]
        self.encoder = Encoder(
            num_types=num_types, d_model=d_model, #num_types 兴趣点数量
            n_layers=n_layers, n_head=n_head, dropout=dropout)

        self.num_types = num_types
        if(C.DATASET=="Foursquare"):
            d_m=256 #768
        else:d_m=1024

        #self.decoder = Decoder(d_model, num_types)
        self.decoder = Decoder(d_model, d_m)
        self.gdata=gdata
        self.device=device

        self.dis_Rep = InterGraphRep(num_types, d_m, self.gdata,device=self.device)
        self.dropout = nn.Dropout(p=0.4)

        self.SDEdiff = SDE_Diffusion(d_m, beta_min=C.Beta_min, beta_max=C.Beta_max, dt=C.Stepsize,device=self.device)

    def DiffGenerator(self, hat_E_u, E_w, target=None):
        local_embs = hat_E_u
        condition_embs = E_w.detach()
        self.step_num = 1000 #扩散步数

        # Reverse-time VP-SDE Generation Process 反向扩散过程
        L_u = self.SDEdiff.ReverseSDE_gener(local_embs, condition_embs, C.DiffSize)

        loss_div = None
        if target is not None: # training phase
            t_sampled = np.random.randint(1, self.step_num) / self.step_num

            # get marginal probability
            mean, std = self.SDEdiff.marginal_prob(target, t_sampled)
            z = torch.randn_like(target)
            perturbed_data = mean + std.unsqueeze(-1) * z

            # train a time-dependent score-based neural network to estimate marginal probability
            score = - self.SDEdiff.Est_score(perturbed_data, condition_embs)

            # Fisher divergence loss_div 通过 SDE 生成的噪声 z 和模型预测 score 计算 Fisher 散度损失。
            #Fisher 散度衡量 真实分布和模型生成分布之间的差异，用于指导 SDE 模型训练。
            loss_div = torch.square(score + z).mean()

        return L_u, loss_div #tensor[1024,64] tensor[]

    def forward(self, event_type,pos_lbl):#tensor(32,700)
        slf_attn_mask_subseq = get_subsequent_mask(event_type)  # M * L * L tensor[32,700,700]
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)  # M x lq x lk tensor[32,700,700]
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)#tensor[32,700,700]
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)#tensor[32,700,700]

        non_pad_mask = get_non_pad_mask(event_type)#tensor[32,700,1]

        # (K M)  event_emb: Embedding
        enc_output = self.event_emb(event_type) #tensor[32,700,1024]

        E_u = self.encoder(event_type, enc_output, slf_attn_mask, non_pad_mask) #tensor[32,1024] # H(j,:)

        #deng
        E_c = self.dis_Rep.encode(self.event_emb.weight) #Tensor[32511,1024]
        E_c = self.dropout(E_c)

        hat_L_u = self.decoder(E_u, self.event_emb.weight, enc_output, slf_attn_mask)

        L_u, loss_diff = self.DiffGenerator(hat_L_u, E_u,target=E_c[pos_lbl])

        prediction=2 * torch.matmul(E_u, E_c.t()) + torch.matmul(L_u, E_c.t())
        return prediction, E_u,loss_diff
