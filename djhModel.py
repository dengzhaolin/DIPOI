import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, softmax
from torchsde import sdeint

import torch.nn.functional as F

import Constants as C


class InterGraphRep(nn.Module):
    def __init__(self, n_poi, hid_dim, G_D: Data,device=C.DEVICE):
        super(InterGraphRep, self).__init__()
        self.n_poi, self.hid_dim = n_poi, hid_dim
        self.GCN_layer = 2

        # aggregating own features:
        edge_index, _ = add_self_loops(G_D.edge_index,num_nodes=n_poi)
        dist_vec = torch.cat([G_D.edge_attr, torch.zeros((n_poi,)).to(C.DEVICE)]) # 补零
        # a_{i,j}^D:
        inter_edgeweight = torch.exp(-(dist_vec ** 2))  # 计算边的权重。距离越远，权重越小（指数衰减），降低远距离 POI 之间的影响。
        self.G_D = Data(edge_index=edge_index, edge_attr=inter_edgeweight)

        self.act = nn.LeakyReLU()
        self.InterDyGCN = nn.ModuleList()
        for _ in range(self.GCN_layer):
            self.InterDyGCN.append(InterDyGCN(self.hid_dim, self.hid_dim))

    def encode(self, poi_embs):
        layer_embs = poi_embs
        inter_embs = [layer_embs]

        for conv in self.InterDyGCN:
            layer_embs = conv(layer_embs, self.G_D) #通过 DisDyGCN 进行图卷积更新
            layer_embs = self.act(layer_embs)
            inter_embs.append(layer_embs)

        R_V = torch.stack(inter_embs, dim=1).mean(1) # 取所有 GCN 层的输出，进行平均池化，得到最终的地理表示 R_V。tensor[1024,64]
        return R_V

class InterDyGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, dist_embed_dim=64):
        super(InterDyGCN, self).__init__(aggr='add') # 选择邻居信息聚合方式为 'add'
        self._cached_edge = None
        self.linear = nn.Linear(in_channels, out_channels)# 线性变换层
        nn.init.xavier_uniform_(self.linear.weight)# 使用 Xavier 初始化

        # dynamic mechanism on diatance # 动态距离机制用于建模边的地理距离信息。通过两层全连接网络（MLP） 将 距离特征（1 维） 映射到 隐藏维度（out_channels）
        self.dist_transform = nn.Sequential(
            nn.Linear(1, dist_embed_dim),  # 1D -> dist_embed_dim
            nn.ReLU(),
            nn.Linear(dist_embed_dim, out_channels)  # dist_embed_dim -> out_channels
        )

        nn.init.xavier_uniform_(self.dist_transform[0].weight)
        nn.init.xavier_uniform_(self.dist_transform[2].weight)

    def forward(self, x, G_D: Data):
        #_cached_edge 用于存储 归一化后的边索引，防止重复计算
        if self._cached_edge is None:
            self._cached_edge = gcn_norm(G_D.edge_index, add_self_loops=False) #归一化边权重
        edge_index, norm_weight = self._cached_edge# 取出归一化边索引 & 权重
        x = self.linear(x)# 线性变换节点特征
        h = self.propagate(edge_index, x=x, norm=norm_weight, dist_vec=G_D.edge_attr) # 进行消息传播
        return h

    def message(self, x_j, norm, dist_vec):
        dist_weight = self.dist_transform(dist_vec.unsqueeze(-1))# 计算距离权重。dist_vec 是 边的距离属性，维度为 [E]（边数），通过 self.dist_transform 映射为 [E, out_channels]
        message_trans = norm.unsqueeze(-1) * x_j * dist_weight# 加权消息传播
        '''
        norm.unsqueeze(-1): 归一化权重（[E] -> [E, 1]）。
        x_j: 邻居节点特征（[E, out_channels]）
        dist_weight: 通过距离转换后的权重（[E, out_channels]）
        最终消息：邻居特征 × 归一化系数 × 距离权重。
        '''
        return message_trans




class SDE_Diffusion(nn.Module):
    def __init__(self, hid_dim, beta_min, beta_max, dt ,device=None):
        super(SDE_Diffusion, self).__init__()
        self.hid_dim = hid_dim
        self.beta_min, self.beta_max = beta_min, beta_max #控制扩散强度的参数（从小噪声到大噪声）
        self.dt = dt #SDE 的时间步长（控制数值求解的精度）。
        self.device = device

        # score-based neural network stacked multiple fully connected layers 评分网络
        self.score_FC = nn.Sequential(
            nn.Linear(2 * hid_dim, 2 * hid_dim),
            nn.BatchNorm1d(2 * hid_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(2 * hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(hid_dim, hid_dim)
        )

        for w in self.score_FC:
            if isinstance(w, nn.Linear):
                nn.init.xavier_uniform_(w.weight)

    # a time-dependent score-based neural network to estimate marginal probability 估计分数
    def Est_score(self, x, condition):
        # this score is used in SDE solving to guide the evolution of stochastic processes 计算 扩散过程的梯度信息，用于 引导 SDE 逆向传播，类似于扩散模型中的 score function
        return self.score_FC(torch.cat((x, condition), dim=-1))

    #  Define the drift term f and diffusion term g of Forward SDE 前向 SDE 公式（24)
    def ForwardSDE_diff(self, x, t):
        def f(_t, y): #漂移项 f(_t, y)引导 y 向 零均值 方向演化（数据去信息化）。
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            return -0.5 * beta_t * y

        def g(_t, y): #扩散项 g(_t, y)：引入噪声，使 y 演化成 高斯噪声。
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs = y.size(0)
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            return (beta_t ** 0.5) * noise

        ts = torch.Tensor([0, t]).to(self.device)
        output = sdeint(SDEsolver(f, g), y0=x, ts=ts, dt=self.dt)[-1]
        return output
    #逆向 SDE 公式（26）
    def ReverseSDE_gener(self, x, condition, T):
        def get_beta_t(_t):
            beta_t_1 = self.beta_min + _t * (self.beta_max - self.beta_min)
            beta_t_2 = self.beta_min + (self.beta_max - self.beta_min) * torch.sin(torch.pi/2 * _t )**2
            beta_t_3 = self.beta_min * torch.exp(torch.log(torch.tensor(self.beta_max / self.beta_min)) * _t)
            beta_t_4 = self.beta_min + _t * (self.beta_max - self.beta_min)**2
            beta_t_5 = 0.1 * torch.exp(6 * _t)
            return beta_t_1

        # drift term f(): {_t: current time point, y: current state, returns the value of the drift term}
        def f(_t, y): #漂移项 f(_t, y)
            beta_t = get_beta_t(_t)
            score = self.score_FC(torch.cat((x, condition), dim=-1))
            ## score = self.score_FC(y)
            drift = -0.5 * beta_t * y - beta_t * score
            return drift

        # diffusion term g(): {_t: current time point, y: current state, returns the value of the diffusion term}
        def g(_t, y):  #扩散项 g(_t, y)，不变，保持与 ForwardSDE_diff 相同的噪声强度。
            beta_t = get_beta_t(_t)
            bs = y.size(0)
            # noise Tensors [bs, self.hid_dim, 1] = [1024, 64, 1]  with all elements of 1
            noise = torch.ones((bs, self.hid_dim, 1), device=y.device)
            diffusion = (beta_t ** 0.5)  * noise
            return diffusion

        def g_diagonal_noise(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim = y.size(0), y.size(1)
            noise = torch.randn((bs, dim), device=y.device)
            diagonal_beta = torch.diag(beta_t * torch.ones(dim, device=y.device))
            diffusion = (diagonal_beta ** 0.5).mm(noise.t()).t()
            return diffusion + y

        def g_vector_noise(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim, brownian_size = y.size(0), y.size(1), y.size(1)
            noise = torch.randn((bs, dim, brownian_size), device=y.device)
            diffusion = (beta_t ** 0.5) * noise
            return diffusion

        def g_full_cov_noise_3d(_t, y):
            beta_t = self.beta_min + _t * (self.beta_max - self.beta_min)
            bs, dim = y.size(0), y.size(1)
            noise = torch.randn((bs, dim, dim), device=y.device)
            covariance_matrix = torch.eye(dim, device=y.device)
            covariance_matrix = covariance_matrix * beta_t
            cholesky_matrix = torch.linalg.cholesky(covariance_matrix)
            diffusion = torch.einsum('bij,jk->bik', noise, cholesky_matrix)
            return diffusion

        ts = torch.Tensor([0, T]).to(self.device)

        # output is a pure (noise-free) location archetype vector L_u
        output = sdeint(SDEsolver(f, g), y0=x, ts=ts, dt=self.dt)[-1]

        return output
    # 计算边际概率，计算 x 在时间 t 的 均值和标准差，用于评估数据的分布变化。
    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        log_mean_coeff = torch.Tensor([log_mean_coeff]).to(x.device)
        mean = torch.exp(log_mean_coeff.unsqueeze(-1)) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

class SDEsolver(nn.Module):
    sde_type = 'stratonovich'   # available: {'ito', 'stratonovich'}
    noise_type = 'scalar'       # available: {'general', 'additive', 'diagonal', 'scalar'}

    def __init__(self, f, g):
        super(SDEsolver).__init__()
        self.f, self.g = f, g

    def f(self, t, y):
        return self.f(t, y)

    def g(self, t, y):
        return self.g(t, y)
