
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class AttentionNodeUpdateNet(nn.Module):


    def __init__(self, cfg, node_model_in_dim, node_agg_fn):
        super(AttentionNodeUpdateNet, self).__init__()
        out_dim = cfg.IN_DIM
        self.node_dim = cfg.IN_DIM
        fc_dims = cfg.FC_DIMS
        norm_layer = nn.LayerNorm
        act_func = nn.ReLU(inplace=True)
        dropout_p = cfg.DROPOUT_P
        self.shared_weight_flag = cfg.SHARED_FLAG
        self.self_loop = cfg.SELF_LOOP
        self.attention_fc = nn.Linear(2 * out_dim, 1)

        if self.shared_weight_flag:
            flow_layers = build_layers(node_model_in_dim, fc_dims + [out_dim], dropout_p, norm_layer, act_func)
            self.flow_shared_mlp = nn.Sequential(*flow_layers)
        else:
            flow_in_layers = build_layers(node_model_in_dim, fc_dims + [out_dim], dropout_p, norm_layer, act_func)
            flow_out_layers = build_layers(node_model_in_dim, fc_dims + [out_dim], dropout_p, norm_layer, act_func)
            self.flow_trk2det_mlp = nn.Sequential(*flow_in_layers)
            self.flow_det2trk_mlp = nn.Sequential(*flow_out_layers)

        if cfg.TMPNN_FLAG:
            node_layers = build_layers(out_dim * 2, [out_dim], dropout_p, norm_layer, act_func)
        else:
            node_layers = build_layers(out_dim, [out_dim], dropout_p, norm_layer, act_func)

        self.node_mlp = nn.Sequential(*node_layers)

        if self.self_loop:
            self.self_mlp = nn.Sequential(*build_layers(out_dim, [out_dim], dropout_p, norm_layer, act_func))

        self.node_agg_fn = node_agg_fn
        self.tmpnn_flag = cfg.TMPNN_FLAG

    def forward(self, x, edge_index, edge_attr, u=None, batch=None):
        x_split = x[:, self.node_dim:]
        x = x[:, :self.node_dim]
        row, col = edge_index

        # 计算检测到跟踪方向的掩码
        flow_det2trk_mask = row < col
        flow_det2trk_row, flow_det2trk_col = row[flow_det2trk_mask], col[flow_det2trk_mask]
        # print("edge_attr.size",edge_attr.size())  # edge_attr.size torch.Size([38794, 32])lj.为什么这个是32,而不是6?解码器进6出32,是个mlp
        flow_det2trk_input = torch.cat([x[flow_det2trk_col], edge_attr[flow_det2trk_mask]], dim=1)

        # 自注意力机制用于检测到跟踪方向
        det2trk_concat = torch.cat([x[flow_det2trk_col], x[flow_det2trk_row]], dim=-1)
        det2trk_attention_weights = F.leaky_relu(self.attention_fc(det2trk_concat))
        det2trk_attention_weights = F.softmax(det2trk_attention_weights, dim=0)
        flow_det2trk = det2trk_attention_weights * flow_det2trk_input

        if self.shared_weight_flag:
            flow_det2trk = self.flow_shared_mlp(flow_det2trk)
        else:
            flow_det2trk = self.flow_det2trk_mlp(flow_det2trk)

        # 计算跟踪到检测方向的掩码
        flow_trk2det_mask = row > col
        flow_trk2det_row, flow_trk2det_col = row[flow_trk2det_mask], col[flow_trk2det_mask]
        flow_trk2det_input = torch.cat([x[flow_trk2det_col], edge_attr[flow_trk2det_mask]], dim=1)

        # 自注意力机制用于跟踪到检测方向
        trk2det_concat = torch.cat([x[flow_trk2det_col], x[flow_trk2det_row]], dim=-1)
        trk2det_attention_weights = F.leaky_relu(self.attention_fc(trk2det_concat))
        trk2det_attention_weights = F.softmax(trk2det_attention_weights, dim=0)
        flow_trk2det = trk2det_attention_weights * flow_trk2det_input

        if self.shared_weight_flag:
            flow_trk2det = self.flow_shared_mlp(flow_trk2det)
        else:
            flow_trk2det = self.flow_trk2det_mlp(flow_trk2det)

        flow_det2trk = self.node_agg_fn(flow_det2trk, flow_det2trk_row, x.size(0), None)
        flow_trk2det = self.node_agg_fn(flow_trk2det, flow_trk2det_row, x.size(0), None)

        # 根据 tmpnn_flag 标志合并两个方向的信息
        if self.tmpnn_flag:
            flow_total = torch.cat([flow_trk2det, flow_det2trk], dim=1)
        else:
            flow_total = flow_trk2det + flow_det2trk
        updated_node_feats = self.node_mlp(flow_total)

        # 如果启用了自循环, 通过自循环多层感知机更新节点特征, 并与之前的结果相加
        if self.self_loop:
            self_node_feats = self.self_mlp(x)
            updated_node_feats = updated_node_feats + self_node_feats

        updated_node_feats = torch.cat((updated_node_feats, x_split), dim=1)  # 拼回去
        return updated_node_feats

def build_layers(in_dim, out_dims, dropout_p, norm_layer, act_func):
    layers = []
    for out_dim in out_dims:
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(norm_layer(out_dim))
        layers.append(act_func)
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        in_dim = out_dim
    return layers

class EdgeEncodeNet(nn.Module):
    def __init__(self, cfg):
        super(EdgeEncodeNet, self).__init__()
        input_dim = cfg.IN_DIM
        out_dim = cfg.OUT_DIM
        fc_dims = cfg.FC_DIMS
        norm_layer = nn.LayerNorm
        act_func = nn.ReLU(inplace=True)
        dropout_p = cfg.DROPOUT_P
        layers = build_layers(input_dim, fc_dims+[out_dim], dropout_p, norm_layer, act_func)
        layers += [nn.Linear(out_dim, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, edge_attr):
        # print("EdgeEncodeNet",edge_attr.size())
        return self.mlp(edge_attr)

class EdgeUpdateNet(nn.Module):
    def __init__(self, cfg, edge_model_in_dim):
        super(EdgeUpdateNet, self).__init__()
        out_dim = cfg.IN_DIM
        fc_dims = cfg.FC_DIMS
        norm_layer = nn.LayerNorm
        act_func = nn.ReLU(inplace=True)
        dropout_p = cfg.DROPOUT_P  # Dropout的概率。

        layers = build_layers(edge_model_in_dim, fc_dims+[out_dim], dropout_p, norm_layer, act_func)
        self.edge_mlp = nn.Sequential(*layers)  # 创建顺序模型。
        self.self_loop = cfg.SELF_LOOP

    def forward(self, src, tgt, edge_attr, u=None, batch=None):
        out = torch.cat([src, tgt, edge_attr], dim=1)
        return self.edge_mlp(out)


class EdgeCNN(nn.Module):

    def __init__(self, cfg, edge_model_in_dim):
        super(EdgeCNN, self).__init__()
        out_dim = cfg.IN_DIM
        fc_dims = cfg.FC_DIMS  # 全连接层的维度列表。
        act_func = nn.ReLU(inplace=True)
        dropout_p = cfg.DROPOUT_P  # Dropout的概率。

        # 初始化卷积层
        conv_layers = []
        in_channels = edge_model_in_dim
        for dim in fc_dims:
            conv_layers.append(nn.Conv1d(in_channels, dim, kernel_size=1))  # 使用1x1卷积替代全连接层
            conv_layers.append(act_func)
            if dropout_p > 0:
                conv_layers.append(nn.Dropout(dropout_p))
            in_channels = dim


        self.node_dim = in_channels
        self.k = 4  # splitparts = 4
        self.mlp1 = nn.Linear(in_channels, in_channels, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(in_channels, in_channels * self.k, bias=False)
        self.softmax = nn.Softmax(dim=1)

        conv_layers.append(nn.Conv1d(in_channels, out_dim, kernel_size=1))
        self.edge_cnn = nn.Sequential(*conv_layers)
        self.norm = nn.LayerNorm(out_dim)
        self.self_loop = cfg.SELF_LOOP  # 是否允许自循环

    def splitNodeFusion(self,node_split):
        b, c = node_split.size()  # node_split

        node_split = node_split.view(b, self.k, self.node_dim)
        a = torch.sum(node_split, dim=1)

        # 通过MLP网络处理全局和，生成加权权重
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.view(b, self.k, self.node_dim)
        # 计算注意力权重
        attention_weights = self.softmax(hat_a)
        weighted_node_split = attention_weights * node_split

        # 加权求和得到输出特征
        node_fusion = torch.sum(weighted_node_split, dim=1)

        return self.mlp1(node_fusion)

    def forward(self, src, tgt, edge_attr, u=None, batch=None):
        src_split = src[:, self.node_dim:]
        tgt_split = tgt[:, self.node_dim:]

        src_split = self.splitNodeFusion(src_split)
        tgt_split = self.splitNodeFusion(tgt_split)

        src = src[:, :self.node_dim]
        tgt = tgt[:, :self.node_dim]

        src = src + src_split
        tgt = tgt + tgt_split

        out = torch.cat([src, tgt, edge_attr], dim=1)
        out = out.unsqueeze(2)
        out = self.edge_cnn(out)
        out = out.squeeze(2)

        # 对输出应用 LayerNorm，注意这里需要把维度调整为 (batch_size, feature_dim)
        out = self.norm(out)
        return out


class EASP(nn.Module):
    def pad_edge_attr(self, edge_attr):
        """
        如果edge_attr的特征数无法被groups整除，从后往前复制特征进行补全
        """
        E, feature_dim = edge_attr.size()
        target_feature_dim = ((feature_dim + self.groups - 1) // self.groups) * self.groups

        if feature_dim == target_feature_dim:
            return edge_attr

        pad_size = target_feature_dim - feature_dim
        # 从后往前循环复制特征
        pad_features = []
        for i in range(pad_size):
            pad_features.append(edge_attr[:, -(i % feature_dim + 1)].unsqueeze(1))

        padded_features = torch.cat(pad_features, dim=1)
        return torch.cat([edge_attr, padded_features], dim=1)

    def reorder_features(self, edge_attr , cp_pattern = 'original',att_order = None):
        """
        根据指定的模式重新排列特征
        """
        if cp_pattern == 'original':
            return edge_attr

        E, feature_dim = edge_attr.size()

        if cp_pattern == 'evenOdd':
            # 将奇数索引特征放在前面，偶数索引特征放在后面
            odd_indices = torch.arange(1, feature_dim, 2)
            even_indices = torch.arange(0, feature_dim, 2)
            indices = torch.cat([odd_indices, even_indices])
            return edge_attr[:, indices]

        elif cp_pattern == 'manual':
            if att_order is None or len(att_order) != feature_dim:
                raise ValueError("Manual pattern requires valid att_order")
            att_order = list(att_order)
            att_order_tensor = torch.tensor(att_order, dtype=torch.long)
            return edge_attr[:, att_order_tensor]

    def __init__(self, edgeCfg ,nodeCfg):
        super(EASP, self).__init__()
        self.in_dim = edgeCfg.IN_DIM
        self.node_dim = nodeCfg.IN_DIM
        self.out_dim = edgeCfg.OUT_DIM
        # self.groups = cfg.GROUPS  # 特征分组数
        self.groups = 3  # 特征分组数
        self.en_node = True
        # 边特征降维卷积，基于组划分
        self.inter_channels = self.in_dim // self.groups

        # 卷积层，用于将每个组的特征进行卷积降维
        self.convs = nn.ModuleList([
            nn.Conv1d(4, 1, kernel_size=1)
            for _ in range(self.groups)
        ])

        # 注意力机制的线性层，用于对节点对之间的关系计算权重
        self.attention_fc = nn.Linear(self.en_node * self.node_dim +2 * self.node_dim, 1)
        fc_dims = edgeCfg.FC_DIMS  # 全连接层的维度列表。
        out_dim = edgeCfg.OUT_DIM
        norm_layer = nn.LayerNorm
        act_func = nn.ReLU(inplace=True)
        dropout_p = edgeCfg.DROPOUT_P
        layers = build_layers(self.groups,fc_dims+[out_dim], dropout_p, norm_layer, act_func)
        layers += [nn.Linear(out_dim, out_dim)]
        self.out_mlp = nn.Sequential(*layers)

    def forward(self, src,  tgt, edge_attr):

        #取原始特征维度tensor,不考虑分离维度创建边特征
        src = src[:, :self.node_dim]
        tgt = tgt[:, :self.node_dim]
        E = edge_attr.size(0)  # E 表示边的数量

        # 补全特征使其可被groups整除
        edge_attr = self.pad_edge_attr(edge_attr)
        # att_order = None
        # edge_attr = self.reorder_features(edge_attr, cp_pattern = 'evenOdd',att_order=att_order)
        att_order = {0,1,2,3,4,5}
        # edge_attr = self.reorder_features(edge_attr, cp_pattern = 'manual',att_order=att_order)
        edge_attr = self.reorder_features(edge_attr, cp_pattern='original', att_order=att_order)
        feature_dim = edge_attr.size(1)
        edge_attr = edge_attr.view(E, self.groups, feature_dim // self.groups)

        node_concat = torch.cat([src, tgt], dim=1)

        if(self.en_node):
            node_diff = src - tgt
            node_concat = torch.cat([node_diff, node_concat], dim=1)

        att_weights = F.leaky_relu(self.attention_fc(node_concat))
        att_weights = torch.sigmoid(att_weights)  # (E, 1) -> 归一化权重

        edge_outputs = []

        node_pair_group = torch.cat([src, tgt], dim=1)  # (E, 128)
        avg_node_pooled = node_pair_group.mean(dim=1, keepdim=True)
        max_node_pooled = node_pair_group.max(dim=1, keepdim=True)[0]

        for i in range(self.groups):
            # 选取当前组的特征
            edge_group = edge_attr[:, i, :]

            # 将节点特征拼接到组特征中进行卷积
            group_input = torch.cat([avg_node_pooled, max_node_pooled, edge_group], dim=1)  # (E, 3, inter_channels)

            group_input = group_input.unsqueeze(2)  # 转3维[E, 4, 1]
            edge_conv_output = self.convs[i](group_input)
            edge_outputs.append(edge_conv_output.squeeze(2) * att_weights)

        edge_output = torch.cat(edge_outputs, dim=1)  # (E, out_dim)
        return self.out_mlp(edge_output)


