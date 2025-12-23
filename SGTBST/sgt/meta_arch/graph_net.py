import copy
import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max, scatter_add
from torch_geometric.nn import MetaLayer, GCNConv, MessagePassing
import torch.nn.functional as F

from projects.SGTBST.sgt.utils import flatten_boxes_dict, mask_boxes_dict
from projects.SGTBST.sgt.meta_arch.graph_build import build_sparse_graph
from projects.SGTBST.sgt.meta_arch.update_method import AttentionNodeUpdateNet
from projects.SGTBST.sgt.meta_arch.update_method import EdgeUpdateNet
from projects.SGTBST.sgt.meta_arch.update_method import EASP
from projects.SGTBST.sgt.meta_arch.update_method import EdgeEncodeNet
from projects.SGTBST.sgt.meta_arch.update_method import EdgeCNN
from torch_scatter import scatter_add

def build_layers(input_dim, fc_dims, dropout_p, norm_layer, act_func, bias=True):
    layers = []
    for fc_dim in fc_dims:
        layers.append(nn.Linear(input_dim, fc_dim, bias=bias))
        layers.append(norm_layer(fc_dim))
        layers.append(act_func)
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        input_dim = fc_dim
    return layers

class EdgeGNN(nn.Module):
    ## reference BST project - https://github.com/HYUNJS/SGT
    def __init__(self, cfg):
        super(EdgeGNN, self).__init__()

        cfg_node = cfg.NODE_MODEL
        cfg_edge = cfg.EDGE_MODEL
        self.n_iter = cfg.N_ITER  # number of message passing iteration
        self.deep_loss = cfg_edge.CLASSIFY.DEEP_LOSS
        self.exclude_init_loss = cfg_edge.CLASSIFY.EXCLUDE_INIT_LOSS
        # self.edge_encoder = EdgeEncodeNet(cfg_edge.ENCODE) # edge feature extracter (encoding edge input data into high dim feature)
        self.edge_encoder = EASP(cfg_edge.ENCODE,cfg_node.UPDATE) # edge feature extracter (encoding edge input data into high dim feature)
        self.MPNet = self.build_MPNet(cfg_node.UPDATE, cfg_edge.UPDATE) # message passing network

    def build_MPNet(self, node_model_param, edge_model_param):
        ## define node aggregation function
        node_agg_fn = node_model_param.AGG_FUNC.lower()
        assert node_agg_fn in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'"
        #动态设置节点聚合方法，名称为node_agg_fn
        if node_agg_fn == 'mean':  # 平均聚合
            # print("mean")
            node_agg_fn = lambda out, row, x_size, _: scatter_mean(out, row, dim=0, dim_size=x_size)
        elif node_agg_fn == 'max':  # 最大聚合
            # print("max")
            node_agg_fn = lambda out, row, x_size, _: scatter_max(out, row, dim=0, dim_size=x_size)[0]
        elif node_agg_fn == 'sum':  # 加和聚合
            # print("sums")
            node_agg_fn = lambda out, row, x_size, _: scatter_add(out, row, dim=0, dim_size=x_size)

        ## define node & edge feature dims
        self.reattach_initial_edge = edge_model_param.REATTACH
        self.reattach_initial_node = node_model_param.REATTACH
        self.skip_initial_edge = edge_model_param.SKIP_CONN
        self.skip_initial_node = node_model_param.SKIP_CONN
        # assert not (self.reattach_initial_edge and self.skip_initial_edge), "Choose only one option for edge feature - Reattach vs Skip"
        # assert not (self.reattach_initial_node and self.skip_initial_node), "Choose only one option for node feature - Reattach vs Skip"
        edge_factor = 2 if self.reattach_initial_edge else 1
        node_factor = 2 if self.reattach_initial_node else 1
        node_feat_dim, edge_feat_dim = node_model_param.IN_DIM, edge_model_param.IN_DIM
        edge_model_in_dim = node_factor * 2 * node_feat_dim + edge_factor * edge_feat_dim
        node_model_in_dim =  node_factor * node_feat_dim + edge_feat_dim
        if self.skip_initial_edge:
            edge_norm_layer = nn.LayerNorm
            self.edge_norm_layers = nn.ModuleList([edge_norm_layer(edge_feat_dim)]*self.n_iter)
        if self.skip_initial_node:
            node_norm_layer = nn.LayerNorm
            self.node_norm_layers = nn.ModuleList([node_norm_layer(node_feat_dim)]*self.n_iter)

        edge_update_model = EdgeCNN(edge_model_param, edge_model_in_dim)
        # node_update_model = NodeUpdateNet(node_model_param, node_model_in_dim, node_agg_fn)
        node_update_model = AttentionNodeUpdateNet(node_model_param, node_model_in_dim, node_agg_fn)

        ## package edge and node update networks into Message Passing Network
        return MetaLayer(edge_model=edge_update_model, node_model=node_update_model)

    def forward(self, graph_batch, **kwargs):
        """
        前向传播函数，用于处理一批图数据。

        参数:
        - graph_batch: 图数据的批处理对象，包含节点特征、边特征和图的连接信息。
        - **kwargs: 额外的关键字参数。

        返回:
        - latent_node_feats_list: 节点的潜在特征列表，包含了每一层的节点特征。
        - latent_edge_feats_list: 边的潜在特征列表，包含了每一层的边特征。
        """
        # 提取节点和边的初始特征
        initial_node_feats = graph_batch.x
        edge_index = graph_batch.edge_index
        initial_edge_feats = graph_batch.edge_attr
        latent_node_feats_list, latent_edge_feats_list = [], []

        # 通过边的连接信息提取源节点和目标节点的特征
        src = initial_node_feats[edge_index[0]]  # 提取源节点特征
        tgt = initial_node_feats[edge_index[1]]  # 提取目标节点特征
        # 使用边编码器更新初始边特征
        initial_edge_feats = self.edge_encoder(src, tgt, initial_edge_feats)

        # 初始化潜在节点和边的特征为它们的初始特征
        latent_node_feats, latent_edge_feats = initial_node_feats, initial_edge_feats

        # 在GNN中进行消息传递迭代
        for step in range(self.n_iter):
            # 如果在训练中且使用深度损失，则将当前的节点和边特征添加到列表中
            if self.training and self.deep_loss and not (self.exclude_init_loss and step == 0):
                latent_node_feats_list.append(latent_node_feats)
                latent_edge_feats_list.append(latent_edge_feats)

            # 如果设置，则在更新前重新附加初始特征
            if self.reattach_initial_node:  #false
                latent_node_feats = torch.cat([initial_node_feats, latent_node_feats], dim=1)
            if self.reattach_initial_edge:  # true
                latent_edge_feats = torch.cat([initial_edge_feats, latent_edge_feats], dim=1)

            # 执行消息传递网络
            latent_node_feats, latent_edge_feats, _ = self.MPNet(
                latent_node_feats, edge_index, latent_edge_feats
            )

            # 如果设置，则使用skip connection
            if self.skip_initial_node:
                latent_node_feats = initial_node_feats + latent_node_feats
                latent_node_feats = self.node_norm_layers[step](latent_node_feats)
            if self.skip_initial_edge:
                latent_edge_feats = initial_edge_feats + latent_edge_feats
                latent_edge_feats = self.edge_norm_layers[step](latent_edge_feats)

        # 将最后更新的节点和边特征添加到列表中
        latent_node_feats_list.append(latent_node_feats)
        latent_edge_feats_list.append(latent_edge_feats)

        return latent_node_feats_list, latent_edge_feats_list




class CrossFrameInteractionGNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg_graph = cfg.MODEL.TRACKER.GNN.GRAPH
        self.gnn = EdgeGNN(cfg.MODEL.TRACKER.GNN)
        self.num_proposals = cfg.MODEL.TRACKER.GNN.TOPK_DET
        self.topk = cfg_graph.TOPK
        self.graph_attr = cfg_graph.ATTR
        self.edge_attr = cfg.MODEL.TRACKER.GNN.EDGE_MODEL.ENCODE.EDGE_ATTR
        self.directional_edge_attr = cfg.MODEL.TRACKER.GNN.EDGE_MODEL.ENCODE.DIRECTIONAL_EDGE_ATTR
        self.output_size = cfg.MODEL.CENTERNET.OUTPUT_SIZE
        self.prev_det_proposal_flag = cfg.MODEL.TRACKER.GNN.GRAPH.PREV_DET_PROPOSAL_FLAG

    def separate_t1_t2_node_feats(self, t1_box_nums, t2_box_nums, updated_features):
        '''
        将更新后的特征分离为t1和t2节点特征。

        :param t1_box_nums: t1盒子数量的列表，长度为批次大小
        :param t2_box_nums: t2盒子数量的列表，长度为批次大小
        :param updated_features: t1和t2节点的合并特征，顺序为concat([t2, t1])
        :return: t1节点特征和t2节点特征
        '''
        # 初始化t1和t2的新特征列表
        new_t1_feats, new_t2_feats = [], []
        # 当前特征索引
        curr_idx = 0
        # 遍历批次中的每个样本
        for b_idx in range(len(t1_box_nums)):
            # 获取当前样本的t1和t2盒子数量
            t1_box_num, t2_box_num = t1_box_nums[b_idx], t2_box_nums[b_idx]
            # 从更新后的特征中提取当前样本的t2节点特征并添加到列表中
            new_t2_feats.append(updated_features.narrow(0, curr_idx, t2_box_num))
            # 从更新后的特征中提取当前样本的t1节点特征并添加到列表中
            new_t1_feats.append(updated_features.narrow(0, curr_idx + t2_box_num, t1_box_num))
            # 更新当前特征索引
            curr_idx = curr_idx + t1_box_num + t2_box_num

        # 将列表中的特征转换为单个张量
        new_t1_feats = torch.cat(new_t1_feats)
        new_t2_feats = torch.cat(new_t2_feats)
        # 返回分离后的t1和t2节点特征
        return new_t1_feats, new_t2_feats


    def build_graph(self, data_dict, history=None):
        '''
        构建图结构的数据准备函数，可选地合并历史轨迹信息
        :param data_dict: 包含图构建所需数据的字典
        :param history: 可选，包含历史轨迹信息的字典，用于在推断时合并历史轨迹
        :return: 准备好的用于构建图的数据字典
        '''
        # 从输入字典中提取分数、特征图特征、节点特征和框信息
        scores = data_dict['tensor']['scores']
        fmap_feats = data_dict['tensor']['fmap_feats']
        node_feats = data_dict['tensor']['node_feats']
        # split_fmap = data_dict['tensor']['split_fmap']
        split_node = data_dict['tensor']['split_node']


        boxes_dict = copy.deepcopy(data_dict['tensor']['boxes_dict'])
        box_nums = copy.deepcopy(data_dict['box_nums'])

        # 如果提供了历史轨迹信息，则合并历史轨迹
        if history is not None:
            # 根据标志和维度情况，准备历史框信息
            if self.prev_det_proposal_flag and history['boxes_dict']['xs'].dim() == 3:
                flatten_history_boxes = flatten_boxes_dict(history['boxes_dict'])
            else:
                flatten_history_boxes = history['boxes_dict']
            # 将当前框信息与历史框信息合并
            for k in boxes_dict.keys():
                boxes_dict[k] = torch.cat([boxes_dict[k], flatten_history_boxes[k]], dim=0)

            # 根据标志和维度情况，准备特征图和节点特征，并与历史特征合并
            if self.prev_det_proposal_flag and history['fmap_feats'].dim() == 3:
                fmap_feats = torch.cat([fmap_feats, history['fmap_feats'].flatten(0, 1)], dim=0)
                node_feats = torch.cat([node_feats, history['node_feats'].flatten(0, 1)], dim=0)
                split_node = torch.cat([split_node, history['split_node'].flatten(0, 1)], dim=0)
            else:
                fmap_feats = torch.cat([fmap_feats, history['fmap_feats']], dim=0)
                node_feats = torch.cat([node_feats, history['node_feats']], dim=0)
                split_node = torch.cat([split_node, history['split_node']], dim=0)
            # 合并分数信息和框数量信息
            scores = torch.cat([scores, history['scores']], dim=1)
            box_nums[0] += history['tids'].size(1)

        # 将合并后的数据组装成用于图构建的字典
        data_for_graph = {'fmap_feats': fmap_feats, 'node_feats': node_feats, 'boxes_dict': boxes_dict,
                          'box_nums': box_nums, 'scores': scores , 'split_node': split_node}
        return data_for_graph

    def forward(self, t1_info, t2_info, history=None):
        assert t1_info is not None and t2_info is not None

        N = len(t1_info['box_nums'])
        images_whwh = self.output_size[::-1] + self.output_size[::-1]
        images_whwh = [torch.tensor(images_whwh, dtype=torch.float32, device=t1_info['tensor']['fmap_feats'].device)] * N

        ## build graph
        t1_info_for_graph = self.build_graph(t1_info, history)
        t2_info_for_graph = self.build_graph(t2_info)
        t1_box_nums, t2_box_nums = t1_info_for_graph['box_nums'], t2_info_for_graph['box_nums']

        graph_data, edge_batch_masks, edge_batch_idx_offsets = build_sparse_graph(t1_info_for_graph, t2_info_for_graph,
                    images_whwh, k=self.topk, edge_attr=self.edge_attr, graph_attr=self.graph_attr, directional_edge_attr=self.directional_edge_attr)
        # print("类名为:", self.__class__.__name__, 'graph_data_size', graph_data.size())
        ## run GNN to update node and edge features
        updated_features = self.gnn(graph_data) # return: Tuple([node_feats] * num_deep_loss, [edge_feats] * num_deep_loss)
        node_features_list, edge_features_list = updated_features[0], updated_features[1]

        ## separate t1 and t2 nodes per each iteration
        new_t1_feats_list, new_t2_feats_list = [], []
        num_deep_loss = len(node_features_list)
        for i in range(num_deep_loss):
            new_t1_feats, new_t2_feats = self.separate_t1_t2_node_feats(t1_box_nums, t2_box_nums, node_features_list[i])
            new_t1_feats = new_t1_feats[:, :new_t1_feats.shape[1] // 5]  # 取full_feats,后面4个分离仅用于更新边特征,不参与损失计算
            new_t2_feats = new_t2_feats[:, :new_t2_feats.shape[1] // 5]  # 取full_feats,后面4个分离仅用于更新边特征,不参与损失计算

            new_t1_feats_list.append(new_t1_feats)
            new_t2_feats_list.append(new_t2_feats)

        edge_dict = {'features': edge_features_list, 'index': graph_data.edge_index, 'batch_masks': edge_batch_masks,
                     'edge_batch_idx_offsets': edge_batch_idx_offsets}
        return new_t1_feats_list, new_t2_feats_list, edge_dict