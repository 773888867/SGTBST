import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torchvision.ops as ops
import numpy as np


def build_sparse_graph(t1_info_for_graph, t2_info_for_graph, images_whwh, k=20,
                       directional_edge_attr=True, edge_attr=[], graph_attr=['iou', 'sim', 'dist']):
    """
    构建一个稀疏图，描述了跟踪框（t1）和检测框（t2）之间的关系。

    参数:
    t1_info_for_graph: 跟踪框的信息字典，包括特征和框的信息。
    t2_info_for_graph: 检测框的信息字典，包括特征和框的信息。
    images_whwh: 图像的宽高列表，用于标准化框的坐标。
    k: 对于每个图属性，保留的边的数量。
    directional_edge_attr: 边的属性是否考虑方向。
    edge_attr: 边的属性列表。
    graph_attr: 图的属性列表，用于定义边的生成方式。

    返回:
    graph_batch: torch_geometric.data.Batch对象，包含整个批次的图。
    edge_batch_masks: 每个图的边在批次中的掩码。
    edge_batch_idx_offsets: 边在批次中的索引偏移。
    """
    t1_split_nodes, t2_split_nodes = t1_info_for_graph['split_node'], t2_info_for_graph['split_node']
    t1_fmap_feats, t2_fmap_feats = t1_info_for_graph['fmap_feats'], t2_info_for_graph['fmap_feats']
    t1_node_feats, t2_node_feats = t1_info_for_graph['node_feats'], t2_info_for_graph['node_feats']
    t1_boxes_dict, t2_boxes_dict = t1_info_for_graph['boxes_dict'], t2_info_for_graph['boxes_dict']
    t1_box_nums, t2_box_nums = t1_info_for_graph['box_nums'], t2_info_for_graph['box_nums']
    device = t1_fmap_feats.device

    if np.array((t1_box_nums) != 0).sum() == 0:
        raise Exception('No batch to build graph')

    ## 创建图形属性以初始化
    ## nodes are in the order of [t2_boxes, t1_boxes]
    get_by_index = lambda data, idx: data[idx, :]
    node_feat_list, edge_idx_list, edge_feat_list, edge_num_list = [], [], [], []
    t1_curr_idx, t2_curr_idx = 0, 0
    for b_idx in range(len(t1_box_nums)):
        t1_box_num, t2_box_num = t1_box_nums[b_idx], t2_box_nums[b_idx]
        if t1_box_num == 0 or t2_box_nums[b_idx] == 0:
            node_feat_list.append([])
            edge_idx_list.append([])
            edge_feat_list.append([])
            edge_num_list.append(0)
            continue

        t1_next_idx = t1_curr_idx + t1_box_num
        t2_next_idx = t2_curr_idx + t2_box_num
        img_wh = images_whwh[b_idx][0:2]
        edge_k = min(k, t2_box_num)

        ## cat拼接t1、t2特征
        t1_node_feat = t1_node_feats[t1_curr_idx:t1_next_idx]
        t2_node_feat = t2_node_feats[t2_curr_idx:t2_next_idx]
        node_feat = torch.cat([t2_node_feat, t1_node_feat], dim=0) # t2-t1 order
        # print("t2_node_feat",t2_node_feat.size())
        # print("t1_node_feat",t1_node_feat.size())

        ## 后来添加的
        t1_split_node = t1_split_nodes[t1_curr_idx:t1_next_idx]
        t2_split_node = t2_split_nodes[t2_curr_idx:t2_next_idx]
        split_feat = torch.cat([t2_split_node, t1_split_node], dim=0)  # t2-t1 order
        # print("node_feat",node_feat.size())
        # print("split_feat",split_feat.size())
        node_feat = torch.cat([node_feat, split_feat], dim=1)

        node_feat_list.append(node_feat)

        ## find trk-det pairs - topk iou score
        t1_box = t1_boxes_dict['tlbr'][t1_curr_idx:t1_next_idx].detach()
        t2_box = t2_boxes_dict['tlbr'][t2_curr_idx:t2_next_idx].detach()
        iou_mat = ops.box_iou(t1_box, t2_box)  # [t1_box_num, t2_box_num]
        iou_max_v, iou_max_idx = iou_mat.topk(k=edge_k, dim=1)

        ## 找到trk-det对- topk Cosine相似性分数
        t1_fmap_feat = t1_fmap_feats[t1_curr_idx:t1_next_idx]
        t2_fmap_feat = t2_fmap_feats[t2_curr_idx:t2_next_idx]
        scaled_t1_fmap_feat = F.normalize(t1_fmap_feat, dim=1, eps=1e-8)
        scaled_t2_fmap_feat = F.normalize(t2_fmap_feat, dim=1, eps=1e-8)
        cos_sim_mat = torch.matmul(scaled_t1_fmap_feat, scaled_t2_fmap_feat.t())
        sim_max_v, sim_max_idx = cos_sim_mat.topk(k=edge_k, dim=1)

        ## 找到trk-det对- topk中心点的成对欧几里得距离
        xs_t1, ys_t1 = t1_boxes_dict['xs'][t1_curr_idx:t1_next_idx], t1_boxes_dict['ys'][t1_curr_idx:t1_next_idx]
        xs_t2, ys_t2 = t2_boxes_dict['xs'][t2_curr_idx:t2_next_idx], t2_boxes_dict['ys'][t2_curr_idx:t2_next_idx]
        xys_t1 = torch.cat([xs_t1, ys_t1], dim=1)
        xys_t2 = torch.cat([xs_t2, ys_t2], dim=1)
        dist_mat = torch.cdist(xys_t1, xys_t2)
        dist_mat_v, dist_max_idx = dist_mat.topk(k=edge_k, largest=False, dim=1)

        ## 创建边连接 [2, #edges]，按照起始 -> 结束 的方向排序
        ## 例如：边索引按照 [t2->t1; t1->t2] 的顺序
        t1_idx_dict, t2_idx_dict = {}, {}
        ## 构建t1节点在边中的索引矩阵
        t1_idx_matrix = (torch.arange(t1_box_num, dtype=torch.long, device=device) + t2_box_num).unsqueeze(1).repeat(1, edge_k)
        if 'iou' in graph_attr:
            t2_idx_dict['iou'] = iou_max_idx.flatten(0, 1)
            t1_idx_dict['iou'] = t1_idx_matrix.flatten(0, 1)
        if 'sim' in graph_attr:
            t2_idx_dict['sim'] = sim_max_idx.flatten(0, 1)
            t1_idx_dict['sim'] = t1_idx_matrix.flatten(0, 1)
        if 'dist' in graph_attr:
            t2_idx_dict['dist'] = dist_max_idx.flatten(0, 1)
            t1_idx_dict['dist'] = t1_idx_matrix.flatten(0, 1)
        ## 根据图属性合并t2节点索引
        t2_idx = torch.cat([t2_idx_dict[k] for k in graph_attr])
        ## 根据图属性合并t1节点索引
        t1_idx = torch.cat([t1_idx_dict[k] for k in graph_attr])
        ## 构建边索引张量
        edge_idx = torch.stack([t2_idx, t1_idx])  # t2 -> t1
        ## 去除重复边
        unique_edge_idx = torch.unique(edge_idx, dim=1)
        ## 将边索引添加到列表中，包括方向反转的边
        edge_idx_list.append(torch.cat([unique_edge_idx, unique_edge_idx.flip(dims=[0])], dim=1))

        ## make edge features - unique_edge_idx[0]: t2 node idx | unique_edge_idx[1]: t1 node idx
        ## 生成边缘特征- unique_edge_idx[0]: t2节点idx | unique_edge_idx[1]: t1节点idx
        edge_feat = []

        ## 计算成对的宽度和高度
        wh_t1 = t1_boxes_dict['wh'][t1_curr_idx:t1_next_idx]
        wh_t2 = t2_boxes_dict['wh'][t2_curr_idx:t2_next_idx]
        if wh_t1.size(-1) == 4:
            wh_t1 = wh_t1.reshape(-1, 2, 2).sum(dim=1).reshape(*wh_t1.shape[0:-1], 2)
        if wh_t2.size(-1) == 4:
            wh_t2 = wh_t2.reshape(-1, 2, 2).sum(dim=1).reshape(*wh_t2.shape[0:-1], 2)

        wh_t1, wh_t2 = F.relu(wh_t1), F.relu(wh_t2)  # to avoid torch.nan when wh_ratio<0
        # wh_t1, wh_t2 = wh_t1.clamp(min=0.5), wh_t2.clamp(min=0.5)  # to constrains the wh ratio value in the certain range
        whs_t1_by_idx = get_by_index(wh_t1, unique_edge_idx[1] - t2_box_num)
        whs_t2_by_idx = get_by_index(wh_t2, unique_edge_idx[0])

        if 'xy_diff' in edge_attr:
            ## scale by image width and height
            scaled_xy_t1 = get_by_index(xys_t1 / img_wh, unique_edge_idx[1] - t2_box_num)
            scaled_xy_t2 = get_by_index(xys_t2 / img_wh, unique_edge_idx[0])
            xy_diff = scaled_xy_t2 - scaled_xy_t1

            if directional_edge_attr:
                xy_diff = torch.cat([xy_diff, xy_diff * (-1)])
            else:
                xy_diff = torch.cat([xy_diff, xy_diff])
            # print("edge_feat.xy_diff.size",xy_diff[:, 0].size())
            # print("edge_feat.xy_diff.size",xy_diff[:, 1].size())
            edge_feat.append(xy_diff[:, 0])
            edge_feat.append(xy_diff[:, 1])

        if 'wh_ratio' in edge_attr:
            wh_ratio = whs_t2_by_idx / (whs_t1_by_idx + 1e-8) ## updated version
            # wh_ratio = torch.log(wh_ratio)
            wh_ratio = torch.log(wh_ratio + 1e-8)  # to avoid -inf when wh_ratio = 0
            # wh_ratio = torch.log(wh_ratio + 1) # this scaling constrains the minimum to be zero
            if directional_edge_attr:
                wh_ratio = torch.cat([wh_ratio, wh_ratio * (-1)])
            else:
                wh_ratio = torch.cat([wh_ratio, wh_ratio])
            # print("edge_feat.wh_ratio.size", wh_ratio[:, 0].size())
            # print("edge_feat.wh_ratio.size", wh_ratio[:, 1].size())
            edge_feat.append(wh_ratio[:, 0])
            edge_feat.append(wh_ratio[:, 1])

        # 计算两个矩阵之间的余弦相似度，并将相似度分数附加到一个列表中。
        if 'cos_sim' in edge_attr:
            sim_score = cos_sim_mat[unique_edge_idx[1] - t2_box_num, unique_edge_idx[0]]
            sim_score = torch.cat([sim_score, sim_score])
            # print("edge_feat.sim_score.size", sim_score.size())
            edge_feat.append(sim_score)

        if 'iou' in edge_attr:
            iou_score = iou_mat[unique_edge_idx[1] - t2_box_num, unique_edge_idx[0]]
            iou_score = torch.cat([iou_score, iou_score])
            # print("edge_feat.iou_score.size", iou_score.size())
            edge_feat.append(iou_score)

        edge_feat = torch.stack(edge_feat, dim=1)
        edge_feat_list.append(edge_feat)
        edge_num_list.append(edge_feat.size(0))

        ## keep node idx
        t1_curr_idx, t2_curr_idx = t1_next_idx, t2_next_idx

    ## build graph batch based on the node and edge initial features
    graphs = []
    edge_batch_masks, total_edge_num, idx_marker = [], sum(edge_num_list), 0
    edge_batch_idx_offsets = torch.zeros([total_edge_num], device=device, dtype=torch.long)
    for b_idx in range(len(t1_box_nums)):
        edge_num = edge_num_list[b_idx]
        new_idx_marker = idx_marker + edge_num
        edge_batch_mask = torch.tensor([False] * total_edge_num)
        edge_batch_mask[idx_marker:new_idx_marker] = True
        edge_batch_idx_offsets[idx_marker:new_idx_marker] = b_idx
        idx_marker = new_idx_marker
        edge_batch_masks.append(edge_batch_mask)

    for node_feat, edge_idx, edge_feat in zip(node_feat_list, edge_idx_list, edge_feat_list):
        if len(edge_feat) == 0:
            continue
        graph = Data(x=node_feat, edge_index=edge_idx, edge_attr=edge_feat)
        # print("build_sparse_graph","graph",graph)
        graphs.append(graph)
    graph_batch = Batch.from_data_list(graphs)

    return graph_batch, edge_batch_masks, edge_batch_idx_offsets
