import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict
from thop.fx_profile import null_print
#
# class PartAttentionWeight(nn.Module):
#     def __init__(self, num_parts=4, in_channels=64, merge=True):
#         super().__init__()
#         self.num_parts = num_parts
#         self.merge = merge
#
#         self.attention = nn.Sequential(
#             nn.Linear(in_channels, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_parts),
#             nn.Softmax(dim=-1)
#         )
#
#         # 根据merge参数初始化不同的bias
#         if merge:
#             # 当merge=True时，底部特征权重更大
#             self.register_parameter('part_bias',
#                                     nn.Parameter(torch.FloatTensor([0.0, 0.0, 0.0, 1.0])))
#         else:
#             # 当merge=False时，所有部分权重相同
#             self.register_parameter('part_bias',
#                                     nn.Parameter(torch.FloatTensor([1.0, 1.0, 1.0, 1.0])))
#
#     def forward(self, features):
#         base_weights = self.attention(features)
#         part_bias = F.softmax(self.part_bias, dim=0)
#         weights = base_weights + part_bias.view(1, 1, -1)
#         weights = F.normalize(weights, p=1, dim=-1)
#         return weights
#
#
# def pipeline_node_feature(node_feature, index, tlbr, use_transform=True, num_parts=4, merge=True):
#     """
#     Args:
#         node_feature: 原始特征图 [B, C, H, W]
#         index: 目标索引 [B, K]
#         tlbr: bbox坐标 [B, K, 4]
#         use_transform: 是否转换特征图格式
#         num_parts: 垂直分割的部分数
#         merge: 是否合并特征
#     Returns:
#         merge=True时: enhanced_features [B, K, C]
#         merge=False时: part_features [B, K, num_parts*C]
#     """
#     device = node_feature.device
#
#     if use_transform:
#         batch, channel = node_feature.shape[:2]
#         node_feature = node_feature.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
#
#     # 1. 收集特征
#     dim = node_feature.size(-1)
#     index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
#     gathered_features = node_feature.gather(dim=1, index=index)  # [B, K, C]
#
#     # 2. 获取高度信息
#     heights = tlbr[..., 3] - tlbr[..., 1]  # [B, K]
#
#     # 3. 垂直分割并处理
#     part_features = []
#     for i in range(num_parts):
#         start_ratio = i / num_parts
#         end_ratio = (i + 1) / num_parts
#
#         start_h = tlbr[..., 1] + heights * start_ratio
#         end_h = tlbr[..., 1] + heights * end_ratio
#
#         y_center = (tlbr[..., 1] + tlbr[..., 3]) / 2
#         part_mask = ((y_center >= start_h) & (y_center < end_h)).float().to(device)
#
#         part_feat = gathered_features * part_mask.unsqueeze(-1)
#         part_features.append(part_feat)
#
#     # 4. 生成权重
#     attention_module = PartAttentionWeight(
#         num_parts=num_parts,
#         in_channels=dim,
#         merge=merge
#     ).to(device)
#     weights = attention_module(gathered_features)  # [B, K, num_parts]
#
#     # 5. 处理特征
#     part_features = torch.stack(part_features, dim=-2)  # [B, K, num_parts, C]
#
#     if merge:
#         # 合并模式：加权求和得到增强特征
#         weights = weights.unsqueeze(-1)  # [B, K, num_parts, 1]
#         enhanced_features = (part_features * weights).sum(dim=-2)  # [B, K, C]
#         return enhanced_features
#     else:
#         # 非合并模式：扩展每个部分的特征到原始大小并拼接
#         B, K, P, C = part_features.shape
#         # 将每个部分的特征扩展到与原始特征相同的大小
#         expanded_parts = []
#         for i in range(num_parts):
#             part = part_features[:, :, i, :]  # [B, K, C]
#             # 使用线性层扩展特征
#             expand_layer = nn.Linear(C, C).to(device)
#             expanded_part = expand_layer(part)  # [B, K, C]
#             expanded_parts.append(expanded_part)
#
#         # 拼接所有部分的特征
#         concatenated_features = torch.cat(expanded_parts, dim=-1)  # [B, K, num_parts*C]
#         return concatenated_features
#
# # 测试代码
# if __name__ == "__main__":
#     # 在GPU上测试
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 创建测试数据
#     B, C, H, W = 8, 64, 152, 272
#     K = 100
#
#     node_feature = torch.randn(B, C, H, W).to(device)
#     index = torch.randint(0, H * W, (B, K)).to(device)
#     tlbr = torch.randint(0, min(H, W), (B, K, 4)).float().to(device)
#
#     # 运行pipeline
#     enhanced_features = pipeline_node_feature(
#         node_feature=node_feature,
#         index=index,
#         tlbr=tlbr,
#         use_transform=True
#     )
#
#     print(f"Input shape: {node_feature.shape}")
#     print(f"Enhanced features shape: {enhanced_features.shape}")

# ---------------------------------------------------------------#
# 门控版本
# 额外添加的可视化模块


# #版本一,速度快但是有问题,h为特征图中个体的高度,但提取后的单个目标h默认为1,出现了多目标特征结合问题,属于是结合了邻居节点的特征
# # 那么其实也是有意义的,聚合邻居节点的特征,大,但是细节可能有问题,可解释性这里可能存在问题
# class GatedPartAttention(nn.Module):
#     def __init__(self, num_parts=4, in_channels=64, merge=True):
#         super().__init__()
#         # self.weight_analyzer = WeightAnalyzer(num_parts)
#         self.num_parts = num_parts
#         self.merge = merge
#
#
#         # 特征转换层，用于生成门控信息
#         self.feature_transform = nn.Sequential(
#             nn.Linear(in_channels, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#
#         # 为每个部分创建独立的门控单元
#         self.gate_networks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(128, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 1),
#                 nn.Sigmoid()
#             ) for _ in range(num_parts)
#         ])
#
#         # 注意力生成器，用于生成额外的注意力权重
#         self.attention_net = nn.Sequential(
#             nn.Linear(128, num_parts),
#             nn.Softmax(dim=-1)
#         )
#
#         # 根据merge参数初始化基础权重
#         if merge:
#             # merge=True时，底部权重更大
#             base_weights = torch.FloatTensor([1.0, 1.0, 1.0, 2.0])
#         else:
#             # merge=False时，权重相等
#             base_weights = torch.FloatTensor([1.0] * num_parts)
#         self.register_parameter('base_weights',
#                                 nn.Parameter(base_weights))
#
#     def forward(self, features):
#         """
#         Args:
#             features: [B, K, C]
#         Returns:
#             weights: [B, K, num_parts]
#         """
#         B, K, C = features.shape
#
#         # 1. 特征转换
#         transformed_features = self.feature_transform(features)  # [B, K, 128]
#
#         # 2. 生成门控信号
#         gates = []
#         for gate_net in self.gate_networks:
#             # 为每个部分生成门控值
#             gate = gate_net(transformed_features)  # [B, K, 1]
#             gates.append(gate)
#         gates = torch.cat(gates, dim=-1)  # [B, K, num_parts]
#
#         # 3. 生成注意力权重
#         attention_weights = self.attention_net(transformed_features)  # [B, K, num_parts]
#
#         # 4. 结合基础权重、门控信号和注意力权重
#         base_weights = F.softmax(self.base_weights, dim=0)  # [num_parts]
#
#         # 扩展基础权重维度以进行广播
#         base_weights = base_weights.view(1, 1, -1).expand(B, K, -1)
#
#         # 组合所有权重
#         final_weights = base_weights * gates * attention_weights
#
#         # 归一化
#         final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-6)
#
#         # 收集权重信息
#         # self.weight_analyzer.collect_weights(
#         #     self.base_weights,
#         #     gates,
#         #     attention_weights,
#         #     final_weights
#         # )
#
#         # print(self.base_weights, gates, attention_weights, final_weights)
#         return final_weights
#
#
# def pipeline_node_feature(node_feature, index, tlbr, use_transform=True, num_parts=4, merge=True):
#     """
#     使用门控机制的pipeline版本
#     """
#     device = node_feature.device
#
#     if use_transform:
#         batch, channel = node_feature.shape[:2]
#         node_feature = node_feature.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
#
#     # 1. 收集特征
#     dim = node_feature.size(-1)
#     index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
#     gathered_features = node_feature.gather(dim=1, index=index)  # [B, K, C]
#
#     # 2. 获取高度信息
#     heights = tlbr[..., 3] - tlbr[..., 1]  # [B, K]
#
#     # 3. 垂直分割
#     part_features = []
#     for i in range(num_parts):
#         start_ratio = i / num_parts
#         end_ratio = (i + 1) / num_parts
#
#         start_h = tlbr[..., 1] + heights * start_ratio
#         end_h = tlbr[..., 1] + heights * end_ratio
#
#         y_center = (tlbr[..., 1] + tlbr[..., 3]) / 2
#         part_mask = ((y_center >= start_h) & (y_center < end_h)).float().to(device)
#
#         part_feat = gathered_features * part_mask.unsqueeze(-1)
#         part_features.append(part_feat)
#
#     # 4. 使用门控注意力生成权重
#     gated_attention = GatedPartAttention(
#         num_parts=num_parts,
#         in_channels=dim,
#         merge=merge
#     ).to(device)
#     weights = gated_attention(gathered_features)  # [B, K, num_parts]
#
#     # 5. 处理特征
#     part_features = torch.stack(part_features, dim=-2)  # [B, K, num_parts, C]
#
#     if merge:
#         # 合并模式
#         weights = weights.unsqueeze(-1)  # [B, K, num_parts, 1]
#         enhanced_features = (part_features * weights).sum(dim=-2)  # [B, K, C]
#         return enhanced_features
#     else:
#         # 扩展模式
#         B, K, P, C = part_features.shape
#         weights = weights.unsqueeze(-1)  # [B, K, num_parts, 1]
#
#         # 对每个部分的特征进行加权和扩展
#         expanded_parts = []
#         for i in range(num_parts):
#             # 提取当前部分的特征
#             part = part_features[:, :, i, :]  # [B, K, C]
#
#             # 创建特征扩展层
#             expand_layer = nn.Sequential(
#                 nn.Linear(C, C),
#                 nn.LayerNorm(C),
#                 nn.ReLU(),
#                 nn.Dropout(0.1)
#             ).to(device)
#
#             # 扩展特征
#             expanded_part = expand_layer(part) * weights[:, :, i, :]
#             expanded_parts.append(expanded_part)
#
#         # 拼接所有部分的特征
#         concatenated_features = torch.cat(expanded_parts, dim=-1)  # [B, K, num_parts*C]
#         return concatenated_features


"""
        版本二:使用平均池化与tlbr信息进行了处理,逻辑上没有问题了,但是效率极低
"""
# class GatedPartAttention(nn.Module):
#     def __init__(self, num_parts=4, in_channels=64, merge=True):
#         super().__init__()
#         self.num_parts = num_parts
#         self.merge = merge
#
#         self.feature_transform = nn.Sequential(
#             nn.Linear(num_parts * in_channels, 128),  # 修改输入维度
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#
#         self.gate_networks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(128, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 1),
#                 nn.Sigmoid()
#             ) for _ in range(num_parts)
#         ])
#
#         self.attention_net = nn.Sequential(
#             nn.Linear(128, num_parts),
#             nn.Softmax(dim=-1)
#         )
#
#         base_weights = torch.FloatTensor([1.0, 1.0, 1.0, 2.0] if merge else [1.0] * num_parts)
#         self.register_parameter('base_weights', nn.Parameter(base_weights))
#
#     def forward(self, features):
#         """
#         Args:
#             features: [B, K, num_parts * C] (已展平的特征)
#         Returns:
#             weights: [B, K, num_parts]
#         """
#         B, K, _ = features.shape
#
#         transformed_features = self.feature_transform(features)
#
#         gates = torch.cat([gate_net(transformed_features) for gate_net in self.gate_networks], dim=-1)
#         attention_weights = self.attention_net(transformed_features)
#
#         base_weights = F.softmax(self.base_weights, dim=0).view(1, 1, -1).expand(B, K, -1)
#         final_weights = base_weights * gates * attention_weights
#         final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-6)
#
#         return final_weights
# def pipeline_node_feature(node_feature, index_group, tlbr, use_transform=True, num_parts=4, merge=True):
#     """使用门控机制的pipeline版本"""
#     device = node_feature.device
#     batch, channel, height, width = node_feature.shape
#     B = batch
#     K = index.size(1)
#
#     # 1. 收集每个检测框的特征
#     gathered_features = []
#     for b in range(B):
#         features_per_image = []
#         for k in range(K):
#             # 获取当前检测框的坐标并确保在特征图范围内
#             t = max(0, int(tlbr[b, k, 1]))
#             l = max(0, int(tlbr[b, k, 0]))
#             bottom = min(height, int(tlbr[b, k, 3]))
#             r = min(width, int(tlbr[b, k, 2]))
#             box_h = bottom  - t
#
#             # 将检测框垂直分成num_parts段
#             part_features = []
#             for p in range(num_parts):
#                 start_h = t + (box_h * p) // num_parts
#                 end_h = t + (box_h * (p + 1)) // num_parts
#
#                 part_feat = node_feature[b, :, start_h:end_h, l:r]
#                 if part_feat.numel() > 0:
#                     part_feat = torch.mean(part_feat, dim=(1, 2))
#                 else:
#                     part_feat = torch.zeros(channel, device=device)
#
#                 part_features.append(part_feat)
#
#             box_features = torch.stack(part_features, dim=0)
#             features_per_image.append(box_features)
#
#         features_per_image = torch.stack(features_per_image, dim=0)
#         gathered_features.append(features_per_image)
#
#     gathered_features = torch.stack(gathered_features, dim=0)  #(16,100,4,64)
#
#     # 2. 使用门控注意力生成权重
#     gated_attention = GatedPartAttention(
#         num_parts=num_parts,
#         in_channels=channel,
#         merge=merge
#     ).to(device)
#
#     flat_features = gathered_features.view(B, K, -1)
#     weights = gated_attention(flat_features)
#
#     if merge:
#         weights = weights.unsqueeze(-1)
#         enhanced_features = (gathered_features * weights).sum(dim=2)
#         return enhanced_features
#     else:
#         weights = weights.unsqueeze(-1)
#         expanded_parts = []
#         for i in range(num_parts):
#             part = gathered_features[:, :, i, :]
#             expand_layer = nn.Sequential(
#                 nn.Linear(channel, channel),
#                 nn.LayerNorm(channel),
#                 nn.ReLU(),
#                 nn.Dropout(0.1)
#             ).to(device)
#             expanded_part = expand_layer(part) * weights[:, :, i, :]
#             expanded_parts.append(expanded_part)
#
#         return torch.cat(expanded_parts, dim=-1)

"""
版本三
"""
#
# class GatedPartAttention(nn.Module):
#     def __init__(self, num_parts=4, in_channels=64, merge=True):
#         super().__init__()
#         self.num_parts = num_parts
#         self.merge = merge
#
#         # 特征转换层，用于生成门控信息
#         self.feature_transform = nn.Sequential(
#             nn.Linear(in_channels, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
#
#         # 为每个部分创建独立的门控单元
#         self.gate_networks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(128, 64),
#                 nn.ReLU(),
#                 nn.Linear(64, 1),
#                 nn.Sigmoid()
#             ) for _ in range(num_parts)
#         ])
#
#         # 注意力生成器，用于生成额外的注意力权重
#         self.attention_net = nn.Sequential(
#             nn.Linear(128, num_parts),
#             nn.Softmax(dim=-1)
#         )
#
#         # 根据merge参数初始化基础权重
#         if merge:
#             # merge=True时，底部权重更大
#             base_weights = torch.FloatTensor([1.0, 1.0, 1.0, 2.0])
#         else:
#             # merge=False时，权重相等
#             base_weights = torch.FloatTensor([1.0] * num_parts)
#         self.register_parameter('base_weights',
#                                 nn.Parameter(base_weights))
#
#     def forward(self, features):
#         """
#         Args:
#             features: [B, K, C]
#         Returns:
#             weights: [B, K, num_parts]
#         """
#         B, K, C = features.shape
#
#         # 1. 特征转换
#         transformed_features = self.feature_transform(features)  # [B, K, 128]
#
#         # 2. 生成门控信号
#         gates = []
#         for gate_net in self.gate_networks:
#             # 为每个部分生成门控值
#             gate = gate_net(transformed_features)  # [B, K, 1]
#             gates.append(gate)
#         gates = torch.cat(gates, dim=-1)  # [B, K, num_parts]
#
#         # 3. 生成注意力权重
#         attention_weights = self.attention_net(transformed_features)  # [B, K, num_parts]
#
#         # 4. 结合基础权重、门控信号和注意力权重
#         base_weights = F.softmax(self.base_weights, dim=0)  # [num_parts]
#
#         # 扩展基础权重维度以进行广播
#         base_weights = base_weights.view(1, 1, -1).expand(B, K, -1)
#
#         # 组合所有权重
#         final_weights = base_weights * gates * attention_weights
#
#         # 归一化
#         final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-6)
#
#         return final_weights
#
#
# def pipeline_node_feature(node_feature, index_group, num_parts=4, merge=True):
#     """
#     优化后的pipeline版本，使用index_group直接获取各个部分的特征
#
#     Args:
#         node_feature: 输入特征 [B, C, H, W] or [B, C, N]
#         index_group: 包含4个tensor的元组，每个tensor包含对应部分的索引
#         use_transform: 是否需要转换特征维度
#         num_parts: 部分数量，默认为4
#         merge: 是否合并特征
#     """
#     device = node_feature.device
#     batch, channel = node_feature.shape[:2]
#     node_feature = node_feature.view(batch, channel, -1).permute((0, 2, 1)).contiguous()
#
#     # 1. 收集每个部分的特征
#     part_features = []
#     for part_idx in index_group:
#         # 扩展索引维度以匹配特征维度
#         dim = node_feature.size(-1)
#         expanded_idx = part_idx.unsqueeze(-1).expand(*part_idx.shape, dim)
#
#         # 收集特征
#         part_feat = node_feature.gather(dim=1, index=expanded_idx)  # [B, K, C]
#         part_features.append(part_feat)
#
#     # 2. 使用门控注意力生成权重
#     gated_attention = GatedPartAttention(
#         num_parts=num_parts,
#         in_channels=dim,
#         merge=merge
#     ).to(device)
#
#     # 使用第一个部分的特征来生成权重（或者可以使用所有部分的平均值）
#     weights = gated_attention(part_features[0])  # [B, K, num_parts]
#
#     # 3. 堆叠所有部分的特征
#     part_features = torch.stack(part_features, dim=-2)  # [B, K, num_parts, C]
#
#     if merge:
#         # 合并模式
#         weights = weights.unsqueeze(-1)  # [B, K, num_parts, 1]
#         enhanced_features = (part_features * weights).sum(dim=-2)  # [B, K, C]
#         return enhanced_features
#     else:
#         # 扩展模式
#         B, K, P, C = part_features.shape
#         weights = weights.unsqueeze(-1)  # [B, K, num_parts, 1]
#
#         # 对每个部分的特征进行加权和扩展
#         expanded_parts = []
#         for i in range(num_parts):
#             # 提取当前部分的特征
#             part = part_features[:, :, i, :]  # [B, K, C]
#
#             # 创建特征扩展层
#             expand_layer = nn.Sequential(
#                 nn.Linear(C, C),
#                 nn.LayerNorm(C),
#                 nn.ReLU(),
#                 nn.Dropout(0.1)
#             ).to(device)
#
#             # 扩展特征
#             expanded_part = expand_layer(part) * weights[:, :, i, :]
#             expanded_parts.append(expanded_part)
#
#         # 拼接所有部分的特征
#         concatenated_features = torch.cat(expanded_parts, dim=-1)  # [B, K, num_parts*C]
#         return concatenated_features
"""
版本四
"""

class GRUBasedFeatureFusion(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.feature_dim = feature_dim

        # 初始权重 [中心特征, 上方特征, 下方特征, 最下方特征]
        initial_weights = torch.FloatTensor([1.0, 0.0, 0.0, 1.0])
        self.register_parameter('base_weights',
                                nn.Parameter(initial_weights))

        # GRU门控机制
        self.update_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        self.reset_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

        self.transform = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh()
        )

    def forward(self, features_group):
        """
        Args:
            features_group: List of 4 tensors, each [B, K, feature_dim]
        Returns:
            enhanced_features: [B, K, feature_dim]
        """
        # 1. 基础加权
        base_weights = F.softmax(self.base_weights, dim=0)
        weighted_features = []
        for feat, weight in zip(features_group, base_weights):
            weighted_features.append(feat * weight)

        # 2. 序列化GRU融合
        current_state = weighted_features[0]  # 以中心特征为初始状态

        for idx in range(1, len(features_group)):
            input_feature = weighted_features[idx]

            # 特征拼接
            combined = torch.cat([input_feature, current_state], dim=-1)

            # 更新门和重置门
            update = self.update_gate(combined)
            reset = self.reset_gate(combined)

            # 重置门作用于当前状态
            reset_state = reset * current_state

            # 生成候选状态
            candidate = self.transform(
                torch.cat([input_feature, reset_state], dim=-1)
            )

            # 更新状态
            current_state = update * current_state + (1 - update) * candidate

        return current_state


def pipeline_node_feature(node_feature, index_group ,merge=True):
    """
    使用GRU门控机制的pipeline版本
    Args:
        node_feature: 输入特征 [B, C, H, W] or [B, C, N]
        index_group: 包含4个tensor的元组，每个tensor包含对应部分的索引
        use_transform: 是否需要转换特征维度
    Returns:
        enhanced_features: [B, K, C]
    """
    device = node_feature.device

    # 1. 特征维度变换（如果需要）
    batch, channel = node_feature.shape[:2]
    node_feature = node_feature.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    # 2. 收集各部分特征
    features_group = []
    for part_idx in index_group:
        # 扩展索引维度以匹配特征维度
        dim = node_feature.size(-1)
        expanded_idx = part_idx.unsqueeze(-1).expand(*part_idx.shape, dim)

        # 收集特征
        part_feat = node_feature.gather(dim=1, index=expanded_idx)  # [B, K, C]
        features_group.append(part_feat)

    if merge :
        # 3. 使用GRU门控机制融合特征
        fusion_module = GRUBasedFeatureFusion(feature_dim=dim).to(device)
        enhanced_features = fusion_module(features_group)  # [B, K, C]
        return enhanced_features
    else :
        return torch.cat(features_group, dim=2)  #list2tensor