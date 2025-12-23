import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class WeightAnalyzer:
    def __init__(self, num_parts=4):
        self.num_parts = num_parts
        self.weight_stats = defaultdict(list)

    def collect_weights(self, base_weights, gate_values, attention_weights, final_weights):
        """
        收集各个组件的权重统计信息
        Args:
            base_weights: [num_parts]
            gate_values: [B, K, num_parts]
            attention_weights: [B, K, num_parts]
            final_weights: [B, K, num_parts]
        """
        stats = {
            'base_weights': base_weights.detach().cpu(),
            'gate_mean': gate_values.mean(dim=[0 ,1]).detach().cpu(),
            'gate_std': gate_values.std(dim=[0 ,1]).detach().cpu(),
            'attention_mean': attention_weights.mean(dim=[0 ,1]).detach().cpu(),
            'attention_std': attention_weights.std(dim=[0 ,1]).detach().cpu(),
            'final_mean': final_weights.mean(dim=[0 ,1]).detach().cpu(),
            'final_std': final_weights.std(dim=[0 ,1]).detach().cpu()
        }

        # 收集每个部分的统计信息
        for key, value in stats.items():
            self.weight_stats[key].append(value)

    def analyze_part_importance(self, final_weights):
        """分析不同部分的重要性分布"""
        # 计算每个部分的平均权重
        part_importance = final_weights.mean(dim=[0 ,1]).detach().cpu()
        return part_importance

    def visualize_weights(self, save_path=None):
        """可视化权重分布"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 基础权重分布
        base_weights = torch.stack(self.weight_stats['base_weights']).mean(0)
        axes[0 ,0].bar(range(self.num_parts), base_weights)
        axes[0 ,0].set_title('Base Weights Distribution')
        axes[0 ,0].set_ylabel('Weight Value')
        axes[0 ,0].set_xlabel('Part Index')

        # 2. 门控值分布
        gate_means = torch.stack(self.weight_stats['gate_mean']).mean(0)
        gate_stds = torch.stack(self.weight_stats['gate_std']).mean(0)
        axes[0 ,1].bar(range(self.num_parts), gate_means,
                      yerr=gate_stds, capsize=5)
        axes[0 ,1].set_title('Gate Values Distribution')
        axes[0 ,1].set_ylabel('Gate Value')
        axes[0 ,1].set_xlabel('Part Index')

        # 3. 注意力权重分布
        att_means = torch.stack(self.weight_stats['attention_mean']).mean(0)
        att_stds = torch.stack(self.weight_stats['attention_std']).mean(0)
        axes[1 ,0].bar(range(self.num_parts), att_means,
                      yerr=att_stds, capsize=5)
        axes[1 ,0].set_title('Attention Weights Distribution')
        axes[1 ,0].set_ylabel('Attention Weight')
        axes[1 ,0].set_xlabel('Part Index')

        # 4. 最终权重分布
        final_means = torch.stack(self.weight_stats['final_mean']).mean(0)
        final_stds = torch.stack(self.weight_stats['final_std']).mean(0)
        axes[1 ,1].bar(range(self.num_parts), final_means,
                      yerr=final_stds, capsize=5)
        axes[1 ,1].set_title('Final Weights Distribution')
        axes[1 ,1].set_ylabel('Weight Value')
        axes[1 ,1].set_xlabel('Part Index')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def compute_weight_correlations(self):
        """计算不同权重组件之间的相关性"""
        correlations = defaultdict(float)

        # 计算各组件间的相关系数
        components = ['base_weights', 'gate_mean', 'attention_mean', 'final_mean']
        for i, comp1 in enumerate(components):
            for comp2 in components[ i +1:]:
                corr = np.corrcoef(
                    torch.stack(self.weight_stats[comp1]).mean(0).numpy(),
                    torch.stack(self.weight_stats[comp2]).mean(0).numpy()
                )[0 ,1]
                correlations[f"{comp1}_vs_{comp2}"] = corr

        return correlations

    def analyze_temporal_patterns(self):
        """分析权重随时间的变化模式"""
        temporal_stats = {}
        for key in ['gate_mean', 'attention_mean', 'final_mean']:
            values = torch.stack(self.weight_stats[key])
            temporal_stats[key] = {
                'trend': values.mean(dim=0),  # 总体趋势
                'variance': values.var(dim=0),  # 变化程度
                'max_change': (values[1:] - values[:-1]).abs().max(dim=0)[0]  # 最大变化
            }
        return temporal_stats

# 修改GatedPartAttention类以收集权重信息
class GatedPartAttention(nn.Module):
    def __init__(self, num_parts=4, in_channels=64, merge=True):
        super().__init__()
        self.weight_analyzer = WeightAnalyzer(num_parts)
        # ... 其他初始化代码保持不变 ...

    def forward(self, features):
        # ... 原有的前向传播代码 ...

        # 收集权重信息
        self.weight_analyzer.collect_weights(
            self.base_weights,
            gates,
            attention_weights,
            final_weights
        )

        return final_weights

    def get_weight_analysis(self):
        """获取权重分析结果"""
        analysis = {
            'correlations': self.weight_analyzer.compute_weight_correlations(),
            'temporal_patterns': self.weight_analyzer.analyze_temporal_patterns(),
        }
        return analysis

# 使用示例
def visualize_weight_distribution(model, save_dir='./weight_analysis'):
    """
    可视化模型权重分布
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # 获取权重分析
    weight_analyzer = model.gated_attention.weight_analyzer

    # 1. 可视化权重分布
    weight_analyzer.visualize_weights(
        save_path=os.path.join(save_dir, 'weight_distribution.png')
    )

    # 2. 分析组件相关性
    correlations = weight_analyzer.compute_weight_correlations()
    print("\nWeight Component Correlations:")
    for key, value in correlations.items():
        print(f"{key}: {value:.3f}")

    # 3. 分析时序模式
    temporal_patterns = weight_analyzer.analyze_temporal_patterns()
    print("\nTemporal Patterns Analysis:")
    for key, stats in temporal_patterns.items():
        print(f"\n{key}:")
        print(f"Mean trend: {stats['trend'].mean().item():.3f}")
        print(f"Mean variance: {stats['variance'].mean().item():.3f}")
        print(f"Max change: {stats['max_change'].mean().item():.3f}")
