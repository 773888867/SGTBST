def test_weight_distribution(model, val_loader, save_dir='./weight_analysis'):
    """
    测试并可视化模型在验证集上的权重分布
    Args:
        model: 完整的MOT模型
        val_loader: 验证集数据加载器
        save_dir: 保存可视化结果的目录
    """
    model.eval()

    with torch.no_grad():
        # 遍历验证集
        for batch_idx, batch_data in enumerate(tqdm(val_loader, desc="Analyzing weights")):
            # 前向传播
            outputs = model(batch_data)

            # 每处理10个batch记录一次权重分布
            if batch_idx % 10 == 0:
                # 获取权重分析结果
                weight_analysis = model.gated_attention.get_weight_analysis()

                # 打印相关性分析结果
                print(f"\nBatch {batch_idx} - Weight Component Correlations:")
                for key, value in weight_analysis['correlations'].items():
                    print(f"{key}: {value:.3f}")

                # 打印时序模式分析结果
                print(f"\nBatch {batch_idx} - Temporal Patterns:")
                for key, stats in weight_analysis['temporal_patterns'].items():
                    print(f"\n{key}:")
                    print(f"Mean trend: {stats['trend'].mean().item():.3f}")
                    print(f"Mean variance: {stats['variance'].mean().item():.3f}")
                    print(f"Max change: {stats['max_change'].mean().item():.3f}")

    # 测试结束后生成完整的可视化结果
    visualize_final_results(model, save_dir)


def visualize_final_results(model, save_dir):
    """
    生成最终的可视化结果
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    weight_analyzer = model.gated_attention.weight_analyzer

    # 1. 生成权重分布图
    weight_analyzer.visualize_weights(
        save_path=os.path.join(save_dir, 'final_weight_distribution.png')
    )

    # 2. 生成时序变化图
    visualize_temporal_changes(weight_analyzer, save_dir)

    # 3. 生成相关性热力图
    visualize_correlation_heatmap(weight_analyzer, save_dir)


def visualize_temporal_changes(weight_analyzer, save_dir):
    """
    可视化权重随时间的变化
    """
    components = ['gate_mean', 'attention_mean', 'final_mean']

    plt.figure(figsize=(15, 5))
    for idx, component in enumerate(components):
        values = torch.stack(weight_analyzer.weight_stats[component])
        plt.subplot(1, 3, idx + 1)
        for part in range(weight_analyzer.num_parts):
            plt.plot(values[:, part].numpy(), label=f'Part {part}')
        plt.title(f'{component} Temporal Changes')
        plt.xlabel('Time Step')
        plt.ylabel('Weight Value')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'temporal_changes.png'))
    plt.close()


def visualize_correlation_heatmap(weight_analyzer, save_dir):
    """
    生成权重组件间的相关性热力图
    """
    import seaborn as sns

    components = ['base_weights', 'gate_mean', 'attention_mean', 'final_mean']
    corr_matrix = np.zeros((len(components), len(components)))

    # 计算相关性矩阵
    for i, comp1 in enumerate(components):
        for j, comp2 in enumerate(components):
            if i <= j:
                values1 = torch.stack(weight_analyzer.weight_stats[comp1]).mean(0).numpy()
                values2 = torch.stack(weight_analyzer.weight_stats[comp2]).mean(0).numpy()
                corr_matrix[i, j] = np.corrcoef(values1, values2)[0, 1]
                corr_matrix[j, i] = corr_matrix[i, j]

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                xticklabels=components,
                yticklabels=components,
                cmap='coolwarm')
    plt.title('Weight Components Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'))
    plt.close()


# 使用示例
if __name__ == '__main__':
    # 假设您已经有了模型和数据加载器
    model = YourMOTModel()  # 您的MOT模型
    val_loader = YourDataLoader()  # 您的数据加载器

    # 运行测试和可视化
    test_weight_distribution(
        model=model,
        val_loader=val_loader,
        save_dir='./weight_analysis'
    )