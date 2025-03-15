import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patheffects as path_effects

# 创建保存目录
os.makedirs("task3_MNIST/comparison_images", exist_ok=True)

# 性能数据
performance_data = {
    "MLP": {
        "Training Time": 35.37,
        "Training Loss": 0.0029,
        "Validation Accuracy": 97.98,
        "Test Accuracy": 98.10,
        "Model Size(MB)": 2.0
    },
    "CNN": {
        "Training Time": 83.70,
        "Training Loss": 0.0141,
        "Validation Accuracy": 98.68,
        "Test Accuracy": 98.83,
        "Model Size(MB)": 0.177
    }
}

# 设置风格
sns.set(style="darkgrid")
plt.style.use('dark_background')

# 创建图1：性能雷达图
def plot_radar_chart():
    # 准备数据
    categories = ['Training Time', 'Training Loss', 'Validation Accuracy', 'Test Accuracy']
    
    # 归一化数据（越小越好的指标取倒数）
    normalized_data = {
        "MLP": [
            1/performance_data["MLP"]["Training Time"] * 100,  # 训练时间（越小越好，取倒数）
            1/performance_data["MLP"]["Training Loss"] * 0.1,  # 训练损失（越小越好，取倒数）
            performance_data["MLP"]["Validation Accuracy"],    # 验证准确率
            performance_data["MLP"]["Test Accuracy"]           # 测试准确率
        ],
        "CNN": [
            1/performance_data["CNN"]["Training Time"] * 100,  # 训练时间（越小越好，取倒数）
            1/performance_data["CNN"]["Training Loss"] * 0.1,  # 训练损失（越小越好，取倒数）
            performance_data["CNN"]["Validation Accuracy"],    # 验证准确率
            performance_data["CNN"]["Test Accuracy"]           # 测试准确率
        ]
    }
    
    # 计算角度
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图
    
    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    # 添加每个类别的标签
    plt.xticks(angles[:-1], categories, color='white', size=12)
    
    # 绘制MLP数据
    values = normalized_data["MLP"]
    values += values[:1]  # 闭合雷达图
    ax.plot(angles, values, 'o-', linewidth=2, label='MLP', color='#FF9671')
    ax.fill(angles, values, alpha=0.25, color='#FF9671')
    
    # 绘制CNN数据
    values = normalized_data["CNN"]
    values += values[:1]  # 闭合雷达图
    ax.plot(angles, values, 'o-', linewidth=2, label='CNN', color='#00D2FC')
    ax.fill(angles, values, alpha=0.25, color='#00D2FC')
    
    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # 设置标题
    plt.title('MLP vs CNN Performance Radar Chart', size=20, color='white', y=1.1)
    
    # 添加网格线
    ax.grid(True, color='gray', linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('task3_MNIST/comparison_images/radar_chart.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

# 创建图2：条形图比较
def plot_bar_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    metrics = ['Training Time', 'Training Loss', 'Test Accuracy', 'Model Size(MB)']
    colors = {'MLP': '#FF9671', 'CNN': '#00D2FC'}
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = [performance_data['MLP'][metric], performance_data['CNN'][metric]]
        
        # 创建条形图
        bars = ax.bar(['MLP', 'CNN'], data, color=[colors['MLP'], colors['CNN']], alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(data),
                   f'{height:.4f}' if height < 0.1 else f'{height:.2f}',
                   ha='center', va='bottom', fontsize=12, color='white',
                   path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
        
        # 设置标题和标签
        ax.set_title(f'{metric} Comparison', fontsize=14)
        ax.set_ylabel(metric, fontsize=12)
        
        # 设置网格线
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # 设置背景色
        ax.set_facecolor('#1E1E1E')
    
    plt.suptitle('MLP vs CNN Performance Metrics', fontsize=20, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('task3_MNIST/comparison_images/bar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 创建图3：训练曲线对比
def plot_training_curves_comparison():
    # 加载训练历史数据
    try:
        # 尝试加载真实的训练历史数据
        mlp_losses = np.load('task3_MNIST/mlp_train_losses.npy')
        mlp_accuracies = np.load('task3_MNIST/mlp_val_accuracies.npy')
        cnn_losses = np.load('task3_MNIST/cnn_train_losses.npy')
        cnn_accuracies = np.load('task3_MNIST/cnn_val_accuracies.npy')
    except:
        # 如果没有真实数据，使用模拟数据
        epochs = 10
        mlp_losses = np.array([0.3278, 0.1110, 0.0680, 0.0470, 0.0322, 0.0218, 0.0139, 0.0095, 0.0056, 0.0029])
        mlp_accuracies = np.array([0.9505, 0.9647, 0.9722, 0.9758, 0.9758, 0.9779, 0.9792, 0.9781, 0.9783, 0.9798])
        cnn_losses = np.array([0.4086, 0.0770, 0.0535, 0.0412, 0.0323, 0.0262, 0.0213, 0.0182, 0.0185, 0.0141])
        cnn_accuracies = np.array([0.9687, 0.9764, 0.9827, 0.9838, 0.9832, 0.9809, 0.9855, 0.9860, 0.9838, 0.9868])
    
    # 创建图形
    fig = plt.figure(figsize=(15, 7))
    gs = GridSpec(1, 2, figure=fig)
    
    # 损失曲线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(mlp_losses, 'o-', color='#FF9671', linewidth=2, markersize=8, label='MLP')
    ax1.plot(cnn_losses, 's-', color='#00D2FC', linewidth=2, markersize=8, label='CNN')
    ax1.set_title('Training Loss Comparison', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    # 准确率曲线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(mlp_accuracies * 100, 'o-', color='#FF9671', linewidth=2, markersize=8, label='MLP')
    ax2.plot(cnn_accuracies * 100, 's-', color='#00D2FC', linewidth=2, markersize=8, label='CNN')
    ax2.set_title('Validation Accuracy Comparison', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    
    # 设置y轴范围，使差异更明显
    ax2.set_ylim([95, 100])
    
    # 添加总标题
    plt.suptitle('MLP vs CNN Training Process', fontsize=20, y=0.98)
    
    # 保存图像
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('task3_MNIST/comparison_images/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

# 创建图4：模型架构可视化对比
def plot_architecture_comparison():
    fig = plt.figure(figsize=(15, 8))
    
    # 设置背景色
    fig.patch.set_facecolor('#1E1E1E')
    
    # MLP架构
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_facecolor('#1E1E1E')
    
    # 绘制MLP层
    layer_positions = [0, 1, 2, 3]
    layer_sizes = [784, 512, 256, 10]
    layer_colors = ['#FF9671', '#FFC75F', '#F9F871', '#FF9671']
    layer_labels = ['Input\n(784)', 'Hidden 1\n(512)', 'Hidden 2\n(256)', 'Output\n(10)']
    
    max_size = max(layer_sizes)
    normalized_sizes = [size/max_size for size in layer_sizes]
    
    for i, (pos, size, color, label) in enumerate(zip(layer_positions, normalized_sizes, layer_colors, layer_labels)):
        circle = plt.Circle((pos, 0), size/2, color=color, alpha=0.7)
        ax1.add_patch(circle)
        ax1.text(pos, 0, label, ha='center', va='center', fontsize=12, color='black', fontweight='bold')
        
        # 添加连接线
        if i < len(layer_positions) - 1:
            ax1.plot([pos, layer_positions[i+1]], [0, 0], 'w--', alpha=0.5)
    
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(-0.6, 0.6)
    ax1.set_title('MLP Architecture', fontsize=16, color='white')
    ax1.axis('off')
    
    # CNN架构
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_facecolor('#1E1E1E')
    
    # 绘制CNN层
    cnn_layers = [
        {'name': 'Input\n28×28×1', 'pos': 0, 'height': 0.8, 'width': 0.6, 'color': '#00D2FC'},
        {'name': 'Conv1\n24×24×6', 'pos': 1, 'height': 0.7, 'width': 0.6, 'color': '#00BFAD'},
        {'name': 'Pool1\n12×12×6', 'pos': 2, 'height': 0.6, 'width': 0.6, 'color': '#009EFA'},
        {'name': 'Conv2\n8×8×16', 'pos': 3, 'height': 0.5, 'width': 0.6, 'color': '#00BFAD'},
        {'name': 'Pool2\n4×4×16', 'pos': 4, 'height': 0.4, 'width': 0.6, 'color': '#009EFA'},
        {'name': 'FC1\n120', 'pos': 5, 'height': 0.3, 'width': 0.6, 'color': '#5E17EB'},
        {'name': 'FC2\n84', 'pos': 6, 'height': 0.2, 'width': 0.6, 'color': '#5E17EB'},
        {'name': 'Output\n10', 'pos': 7, 'height': 0.15, 'width': 0.6, 'color': '#00D2FC'}
    ]
    
    for i, layer in enumerate(cnn_layers):
        rect = Rectangle((layer['pos'], -layer['height']/2), layer['width'], layer['height'], 
                         color=layer['color'], alpha=0.7)
        ax2.add_patch(rect)
        ax2.text(layer['pos'] + layer['width']/2, 0, layer['name'], 
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        
        # 添加连接线
        if i < len(cnn_layers) - 1:
            ax2.plot([layer['pos'] + layer['width'], cnn_layers[i+1]['pos']], 
                    [0, 0], 'w--', alpha=0.5)
    
    ax2.set_xlim(-0.5, 8)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_title('CNN Architecture', fontsize=16, color='white')
    ax2.axis('off')
    
    plt.suptitle('MLP vs CNN Architecture Comparison', fontsize=20, color='white', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('task3_MNIST/comparison_images/architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 执行所有可视化
if __name__ == "__main__":
    print("Generating performance radar chart...")
    plot_radar_chart()
    
    print("Generating bar comparison chart...")
    plot_bar_comparison()
    
    print("Generating training curves comparison...")
    plot_training_curves_comparison()
    
    print("Generating architecture comparison...")
    plot_architecture_comparison()
    
    print("All visualization images have been saved to task3_MNIST/comparison_images/ directory") 