import matplotlib.pyplot as plt
import numpy as np

# 配置全局字体和样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.0

# 创建图形
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# 定义参数值和评估指标
params = ['0.2', '0.5', '0.8']  # $d_m$参数值

# 精确颜色和标记映射
colors = {
    'Recall@5': 'black',
    'Recall@10': 'red',
    'Recall@20': 'blue',
}

colors1 = {
    'NDCG@5': 'green',
    'NDCG@10': 'purple',
    'NDCG@20': '#FFD700'  # 金色替代黄色
}

markers = {
    'Recall@5': 's',  # 方形
    'Recall@10': 'o',  # 圆形
    'Recall@20': '^',  # 上三角形
}

markers1 = {
    'NDCG@5': 'v',  # 下三角形
    'NDCG@10': 'D',  # 菱形
    'NDCG@20': '*'  # 星形
}

# 根据图中文本创建数据
metrics = {
    'Recall@5': [0.0294, 0.0310, 0.0273],
    'Recall@10': [0.0467, 0.0486, 0.0455],
    'Recall@20': [0.0761, 0.0770, 0.0744]
}

metrics1 = {
    'NDCG@5': [0.0523, 0.0540, 0.0490],
    'NDCG@10': [0.0546, 0.0552, 0.0510],
    'NDCG@20': [0.0643, 0.0672, 0.0621]
}

# 为每个指标绘制折线
for metric in metrics:
    values = metrics[metric]
    ax.plot(params, values,
            color=colors[metric],
            marker=markers[metric],
            markersize=7,
            markeredgewidth=1,
            markerfacecolor='white',
            linewidth=2,
            label=metric)

for metric in metrics1:
    values = metrics1[metric]
    ax.plot(params, values,
            color=colors1[metric],
            marker=markers1[metric],
            markersize=7,
            markeredgewidth=1,
            markerfacecolor='white',
            linewidth=2,
            label=metric)

# 设置轴范围和网格
ax.set_ylim(0.02, 0.08)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)  # 网格置于数据下方

# 设置标题和标签
ax.set_xlabel('Yelp ($μ$)', fontsize=12, labelpad=10)

# 关键修改：创建分行的图例系统
# 第一行：Recall指标
recall_lines = [plt.Line2D([0], [0],
                           color=colors[metric],
                           marker=markers[metric],
                           markeredgewidth=1,
                           markerfacecolor='white',
                           markersize=7) for metric in metrics]

# 第二行：NDCG指标
ndcg_lines = [plt.Line2D([0], [0],
                         color=colors1[metric],
                         marker=markers1[metric],
                         markeredgewidth=1,
                         markerfacecolor='white',
                         markersize=7) for metric in metrics1]

# 创建图例（分为两行）
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

# 创建两行图例内容
recall_legend = ax.legend(recall_lines, list(metrics.keys()),
                          loc='upper center',
                          frameon=False,
                          ncol=3,
                          bbox_to_anchor=(0.5, 1.24))

ndcg_legend = ax.legend(ndcg_lines, list(metrics1.keys()),
                        loc='upper center',
                        frameon=False,
                        ncol=3,
                        bbox_to_anchor=(0.5, 1.13))

# 将图例添加到图形中
ax.add_artist(recall_legend)
ax.add_artist(ndcg_legend)




# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # 为顶部图例留空间

# 显示和保存图像
plt.savefig('foursquare_dm_performance.png', dpi=300, bbox_inches='tight')
plt.show()