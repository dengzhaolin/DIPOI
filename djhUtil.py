import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict
import Constants as C

def build_inter_graph(dataName, device,sep='\s+'):
    """
    构建地理图函数
    :param dataName: 输入数据
    :param n_poi_total: 已知的POI总数（包含训练集未出现的POI）
    :param sep: 数据分隔符，默认为空白符
    :return: 构建完成的Data对象
    """
    # 读取数据 ================================================================
    df = pd.read_csv(f"data/{dataName}/{dataName}_test.txt", sep=sep, header=None,
                     names=['user_id', 'poi_id', 'type'])
    n_poi_total=C.poi_dict.get(dataName)

    # 数据验证 ================================================================
    # 检查是否存在超过已知最大ID的POI
    max_in_data = df['poi_id'].max()
    if max_in_data >= n_poi_total:
        raise ValueError(f"数据包含非法POI ID {max_in_data}，超过已知最大ID {n_poi_total-1}")

    # 构建边关系 =============================================================
    user_sequences = df.groupby('user_id')['poi_id'].apply(list).tolist()

    edge_counter = defaultdict(int)
    for seq in user_sequences:
        for i in range(len(seq)-1):
            src, dst = seq[i], seq[i+1]
            if src == dst:
                continue  # 跳过自环边
            # 无向边统一表示
            u, v = (src, dst) if src < dst else (dst, src)
            edge_counter[(u, v)] += 1

    # 转换为PyG格式 ==========================================================
    # 生成双向边（无向图）
    geo_edges = []
    edge_weights = []
    for (u, v), cnt in edge_counter.items():
        geo_edges.extend([[u, v], [v, u]])  # 显式添加双向边
        edge_weights.extend([cnt, cnt])

    # 转换为Tensor
    if len(geo_edges) == 0:
        # 处理无边情况（创建空Tensor）
        geo_edges = torch.LongTensor(size=(2, 0))
        edge_weights = torch.FloatTensor([])
    else:
        geo_edges = torch.LongTensor(geo_edges).t().contiguous()  # [2, E]
        edge_weights = torch.FloatTensor(edge_weights)  # [E]

    # 权重归一化 =============================================================
    if edge_weights.numel() > 0:
        max_weight = edge_weights.max()
        edge_weights = edge_weights / max_weight if max_weight > 0 else edge_weights
    else:
        print("警告：图中没有有效边")

    # 构建图对象 =============================================================
    graph_data = Data(
        edge_index=geo_edges.to(C.DEVICE),      # 显式指定设备
        edge_attr=edge_weights.to(C.DEVICE),  # 同步设备
        num_nodes=n_poi_total
    )
    print(f"\n交互图构建结果：data/{dataName}/{dataName}_test.txt")
    print(f"总节点数: {graph_data.num_nodes}（包含所有POI）")
    print(f"有效边数量: {graph_data.num_edges}")
    if graph_data.num_edges > 0:
        print(f"边权重范围: [{graph_data.edge_attr.min():.4f}, {graph_data.edge_attr.max():.4f}]")
    else:
        print("边权重：无有效边")

    return graph_data

# 使用示例 ===================================================================
if __name__ == "__main__":

    # 假设已知总共有1000个POI（ID 0-999）
    geo_graph = build_inter_graph("Yelp2018",C.DEVICE)
