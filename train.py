import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import re
from collections import Counter
import torch.cuda.amp as amp
import random
import shutil
from pathlib import Path
import matrix_make
import torch.optim as opt
import torch as T
import os
import pandas as pd
import multiprocessing
import numpy as np
from torch_geometric.nn import GATConv
from torch.nn import Linear
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from multiprocessing import Pool
from tqdm import tqdm
import SDF_dispose
from rdkit import Chem
from rdkit import RDLogger
import test
import math, sys, traceback
import time
from torch_geometric.utils import dropout_edge
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch.utils.data import DataLoader
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import SDF_make
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def precompute_all_graphs(active_list, A, batch_size=256):  
    """针对A100优化的批量预加载函数，同时计算并绑定 ECFP 指纹"""
    supplier = Chem.SDMolSupplier(A+".sdf")
    graphs = []
    listuseful = []
    mol_names = []
    
    # 分批处理分子
    for batch_start in range(0, len(active_list), batch_size):
        batch_end = min(batch_start + batch_size, len(active_list))
        batch_indices = active_list[batch_start:batch_end]
        
        batch_graphs = []
        batch_listuseful = []
        batch_mol_names = []
        
        for idx in batch_indices:
            try:
                mol_name = "未知"
                mol = supplier[idx]
                if mol is None:
                    print(f"{idx} 是无效分子")
                    continue

                # 1. 生成基础的 PyG 图结构特征
                graph = SDF_dispose.molecule_to_pyg_graph(mol)
                
                # ================= 新增：计算并绑定 ECFP 指纹 =================
                # 计算 Morgan 指纹 (radius=2 相当于 ECFP4, nBits=2048 是标准维度)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fp_array = np.zeros((1,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, fp_array)
                
                # 将 numpy 数组转为 Tensor，并挂载到图对象的 .ecfp 属性上
                # 调整形状为 [1, 2048]，这样后续 PyG 组装 Batch 时可以直接竖向拼接
                graph.ecfp = torch.tensor(fp_array, dtype=torch.float32).view(1, -1)
                # ===============================================================

                if mol.HasProp("name"):
                    mol_name = mol.GetProp("name")
                    if int(mol_name.replace('name', '')) != idx:
                        print(f'{idx}错位')
                    else:
                        batch_mol_names.append(mol_name)

                batch_graphs.append(graph)
                batch_listuseful.append(idx)

            except Exception as e:
                print(f"处理分子 {idx} 时出错: {e}")
                continue
        
        # 批量转移到GPU - A100可以并行处理更多 (你之前的注释保留了)
        for graph in batch_graphs:
            graph.x = graph.x#.to(device, non_blocking=True)
            graph.edge_index = graph.edge_index#.to(device, non_blocking=True)
        
        graphs.extend(batch_graphs)
        listuseful.extend(batch_listuseful)
        mol_names.extend(batch_mol_names)

    return graphs, listuseful, mol_names

# 简单的数据保存
def save_vector_to_txt(vector, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in vector:
            file.write(f"{item}\n")

def extract_digits(s):
    # 把开头的所有非数字字符替换为空
    return int(re.sub(r'^\D+', '', s))

class GNN(nn.Module):
    def __init__(self, INPUT):
        super(GNN, self).__init__()
        # Building INPUT
        self.INPUT = INPUT
        # Defining base variables
        self.CHECKPOINT_DIR = self.INPUT["CHECKPOINT_DIR"]
        self.CHECKPOINT_FILE = os.path.join(self.CHECKPOINT_DIR, self.INPUT["NAME"])
        self.SIZE_LAYERS = self.INPUT["SIZE_LAYERS"]
        self.initial_conv = GATConv(self.SIZE_LAYERS[0], self.SIZE_LAYERS[1])
        self.conv1 = GATConv(self.SIZE_LAYERS[1], self.SIZE_LAYERS[2])
        self.conv2 = GATConv(self.SIZE_LAYERS[2], self.SIZE_LAYERS[2])
        self.linear = Linear(self.SIZE_LAYERS[2], self.SIZE_LAYERS[3])
        self.optimizer = opt.Adam(self.parameters(), lr=self.INPUT["LR"])
        self.criterion = nn.MSELoss()

    # def forward(self, x, edge_index):  # forward propagation includes defining layers
    #     out = F.relu(self.initial_conv(x, edge_index=edge_index))
    #     out = F.relu(self.conv1(out, edge_index=edge_index))
    #     out = F.relu(self.conv2(out, edge_index=edge_index))
    #     return self.linear(out)
    def forward(self, x, edge_index, drop_rate=0.0):  
        # 如果 drop_rate 大于 0 且模型处于训练模式，则随机丢弃边（图扰动）
        if drop_rate > 0.0 and self.training:
            edge_index, _ = dropout_edge(edge_index, p=drop_rate,force_undirected=True)
            # 2. 原子掩蔽 (Node Masking)：随机遮住 15% 的原子
            num_nodes = x.size(0)
            mask_indices = torch.rand(num_nodes, device=x.device) < drop_rate
            
            # 必须使用 clone()，防止在原地修改覆盖了原始数据
            x = x.clone()
        out = F.relu(self.initial_conv(x, edge_index=edge_index))
        out = F.relu(self.conv1(out, edge_index=edge_index))
        out = F.relu(self.conv2(out, edge_index=edge_index))
        return self.linear(out)

    def save_checkpoint(self):
        print(f'保存模型到 {self.CHECKPOINT_FILE}...')
        T.save(self.state_dict(), self.CHECKPOINT_FILE)

    def load_checkpoint(self):
        print(f'从 {self.CHECKPOINT_FILE} 加载模型...')
        self.load_state_dict(T.load(self.CHECKPOINT_FILE))

# 分类器网络
class ClassifierNetwork(nn.Module):
    def __init__(self, gnn_dim=32, ecfp_dim=2048, num_classes=2):
        super(ClassifierNetwork, self).__init__()
        
        # 塔 A：ECFP 指纹专属降维通道
        # 负责将 2048 维的宏观特征浓缩为 128 维的高级稠密特征
        self.ecfp_branch = nn.Sequential(
            nn.Linear(ecfp_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 融合后的总维度：图结构特征 (32) + 浓缩指纹特征 (128) = 160
        fusion_dim = gnn_dim + 64
        
        # 融合决策塔：接收 160 维的联合特征进行最终分类
        self.final_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, num_classes)
        )
        
    def forward(self, gnn_emb, ecfp_emb):
        # 1. 指纹特征通过专属网络降维
        ecfp_features = self.ecfp_branch(ecfp_emb)
        
        # 2. 将 GNN 特征与降维后的指纹特征拼接 (维度: 32 + 128 = 160)
        combined_features = torch.cat([gnn_emb, ecfp_features], dim=1)
        
        # 3. 输出分类预测
        return self.final_classifier(combined_features)
#==================================================
class FixedBatchCircleLoss(nn.Module):
    def __init__(self, margin=0.25, gamma=64):  # 降低gamma值
        """
        修正的批量Circle Loss
        """
        super(FixedBatchCircleLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma  # 从256降到64，避免数值爆炸
        self.O_p = 1 + margin
        self.O_n = -margin
        self.delta_p = 1 - margin
        self.delta_n = margin
        
    def forward(self, anchors, positives_list, negatives_list):
        """
        修正的批量计算Circle Loss
        """
        batch_size = anchors.size(0)
        total_loss = 0.0
        valid_count = 0
        
        for i in range(batch_size):
            anchor = anchors[i].unsqueeze(0)  # [1, embed_dim]
            positives = positives_list[i]     # [num_pos, embed_dim]
            negatives = negatives_list[i]     # [num_neg, embed_dim]
            
            if positives.size(0) == 0 or negatives.size(0) == 0:
                continue
                
            # 1. 计算余弦相似度（确保数值稳定）
            anchor_norm = F.normalize(anchor, p=2, dim=1)
            positives_norm = F.normalize(positives, p=2, dim=1)
            negatives_norm = F.normalize(negatives, p=2, dim=1)
            
            # 使用更稳定的矩阵乘法
            s_p = torch.mm(anchor_norm, positives_norm.t()).squeeze(0)  # [num_pos]
            s_n = torch.mm(anchor_norm, negatives_norm.t()).squeeze(0)  # [num_neg]
            
            # 2. 数值稳定性处理
            s_p = torch.clamp(s_p, -1.0 + 1e-7, 1.0 - 1e-7)
            s_n = torch.clamp(s_n, -1.0 + 1e-7, 1.0 - 1e-7)
            
            # 3. 计算自适应权重
            with torch.no_grad():
                alpha_p = torch.clamp(self.O_p - s_p, min=0)  # [num_pos]
                alpha_n = torch.clamp(s_n - self.O_n, min=0)  # [num_neg]
            
            # 4. 计算差距项
            delta_s_p = self.delta_p - s_p  # 正样本差距
            delta_s_n = s_n - self.delta_n  # 负样本差距
            
            # 5. 加权logits（使用更稳定的计算）
            logit_p = -self.gamma * alpha_p * delta_s_p  # [num_pos]
            logit_n = self.gamma * alpha_n * delta_s_n   # [num_neg]
            
            # 6. 稳定化的损失计算
            # 分别计算正负样本的logsumexp
            logsumexp_p = torch.logsumexp(logit_p, dim=0)
            logsumexp_n = torch.logsumexp(logit_n, dim=0)
            
            # 最终损失
            loss = F.softplus(logsumexp_n + logsumexp_p)
            
            total_loss += loss
            valid_count += 1
        
        return total_loss / max(valid_count, 1)
#==================================================

def count_sdf_molecules_accurate(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file)
    valid_count = 0
    for mol in supplier:
        if mol is not None:  # 仅统计有效分子
            valid_count += 1
    return valid_count

def extract_active_property(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file)
    active_data = []
    active_0_name = []
    active_1_name = []

    for idx, mol in enumerate(supplier):
        if mol is None:
            print(f"警告: 第 {idx + 1} 个分子无效，跳过")
            continue
        # else:
            # writer.write(mol)
            # print('保存')

        # 检查是否存在 <Active> 属性
        if mol.HasProp("Active"):
            if mol.HasProp("name"):
                name = mol.GetProp("name")
                active = mol.GetProp("Active")
                active_data.append(active)
                if active == '0':
                    active_0_name.append(name)
                elif active == '1':
                    active_1_name.append(name)

        else:
            print(f"第 {idx + 1} 个分子缺少 <Active> 属性")
            active_data.append(None)  # 或默认值
    #返回值分别为总的ACTIVE属性列表，格式为分子序号+属性，他们对应的序号
    return active_data,active_0_name,active_1_name

#==================================================
def load_and_validate_gesim_csv(csv_path, expected_rows, expected_cols):
    """
    加载并验证GESim CSV文件
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
        
        # 验证维度
        if df.shape[0] != expected_rows or df.shape[1] != expected_cols:
            print(f"CSV维度不匹配: 期望({expected_rows}, {expected_cols}), 实际{df.shape}")
            return None
            
        # 验证数值范围
        if df.min().min() < 0 or df.max().max() > 1:
            print("CSV包含超出[0,1]范围的相似度值")
            return None
            
        print(f"成功加载GESim CSV: {csv_path}, 形状: {df.shape}")
        return df
        
    except Exception as e:
        print(f"加载GESim CSV失败: {e}")
        return None

def get_gesim_hard_samples(gesim_df, anchor_idx, candidate_indices, k, device):
    """
    从GESim CSV中获取困难样本
    """
    if anchor_idx not in gesim_df.index:
        return []
    
    # 获取相似度分数
    similarities = []
    for idx in candidate_indices:
        if str(idx) in gesim_df.columns:
            sim = gesim_df.loc[anchor_idx, str(idx)]
            similarities.append((idx, sim))
    
    # 按相似度排序并选择前k个
    similarities.sort(key=lambda x: x[1], reverse=True)
    selected = [idx for idx, sim in similarities[:k]]
    
    return selected
#==================================================
# def evaluate_test_accuracy_fixed(encoder, classifier, test_active_values, test_file_path, device):
#     """修正的测试集准确率评估函数"""
#     encoder.eval()
#     classifier.eval()
    
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         # 预加载所有测试分子的图数据
#         all_test_indices = list(range(len(test_active_values)))
#         all_graphs, valid_indices, _ = precompute_all_graphs(all_test_indices, test_file_path)
        
#         if len(all_graphs) == 0:
#             print("无法获取测试分子图数据")
#             return 0.0
        
#         # 批量处理所有测试分子
#         batch = Batch.from_data_list(all_graphs).to(device)
        
#         # 通过GNN编码器
#         h = encoder(batch.x, batch.edge_index)
#         g = global_mean_pool(h, batch.batch)
        
#         # 通过分类器
#         out = classifier(g)
#         _, predicted = torch.max(out.data, 1)
        
#         # 计算准确率
#         for i, mol_idx in enumerate(valid_indices):
#             if mol_idx < len(test_active_values):
#                 predicted_label = str(predicted[i].item())
#                 true_label = test_active_values[mol_idx]
                
#                 total += 1
#                 if predicted_label == true_label:
#                     correct += 1
    
#     accuracy = 100 * correct / total if total > 0 else 0
    
#     print(f"测试集总分子数: {len(test_active_values)}")
#     print(f"成功预测分子数: {total}")
#     print(f"预测正确数: {correct}")
#     print(f"预测准确率: {accuracy:.2f}%")
    
#     return accuracy
#======================================================================================
def evaluate_test_accuracy_fixed(test_graphs_cache, test_valid_cache, encoder, classifier, test_active_values, test_file_path, device, batch_size=256, verbose=False):
    """
    分批评估测试集准确率和 AUC（避免一次性把整个 test set 放到 GPU）。
    batch_size: 每次送到 GPU 的分子数（根据显存调小或调大）。
    """
    import gc
    from sklearn.metrics import roc_auc_score
    import torch.nn.functional as F
    
    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0
    
    # 用于计算 AUC 的列表
    all_trues = []
    all_probs = []

    # 1) 先得到所有测试索引与对应的图（图保存在 CPU 或依据 precompute_all_graphs 实现）
    all_test_indices = list(range(len(test_active_values)))
    
    if len(test_graphs_cache) == 0:
        print("无法获取测试分子图数据")
        return 0.0, 0.5  # 返回两个值：准确率 0.0, AUC 0.5

    # 2) 按 batch_size 分块评估，每块只在 GPU 上存在短时间
    with torch.no_grad():
        for start in range(0, len(test_graphs_cache), batch_size):
            end = min(start + batch_size, len(test_graphs_cache))
            sub_graphs = test_graphs_cache[start:end]

            # 确保 sub_graphs 的张量在 CPU，Batch.from_data_list 会做统一搬运
            batch = Batch.from_data_list(sub_graphs).to(device)

            # encoder 前向并池化
            # 注意：如果之后你在 GNN 里加了 drop_rate，这里需要加上 drop_rate=0.0
            h = encoder(batch.x, batch.edge_index)
            gnn_embeddings = global_mean_pool(h, batch.batch)    # [B, 32]

            # ================= 新增：提取测试集的指纹特征 =================
            ecfp_embeddings = batch.ecfp.view(-1, 2048)          # [B, 2048]
            # ==============================================================

            # 2. classifier 双塔前向预测
            out = classifier(gnn_embeddings, ecfp_embeddings)    # [B, num_classes]
            
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.cpu()              # 立刻回 CPU
            
            # 核心：获取预测为 1 (Active) 的概率，用于计算 AUC
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

            # labels：batch 中的分子对应 global valid_indices[start:end]
            for i_local, global_idx in enumerate(test_valid_cache[start:end]):
                if global_idx < len(test_active_values):
                    true_label_str = test_active_values[global_idx]
                    pred_label_str = str(predicted[i_local].item())
                    
                    total += 1
                    if pred_label_str == true_label_str:
                        correct += 1
                        
                    # 为 AUC 收集真实标签 (转为 int) 和 概率
                    all_trues.append(int(true_label_str))
                    all_probs.append(probs[i_local])

            # 释放本 batch 的大对象并清理显存
            del batch, h, gnn_embeddings, ecfp_embeddings, out, predicted
            gc.collect()
            torch.cuda.empty_cache()

            if verbose:
                print(f"Evaluated {min(end, len(test_graphs_cache))}/{len(test_graphs_cache)} test graphs. current acc = {100*correct/total if total>0 else 0:.2f}%")

    # 计算最终准确率
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    try:
        # 使用 sklearn 计算 ROC-AUC
        auc_score = roc_auc_score(all_trues, all_probs)
    except ValueError:
        print("警告：测试集中仅包含单一类别，无法计算有效AUC，默认返回0.5")
        auc_score = 0.5

    print(f"测试集总分子数: {len(test_active_values)}")
    print(f"成功预测分子数: {total}")
    print(f"预测正确数: {correct}")
    print(f"预测准确率: {accuracy:.2f}%")
    print(f"测试集 ROC-AUC: {auc_score:.4f}")
    
    # 同时返回 accuracy 和 auc_score
    return accuracy, auc_score

#======================================================================================
# 绘制并保存训练进度图表的函数
def plot_training_progress(contrastive_losses, classification_losses, train_accuracies, test_accuracies, total_epochs, A,final_model_path):
    plt.figure(figsize=(20, 5))
    
    # 绘制对比学习损失
    plt.subplot(1, 4, 1)
    plt.plot(range(total_epochs), contrastive_losses, label="Contrastive Loss", color="purple", marker="^")
    plt.title("Contrastive Learning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # 绘制分类损失
    plt.subplot(1, 4, 2)
    plt.plot(range(total_epochs), classification_losses, label="Classification Loss", color="red", marker="^")
    plt.title("Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # 绘制训练准确率
    plt.subplot(1, 4, 3)
    plt.plot(range(total_epochs), train_accuracies, label="Train Accuracy", color="blue", marker="o")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    
    # 绘制测试准确率
    plt.subplot(1, 4, 4)
    plt.plot(range(total_epochs), test_accuracies, label="Test Accuracy", color="green", marker="s")
    plt.axhline(y=85, color='r', linestyle='--', label='Early Stop Threshold (85%)')
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    final_model_path = final_model_path.replace('.pth',".jpg")
    plt.tight_layout()
    plt.savefig(final_model_path, dpi=300)
    plt.show()
    print('图像完成',final_model_path)
#===================================================================================
def matrixmake(active_1_name, active_0_name, number_matrix, Matrix, graph_matrix, A, n, similer_nuber):
    # 先把名字转成分子编号
    active_0_ids = [extract_digits(x) for x in active_0_name]
    active_1_ids = [extract_digits(x) for x in active_1_name]

    # 构建 number_matrix
    for idx in range(len(active_1_name)):
        # Matrix 每一行的语义：
        # [0 : similer_nuber)                     -> hard negatives
        # [similer_nuber : 2*similer_nuber)      -> hard positives
        # [2*similer_nuber]                      -> anchor
        hard_neg_ids = [int(Matrix[idx, j]) for j in range(similer_nuber)]
        hard_pos_ids = [int(Matrix[idx, j + similer_nuber]) for j in range(similer_nuber)]
        anchor_id = int(Matrix[idx, 2 * similer_nuber])

        # random negative: 从剩余的负类里选，排除 hard negatives
        neg_exclude = set(hard_neg_ids)
        neg_pool = [mid for mid in active_0_ids if mid not in neg_exclude]

        # random positive: 从剩余的正类里选，排除 hard positives 和 anchor
        pos_exclude = set(hard_pos_ids)
        pos_exclude.add(anchor_id)
        pos_pool = [mid for mid in active_1_ids if mid not in pos_exclude]

        # 这里按你当前 n 的定义，正常不会不够；留断言只是防御式检查
        if len(neg_pool) < n:
            raise ValueError(
                f"Row {idx}: negative random pool 不足，need={n}, got={len(neg_pool)}"
            )
        if len(pos_pool) < n:
            raise ValueError(
                f"Row {idx}: positive random pool 不足，need={n}, got={len(pos_pool)}"
            )

        rand_neg_ids = random.sample(neg_pool, n)
        rand_pos_ids = random.sample(pos_pool, n)

        # 按原来的列布局写回，后续 graph_matrix 和大部分训练逻辑都不用改
        # [0 : n) -> random negatives
        for j in range(n):
            number_matrix[idx, j] = rand_neg_ids[j]

        # [n : n+similer_nuber) -> hard negatives
        for j in range(similer_nuber):
            number_matrix[idx, j + n] = hard_neg_ids[j]

        # [n+similer_nuber : n+similer_nuber+n) -> random positives
        for j in range(n):
            number_matrix[idx, j + n + similer_nuber] = rand_pos_ids[j]

        # [n+similer_nuber+n : 2*(n+similer_nuber)) -> hard positives
        for j in range(similer_nuber):
            number_matrix[idx, j + 2 * n + similer_nuber] = hard_pos_ids[j]

        # 最后一列 -> anchor
        number_matrix[idx, 2 * (n + similer_nuber)] = anchor_id

    # 获取所有唯一索引
    onlylist = np.unique(np.array(number_matrix))
    print(f"number_matrix 形状: {number_matrix.shape}，总个数{number_matrix.shape[0]*number_matrix.shape[1]}")
    print(f"需要预计算 {len(onlylist)} 个唯一分子...")
    print(f"总体分子数量：{len(active_0_name)+len(active_1_name)}减少比例：{len(onlylist)/(len(active_0_name)+len(active_1_name))}")

    # 预计算所有分子图
    all_graphs, valid_indices, _ = precompute_all_graphs(onlylist.tolist(), A)
    print(f"成功预计算 {len(valid_indices)} 个分子图")

    # 建立索引到图的映射字典
    index_to_graph = dict(zip(valid_indices, all_graphs))

    # 填充 graph_matrix
    for idx in range(len(active_1_name)):
        for j in range(len(number_matrix[idx])):
            index_val = number_matrix[idx, j]
            if index_val in index_to_graph:
                graph_matrix[idx, j] = index_to_graph[index_val]
            else:
                graph_matrix[idx, j] = None
                print("none")

    return number_matrix, graph_matrix, all_graphs, valid_indices
#===================================================================================
def unwrap_state_dict(module):
    """
    如果 module 被包装（如 torch.compile 生成的 wrapper），尝试取 module._orig_mod.state_dict()
    否则返回 module.state_dict()
    """
    try:
        # torch.compile 等可能把原模块放在 _orig_mod 属性下
        orig = getattr(module, "_orig_mod", None)
        if orig is not None:
            return orig.state_dict()
    except Exception:
        pass
    # fallback
    return module.state_dict()

def number_to_label(num):
            """
            把 number -> label 的逻辑放这里。
            情形 A: 如果 active_1_name 是 active 分类的 name 列表（例如包含 'name4'），
                    则用 'num in active_1_name' 判定 label。
            情形 B: 如果你在别处有 number->label 的映射表（例如 number_label_map），
                    请把下面的判断替换为 number_label_map[num]。
            """
            # 假设 number 是形如 'name4' 或 '4' 的字符串/数字
            key = 'name'+str(num)
            # 优先检查你已有的 active_1_name（假设它是 active 类名字的集合/列表）
            if key in active_1_name:
                return 1
            # 否则默认 0 （如有更精确的映射请替换此处）
            elif key in active_0_name:
                return 0
            else:
                raise ValueError("类别值无效")

def TRAIN(A,n,similer_nuber,bili,sim_baifenbi,randon_baifenbi,savepth):
    # n为随机个数，similer_nuber为相似个数
    jieyue=0
    scaler = torch.cuda.amp.GradScaler()
    # MAX_GRAPH_PER_BATCH = 2000
    # 关闭RDKit的警告信息
    RDLogger.DisableLog('rdApp.*')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    active_values, active_0_name, active_1_name = extract_active_property(A+".sdf")
    if len(active_1_name) < n:
        n = len(active_1_name)
    if len(active_1_name) < similer_nuber:
        similer_nuber = len(active_1_name)  
    # 加载测试集数据
    if 'no_balance' in A:
        test_file_path = '/root/autodl-tmp/13_zhouxiaopeng/ours/validation/bace'
    else:
        test_file_path = A.replace('train/'+str(bili[0])+str(bili[1])+str(bili[2]), 'validation/')
        
    test_active_values, test_active_0_name, test_active_1_name = extract_active_property(test_file_path+ ".sdf")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备是{device}')
    
    # 构建训练矩阵
    graph_matrix = np.zeros((len(active_1_name), 2*(n+similer_nuber)+1), dtype=list)
    number_matrix = np.zeros((len(active_1_name), 2*(n+similer_nuber)+1), dtype=list)
    Matrix, k = matrix_make.matrix(A,similer_nuber)
    print('相似矩阵生成完成')
    # number_matrix,graph_matrix = matrixmake(active_1_name,active_0_name,number_matrix,Matrix,graph_matrix,A)

    # 每一次随机数相同的情况

    # 超参数
    # ========== 训练节奏 / 调度超参（你要求的 4 段节奏在这里配置） ==========
    total_epochs = 500          # 最大训练轮数（不超过400）
    # 图扰动（edge/node dropout）调度
    perturb_start = 100         # 从哪个 epoch 开始引入扰动（包含）
    perturb_ramp_end = 150      # 在哪一轮结束线性增加（150 之前为 ramp）
    target_drop_rate = 0.25     # 目标扰动强度（建议初始试 0.05~0.2）
    # 第二阶段（classification）调度
    second_phase_start = 200    # >=200 开启第二阶段（分类微调）
    second_phase_enhance_start = 200
    second_phase_enhance_end = 300  # 在 200-299 做增强行为
    # 早停（仅在 >300 时生效）
    early_stop_epoch_start = 301  # 在哪一轮之后才启用早停判定（你的要求：大于300后）
    early_stop_patience = 50      # 早停耐心（可调）
    # =======================================================================
    lr = 1e-3  # 学习率
    temperature = 0.05  # 对比学习温度参数
    all_reduce_molecule_numbers = []

    # 早停参数
    early_stop_threshold = 0.90  # 测试准确率阈值
    patience = 50  # 耐心值
    best_test_acc = 0.0
    patience_counter = 0
    
    # 性能指标跟踪变量
    contrastive_losses = []  # 对比学习损失历史
    classification_losses = []  # 分类损失历史
    train_accuracies = []  # 训练准确率历史
    test_accuracies = []  # 测试准确率历史

    # 获取特征维度
    if len(graph_matrix[0]) == 0:
        raise ValueError("无法获取分子图数据")
    graph,_,_=precompute_all_graphs([5],A)
    num_node_features = graph[0].x.size(1)
    # 创建GNN编码器
    print("创建GNN编码器...")
    hidden_dim1 = 64
    hidden_dim2 = 64
    encoder_output_dim = 32
    
    gnn_input = {
        "LR": lr,
        "NAME": "GNN_Encoder",
        "CHECKPOINT_DIR": "",
        "SIZE_LAYERS": [num_node_features, hidden_dim1, hidden_dim2, encoder_output_dim]
    }
    
    encoder = GNN(gnn_input).to(device)
    print("GNN编码器创建成功（未编译）")
    
    # 定义分类器
    classifier = ClassifierNetwork(gnn_dim=encoder_output_dim, ecfp_dim=2048, num_classes=2).to(device)    
    contrastive_optimizer = optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
    classification_optimizer = optim.AdamW(classifier.parameters(), lr=1e-5, weight_decay=1e-5)
    #=========================================================================
    w0 = len(active_values) / (2 * len(active_0_name))
    w1 = len(active_values) / (2 * len(active_1_name))

    # 可选：限制权重上界，避免过大
    max_weight = 10.0
    w0 = min(w0, max_weight)
    w1 = min(w1, max_weight)
    class_weights_tensor = torch.tensor([w0, w1], dtype=torch.float, device=device)

    # 最终 criterion：CrossEntropyLoss 支持 weight 参数
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 打印以便调试／记录
    # print(f"Class counts (train): {class_counts}, class_weights: {class_weights}")
    #=========================================================================
    
    # 学习率调度器
    contrastive_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        contrastive_optimizer, T_max=total_epochs, eta_min=1e-5
    )
    classification_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        classification_optimizer, T_max=total_epochs, eta_min=1e-6
    )
    try:
        sample_graphs = None
        try:
            if 'graph' in locals() and isinstance(graph, list) and len(graph) > 0:
                sample_graphs = graph
        except:
            sample_graphs = None

        # 若没有可复用样本，则尝试构造一个最小索引样本（可能越界则捕获异常）
        if not sample_graphs:
            sample_graphs, _, _ = precompute_all_graphs([5], A)

        if sample_graphs and len(sample_graphs) > 0:
            sample_batch = Batch.from_data_list(sample_graphs).to(device)
            # 用 autocast 执行一次前向（如果希望也编译 backward，可加 scaled backward，但通常前向就能触发编译）
            with torch.amp.autocast('cuda'):
                h = encoder(sample_batch.x, sample_batch.edge_index, drop_rate=0.0)
                g = global_mean_pool(h, sample_batch.batch)
                
                # ==== 修改的地方 ====
                # 1. 把预热样本的指纹也提出来
                sample_ecfp = sample_batch.ecfp.view(-1, 2048) 
                
                # 2. 传 2 个参数给双塔分类器
                _ = classifier(g, sample_ecfp) 
                # ========================================
            print("Warm-up 完成：已触发 torch.compile 的首轮编译。")
        else:
            print("Warm-up 跳过：未能获取有效 sample_graphs。")
    except Exception as e:
        print("Warm-up 阶段出现异常（可忽略）：", e)

    # ================= 新增：预加载测试集 =================
    
    t_start = time.time()
    all_test_indices = list(range(len(test_active_values)))
    # 调用 precompute_all_graphs 一次性读入内存
    test_graphs_cache, test_valid_cache, _ = precompute_all_graphs(all_test_indices, test_file_path)
    # ====================================================
    # ================= 统计模型可训练参数量 =================
    # encoder_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    # classifier_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    # total_trainable_params = encoder_params + classifier_params
    
    # print("\n" + "="*40)
    # print(f"📊 模型参数量统计:")
    # print(f"  - GNN 编码器参数量: {encoder_params:,}")
    # print(f"  - 多模态分类器参数量: {classifier_params:,}")
    # print(f"  - 总可训练参数量: {total_trainable_params:,}")
    # print("="*40 + "\n")
    # ========================================================
    print("开始训练...")
    
    for epoch in range(total_epochs):

        if epoch == 0:
            epochtime=time.time()
        else:
            print(time.time()-epochtime)
            epochtime=time.time()
        # --- 计算当前 epoch 的图扰动强度（严格按你的节奏）
        if epoch < perturb_start:
            current_drop_rate = 0.0
        elif perturb_start <= epoch < perturb_ramp_end:
            frac = (epoch - perturb_start) / max(1, (perturb_ramp_end - perturb_start))
            current_drop_rate = frac * target_drop_rate
        else:
            current_drop_rate = target_drop_rate
        print(f'Epoch: {epoch+1}/{total_epochs}')
        encoder.train()
        classifier.eval()
        if epoch % 40 == 0:
            number_matrix,graph_matrix,train_graphs, train_valid_indices = matrixmake(active_1_name,active_0_name,number_matrix,Matrix,graph_matrix,A,n,similer_nuber)
        begin_time = time.time()
        # #==================================================
        # train_graph_matrix = graph_matrix
        # train_number_matrix = number_matrix
        # if epoch <100:
        #     print("无数据增强")
        # ================= 每个 Epoch 动态且同步的随机抽样 =================
        total_rows = len(active_1_name)  # 矩阵的总行数就是阳性分子的个数
        
        if total_rows > 700 and total_rows < 1000:
            # 设定你想要抽取的行数（比如固定抽 600 行，你可以随时修改）
            sample_size = 600  
            
            # 从 0 到 total_rows-1 中，随机抽取 600 个不重复的数字作为行索引
            random_indices = np.random.choice(total_rows, sample_size, replace=False)
            
            # 使用 numpy 的高级切片机制，同时对图矩阵和数字矩阵进行切片
            # 这样截取出来的行绝对是一一对应的！
            train_graph_matrix = graph_matrix[random_indices]
            train_number_matrix = number_matrix[random_indices]
        elif total_rows > 1000:
            # 设定你想要抽取的行数（比如固定抽 600 行，你可以随时修改）
            sample_size = 300  
            
            # 从 0 到 total_rows-1 中，随机抽取 600 个不重复的数字作为行索引
            random_indices = np.random.choice(total_rows, sample_size, replace=False)
            
            # 使用 numpy 的高级切片机制，同时对图矩阵和数字矩阵进行切片
            # 这样截取出来的行绝对是一一对应的！
            train_graph_matrix = graph_matrix[random_indices]
            train_number_matrix = number_matrix[random_indices]
        else:
            # 如果分子数不到 600（如 bace, clintox），就全量参与计算
            train_graph_matrix = graph_matrix
            train_number_matrix = number_matrix
        # ==================================================================

        if epoch < 100:
            print("无数据增强")
        print("  进行行级批量Circle Loss对比学习...")
            

        # 使用修正的Circle Loss
        batch_circle_loss = FixedBatchCircleLoss(margin=0.25, gamma=64).to(device)

        batch_size = 128  # 减小批次大小确保稳定性

        for batch_start in tqdm(range(0, train_graph_matrix.shape[0], batch_size)):
            edge_time=time.time()
            batch_end = min(batch_start + batch_size, train_graph_matrix.shape[0])
        #==========================================
            batch_indices = range(batch_start, batch_end)
            
            batch_anchors = []
            batch_positives_list = []
            batch_negatives_list = []
            
            # 预计算整个批次的嵌入
            all_row_embeddings = []

            # 提取本 batch 对应的 number_matrix（注意 train_number_matrix 在外层已被赋值）
            # train_number_matrix 的形状为 [num_rows, num_cols]
            batch_row_indices = list(batch_indices)

            # 1) 收集 batch 内所有编号（可能有重复、None 或 0 等占位）
            unique_ids = []
            for r in batch_row_indices:
                cols = train_number_matrix[r]
                for num in cols:
                    # 只接受整数索引（根据你的 matrixmake 逻辑，这里通常是 int）
                    try:
                        if num is None:
                            continue
                        # 有时 num 是 numpy 类型或字符串，尝试转换为 int
                        nid = int(num)
                        unique_ids.append(nid)
                    except Exception:
                        continue
            if len(unique_ids) == 0:
                # 如果没有任何有效图（极端情况），按原逻辑生成空占位
                for r in batch_row_indices:
                    all_row_embeddings.append(torch.zeros((0, encoder_output_dim), device=device))
            else:
                unique_ids = list(set(unique_ids))

                # 2) 从 train_graph_matrix 构建 id -> Data 映射（取第一处出现的 Data）
                id_to_graph = {}
                for r in batch_row_indices:
                    cols_graphs = train_graph_matrix[r]
                    cols_nums = train_number_matrix[r]
                    for j, num in enumerate(cols_nums):
                        try:
                            if num is None:
                                continue
                            nid = int(num)
                        except Exception:
                            continue
                        if nid not in id_to_graph:
                            g = cols_graphs[j]
                            # 只接受有效的 Data 对象
                            if g is None:
                                continue
                            if hasattr(g, 'x') and isinstance(getattr(g, 'x'), torch.Tensor):
                                id_to_graph[nid] = g

                # 3) 保证 unique_ids 只保留在 id_to_graph 中存在的
                unique_ids = [uid for uid in unique_ids if uid in id_to_graph]
                if len(unique_ids) == 0:
                    # 没有可编码的图，按占位返回
                    for r in batch_row_indices:
                        all_row_embeddings.append(torch.zeros((0, encoder_output_dim), device=device))
                else:
                    # 4) 按 chunk 对所有唯一图做前向（保留 grad，放在 device 上）
                    enc_chunk = 1024  # 根据显存可调整
                    index_to_emb = {}  # int id -> tensor [D], 保存在 device，require_grad True
                    # 我们需要保证顺序可重复使用，按 unique_ids 分块
                    for s in range(0, len(unique_ids), enc_chunk):
                        chunk_ids = unique_ids[s:s+enc_chunk]
                        chunk_graphs = []
                        chunk_valid_ids = []
                        for uid in chunk_ids:
                            g = id_to_graph.get(uid, None)
                            if g is None:
                                continue
                            # 把 Data 放到 device（为 encoder 前向）
                            try:
                                g_dev = g.to(device)
                            except Exception:
                                # fallback: 尝试单独搬运张量字段
                                if hasattr(g, 'x') and isinstance(g.x, torch.Tensor):
                                    g.x = g.x.to(device)
                                if hasattr(g, 'edge_index') and isinstance(g.edge_index, torch.Tensor):
                                    g.edge_index = g.edge_index.to(device)
                                g_dev = g
                            chunk_graphs.append(g_dev)
                            chunk_valid_ids.append(uid)

                        if len(chunk_graphs) == 0:
                            continue

                        batch_data = Batch.from_data_list(chunk_graphs).to(device)
                        # 前向（保持梯度）
                        with torch.amp.autocast('cuda', enabled=True):
                            h = encoder(batch_data.x, batch_data.edge_index, drop_rate=current_drop_rate)
                            pooled = global_mean_pool(h, batch_data.batch)  # [len(chunk_graphs), D]

                        # pooled 中第 k 行对应 chunk_valid_ids[k]
                        for k_uid, emb in zip(chunk_valid_ids, pooled):
                            # emb 保持在 device 上，不 detach
                            index_to_emb[int(k_uid)] = emb

                        # 释放临时
                        del batch_data, h, pooled, chunk_graphs
                        # 注意暂时不要清空 index_to_emb，因为后续需要用到它（参与 loss，需保留 grad）
                        torch.cuda.empty_cache()

                    # 5) 对每一行重建 row_embedding（按列顺序），缺失位置放 0 向量以保持列对齐
                    zero_vec = lambda: torch.zeros((1, encoder_output_dim), device=device)
                    for r in batch_row_indices:
                        cols_nums = train_number_matrix[r]
                        row_emb_list = []
                        for num in cols_nums:
                            try:
                                nid = int(num)
                            except Exception:
                                nid = None
                            if nid is None or nid not in index_to_emb:
                                # 占位（与原来缺失行为一致：保持列数一致以便后续按列索引）
                                row_emb_list.append(zero_vec())
                            else:
                                # index_to_emb[nid] 的形状为 [D], 需要 [1, D]
                                e = index_to_emb[nid]
                                if e.dim() == 1:
                                    e = e.unsqueeze(0)
                                row_emb_list.append(e)

                        # 合并为 [num_cols, D]
                        if len(row_emb_list) == 0:
                            row_embedding = torch.zeros((0, encoder_output_dim), device=device)
                        else:
                            row_embedding = torch.cat(row_emb_list, dim=0)  # [num_cols, D]
                        all_row_embeddings.append(row_embedding)
            print(f"embedding用时{time.time()-edge_time}秒")
            edge_time= time.time()
            # 为每个样本构建正负样本
            for i, row_idx in enumerate(batch_indices):
                if all_row_embeddings[i] is None:
                    continue
                    
                row_embedding = all_row_embeddings[i]
                
                anchor = row_embedding[-1].unsqueeze(0)  # [1, embed_dim]
                
                # 获取正样本：相似活性分子
                positive_start = n + similer_nuber
                
                positives = row_embedding[positive_start:positive_start*2]  # [num_pos, embed_dim]
                
                # 获取负样本：相似非活性分子
                negatives = row_embedding[0:n+similer_nuber]  # [num_neg, embed_dim]
                
                if positives.size(0) > 0 and negatives.size(0) > 0:
                    batch_anchors.append(anchor.squeeze(0))  # [embed_dim]
                    batch_positives_list.append(positives)
                    batch_negatives_list.append(negatives)
            print(f"构建正负样本用时{time.time()-edge_time}秒")
            edge_time= time.time()
            # 批量计算Circle Loss
            if batch_anchors:
                batch_anchors = torch.stack(batch_anchors)  # [batch_size, embed_dim]
                
                with torch.amp.autocast('cuda'):
                    contrastive_loss = batch_circle_loss(batch_anchors, batch_positives_list, batch_negatives_list)
                
                # 优化步骤
                contrastive_optimizer.zero_grad(set_to_none=True)
                scaler.scale(contrastive_loss).backward()
                
                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                
                scaler.step(contrastive_optimizer)
                scaler.update()
                print(f"loss计算完成{time.time()-edge_time}秒,损失为{contrastive_loss}")
                edge_time = time.time()
            else:
                contrastive_loss = torch.tensor(0.0, device=device)
        print(f"第一阶段结束{time.time()-begin_time}第二阶段")
        begin_time = time.time()
        if epoch>= second_phase_start:
            print("第二阶段")
            #=====================================================
            # 2) 训练：使用 encoder 和 classifier 训练模型
            encoder.eval()
            classifier.train()

            # 1) 准备：获取本次训练需要的所有唯一分子的索引（与前面对比学习里的 unique_graphs/unique_numbers 类似）
            #    这里我们从 number_matrix 中提取所有 unique id（str 形式）
            all_needed_numbers = []
            for row in train_number_matrix:
                for num in row:
                    all_needed_numbers.append(str(num))
            all_needed_numbers = list(dict.fromkeys(all_needed_numbers))

            # 先尝试从已有 graph_matrix 取图（若存在）
            index_to_graph = {}
            for i_row, row in enumerate(train_number_matrix):
                for j_col, num in enumerate(row):
                    key = str(num)
                    g = train_graph_matrix[i_row][j_col]  # 注意：graph_matrix 保存的可能在 CPU
                    if g is not None and key not in index_to_graph:
                        index_to_graph[key] = g

            # 对仍缺失的 key，批量调用 precompute_all_graphs 重新加载（并映射到字符串 key）
            missing_keys = [k for k in all_needed_numbers if k not in index_to_graph]
            if len(missing_keys) > 0:
                # 把 missing_keys 转成整数索引（你的 SDF 索引通常是数字或 'name123'）
                missing_indices = []
                key_from_index = {}
                for k in missing_keys:
                    try:
                        idx = int(str(k).replace('name',''))
                        missing_indices.append(idx)
                        key_from_index[idx] = k
                    except:
                        # 跳过不能解析的 key（不常见）
                        continue

                # 分块调用预加载（precompute_all_graphs 已按 batch 分批读取 .sdf）
                chunk_pre = 1024
                for s in range(0, len(missing_indices), chunk_pre):
                    sub = missing_indices[s:s+chunk_pre]
                    graphs, valid_idxs, _ = precompute_all_graphs(sub, A)
                    # precompute_all_graphs 的 valid_idxs 返回实际有效的全局索引（int）
                    for g_idx, g in zip(valid_idxs, graphs):
                        k = key_from_index.get(g_idx, str(g_idx))
                        if k not in index_to_graph:
                            index_to_graph[k] = g

            # =====================================================
            # 2) 构建去重后的训练数据集，并进行端到端微调
            # =====================================================
            encoder.train()     # 核心：解冻 Encoder
            classifier.train() 

            train_cls_graphs = []
            train_cls_labels = []

            # 极其巧妙的去重：index_to_graph 是一个字典，遍历它的 values 天生就是无重复的分子！
            for key, g in index_to_graph.items():
                if g is None:
                    continue
                # 获取标签
                try:
                    lab = number_to_label(key)
                except Exception:
                    try:
                        lab = int(active_values[int(key)])
                    except Exception:
                        continue
                
                # 确保图的张量格式正确
                if not hasattr(g, 'x') or g.x is None: continue
                if not isinstance(g.x, torch.Tensor): g.x = torch.tensor(g.x)
                if not isinstance(g.edge_index, torch.Tensor): g.edge_index = torch.tensor(g.edge_index)
                
                train_cls_graphs.append(g)
                train_cls_labels.append(lab)

            # 打乱数据以防过拟合
            if len(train_cls_graphs) > 0:
                combined = list(zip(train_cls_graphs, train_cls_labels))
                random.shuffle(combined)
                train_cls_graphs, train_cls_labels = zip(*combined)

            classification_loss_total = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            # 设置 Batch Size 控制显存
            cls_batch_size = 128  

            # 实时前向与反向传播 (取代了原来提前把 embedding 存进字典的耗显存做法)
            for i in range(0, len(train_cls_graphs), cls_batch_size):
                batch_graphs = train_cls_graphs[i : i+cls_batch_size]
                if len(batch_graphs) < 2:
                    continue
                batch_labels = torch.tensor(train_cls_labels[i : i+cls_batch_size], dtype=torch.long, device=device)
                
                batch_data = Batch.from_data_list(batch_graphs).to(device)
                
                classification_optimizer.zero_grad(set_to_none=True)
                contrastive_optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    # 1. GNN 提取图结构局部特征 (保持不变)
                    h = encoder(batch_data.x, batch_data.edge_index, drop_rate=current_drop_rate)
                    gnn_embeddings = global_mean_pool(h, batch_data.batch)
                    
                    # ================= 新增：指纹特征提取 =================
                    # 从 PyG 的 Batch 背包中取出自动拼接好的 2048 维指纹
                    # 使用 view(-1, 2048) 确保维度绝对安全，不受单样本影响
                    ecfp_embeddings = batch_data.ecfp.view(-1, 2048)
                    # =======================================================
                    
                    # 2. 关键：同时将两股特征送入双塔分类器！
                    classifier_out = classifier(gnn_embeddings, ecfp_embeddings)
                    
                    # 3. 计算交叉熵损失
                    loss = criterion(classifier_out, batch_labels)
                    
                scaler.scale(loss).backward()
                
                # 关键：同时更新两个网络的参数
                scaler.step(classification_optimizer)
                scaler.step(contrastive_optimizer)
                scaler.update()
                
                classification_loss_total += loss.item()
                _, predicted = torch.max(classifier_out.data, 1)
                total_predictions += batch_labels.size(0)
                correct_predictions += (predicted == batch_labels).sum().item()

            # 计算本 Epoch 的平均指标
            classification_loss = classification_loss_total / (math.ceil(len(train_cls_graphs) / cls_batch_size) if len(train_cls_graphs) > 0 else 1)
            train_accuracy = 100 * correct_predictions / total_predictions if total_predictions > 0 else 0.0

            # 6) 评估测试集准确率（保留你原来的评估函数）
            test_accuracy,test_auc = evaluate_test_accuracy_fixed(test_graphs_cache, test_valid_cache,encoder, classifier, test_active_values, test_file_path, device)
            # 4. 记录指标
            contrastive_losses.append(contrastive_loss.item())
            classification_losses.append(classification_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            
            
            contrastive_scheduler.step()
            classification_scheduler.step()

        
            print(f"Epoch [{epoch + 1}/{total_epochs}]:")
            print(f"  对比损失: {contrastive_loss.item():.4f}")
            print(f"  分类损失: {classification_loss:.4f}")
            print(f"  训练准确率: {train_accuracy:.2f}%")
            print(f"  测试准确率: {test_accuracy:.2f}%")
            if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                print("警告：对比损失出现NaN或Inf！")
                break
            # 6. 早停检查
            if test_auc > best_test_acc:
                best_test_acc = test_auc
                patience_counter = 0
                # 保存最佳模型
                best_model_path = savepth
                torch.save({
                            'encoder_state_dict': unwrap_state_dict(encoder),
                            'classifier_state_dict': unwrap_state_dict(classifier),
                            'epoch': epoch,
                            'test_accuracy': test_accuracy
                        }, best_model_path)
                print(f"  新的最佳模型已保存，测试准确率: {best_test_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"  早停计数器: {patience_counter}/{patience}")
                
        # # 早停条件
        # if epoch >= early_stop_epoch_start:
        #     if test_auc >= early_stop_threshold :
        #         print(f"测试准确率达到{early_stop_threshold}%，触发早停！")
        #         break
                    
        #     if patience_counter >= patience:
        #         print(f"测试准确率连续{patience}轮未改善，触发早停！")
        #         break

        
    print("训练完成！")
    print(f"最佳测试准确率: {best_test_acc:.2f}%")
    
    # 保存最终模型
    final_model_path = best_model_path
    
    print(f"最终模型已保存到{final_model_path}")
    
    # 绘制训练进度
    plot_training_progress(contrastive_losses, classification_losses, train_accuracies, test_accuracies, len(contrastive_losses), A, final_model_path)
    
    return epoch, all_reduce_molecule_numbers, len(active_values), test_accuracy

if __name__ == '__main__':
    for chaocan in range(4):
        # chaocan =chaocan + 1
        for i in range(1):
            if i == 0:
                bili=[8,1,1]
            elif i == 1:
                bili=[6,2,2]
            elif i == 2:
                bili=[4,3,3]
            elif i == 3:
                bili=[2,4,4]
            # =========================================================================
            daimafangshi = 0
            #这里是模型方式，0是比例 1是固定数字
            # =========================================================================
            print(str(bili[0])+':'+str(bili[1])+':'+str(bili[2]))
            multiprocessing.set_start_method('spawn', force=True)
            filesss=[
                # "train/nr-ar",
                # 'train/nr-ahr',
                # 'train/nr-ar-lbd',
                # 'train/nr-aromatase',
                # 'train/nr-er',
                # 'train/nr-er-lbd',
                # 'train/nr-ppar-gamma',
                # 'train/sr-are',
                # 'train/sr-atad5',
                # 'train/sr-hse',
                # 'train/sr-mmp',
                # "train/sr-p53",
                # "train/ABBBP",
                "train/bace",
                # "train/clintox",
                # 'train/HIV',
                # 'train/bace_no_balance1',
                # 'train/bace_no_balance2',
                # 'train/bace_no_balance3',
                ]
            files=[]
            for fil in filesss:
                files.append(fil.replace('train/','ours/train/'+str(bili[0])+str(bili[1])+str(bili[2])))
            print(files)
            print("开始训练所有模型...")
            print("=" * 50)
            aooepoch = 0
            txt = ['训练集,验证集,测试集比例为:'+str(bili[0])+':'+str(bili[1])+':'+str(bili[2])+'\n']
            total_start_time = time.time()

            # 测试单个配置
            # for n in range(50, 160, 50):
            # for similer_nuber in range(50, 160, 50):

            testfiles = []
            pthfiles = []
            start_time = time.time()
            print(files)
            for file in files:
                _, active_0_name, active_1_name = extract_active_property(file+".sdf")
                        # n为随机个数，similer_nuber为相似个数
                if daimafangshi == 0:
                    baifenshu=min(len(active_0_name), len(active_1_name))
                            # if chaocan //2 == 0:
                            #     n = baifenshu * 4
                            #     similer_nuber = baifenshu * (chaocan + 4 )
                            # elif chaocan //2 == 1:
                            #     n = baifenshu * 5
                            #     similer_nuber = baifenshu * (chaocan + 2 )
                    # randon_numberbili = (chaocan//3)*2+1
                    # similer_numbbili = (chaocan %3)*2+1
                    # print(f"随机的比例为{randon_numberbili*10}%,相似的比例为{similer_numbbili*10}%")
                    randon_numberbili = chaocan*25
                    similer_numbbili = 30
                    similer_nuber = (baifenshu * similer_numbbili)//100
                    n = ((baifenshu-similer_nuber) * randon_numberbili)//100
                    print(f"随机的比例为{randon_numberbili}%,相似的比例为{similer_numbbili}% ;个数分别为随机的{n}， 相似的{similer_nuber}")
                elif daimafangshi == 1:
                    n = similer_nuber=50
                aooepoch=aooepoch+1
                print(f"开始训练模型{file}，随机个数：{n}，相似个数：{similer_nuber},总体训练次数：{aooepoch}")
                if daimafangshi == 1:
                    print(f"这里为固定参数110")
                    epoch, all_reduce_molecule_numbers, listmo, test_accuracy = TRAIN(file, n, similer_nuber, bili,sim_baifenbi=0, randon_baifenbi=0,savepth=file.replace('train', 'pth') +'randon_number-'+str(n)+'similer_nuber-'+str(similer_nuber)+ '_best_model.pth')
                elif daimafangshi == 0:
                    print(f"这里为固定比例")
                    epoch, all_reduce_molecule_numbers, listmo, test_accuracy = TRAIN(file, n, similer_nuber, bili,sim_baifenbi=0, randon_baifenbi=0,savepth=file.replace('train', 'pth') +'randon_number-'+str(n)+'similer_nuber-'+str(similer_nuber)+ '_best_model.pth')
                wenben = [f"文件名：{file}随机个数： {n} 相似个数:{similer_nuber} 训练轮数：{epoch} 用到的计算数据占总的比例：{sum(all_reduce_molecule_numbers)/(epoch+1)/listmo*100 if listmo > 0 else 0} % 准确率：{test_accuracy}"]
                txt.append(wenben)
                pthfiles.append(file.replace('train', 'pth') +'randon_number-'+str(110)+'similer_nuber-'+str(110)+ '_best_model.pth')
                testfiles.append(file.replace('train', 'test'))
                evertime = time.time() - start_time
                print(f"训练完成，用时 :{evertime}秒")
                # filedict = {pthfiles:testfiles for pthfiles, testfiles in zip(pthfiles, testfiles)}
                # results, predictions=test.test(filedict)
                # txt.append(results)
                end_time = time.time()
                print(f"所有模型训练完成，用时 :{end_time - start_time}秒")
                txt.append(f'所有模型训练完成，用时 :{end_time - start_time}秒')
                filename = file.replace('train', 'out')+"trainout"+str(bili[0])+str(bili[1])+str(bili[2])+f"随机个数： {n} 相似个数:{similer_nuber}"+".txt"
                save_vector_to_txt(txt, filename)
                print(f"向量已成功保存到 {filename}")
            total_time = time.time() - total_start_time
            print(f"\n" + "=" * 60)
            print(f" 所有训练完成!")
            print(f" 总用时:  ({total_time/3600:.1f}小时)")
            print("=" * 50)