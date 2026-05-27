import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import re
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import SDF_make
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def precompute_all_graphs(active_list, A, batch_size=256):  # A100可以处理更大的批次
    """针对A100优化的批量预加载函数"""
    supplier = Chem.SDMolSupplier(A+".sdf")
    graphs = []
    listuseful = []
    mol_names = []
    
    # print(f"开始批量处理 {len(active_list)} 个分子，批次大小: {batch_size}")
    
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

                graph = SDF_dispose.molecule_to_pyg_graph(mol)
                
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
        
        # 批量转移到GPU - A100可以并行处理更多
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

    def forward(self, x, edge_index):  # forward propagation includes defining layers
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
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super(ClassifierNetwork, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),  # 增加dropout
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 增加一层
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)
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
def evaluate_test_accuracy_fixed(encoder, classifier, test_active_values, test_file_path, device, batch_size=256, verbose=False):
    """
    分批评估测试集准确率（避免一次性把整个 test set 放到 GPU）。
    batch_size: 每次送到 GPU 的分子数（根据显存调小或调大）。
    """
    import gc
    encoder.eval()
    classifier.eval()

    correct = 0
    total = 0

    # 1) 先得到所有测试索引与对应的图（图保存在 CPU 或依据 precompute_all_graphs 实现）
    all_test_indices = list(range(len(test_active_values)))
    all_graphs, valid_indices, _ = precompute_all_graphs(all_test_indices, test_file_path)
    if len(all_graphs) == 0:
        print("无法获取测试分子图数据")
        return 0.0

    # 2) 按 batch_size 分块评估，每块只在 GPU 上存在短时间
    with torch.no_grad():
        for start in range(0, len(all_graphs), batch_size):
            end = min(start + batch_size, len(all_graphs))
            sub_graphs = all_graphs[start:end]

            # 确保 sub_graphs 的张量在 CPU（如果 precompute_all_graphs 已经把它们移到 GPU，则可跳过）
            # 在这里把 sub_graphs 统一移动到 device（Batch.from_data_list(...).to(device) 会做统一搬运）
            batch = Batch.from_data_list(sub_graphs).to(device)

            # encoder 前向并池化
            # 注意：encoder 的前向在你的脚本有两种签名（有无 batch 参数），此处按你原代码使用 encoder(batch.x, batch.edge_index)
            h = encoder(batch.x, batch.edge_index)
            g = global_mean_pool(h, batch.batch)    # [B, D]

            # classifier 前向
            out = classifier(g)                      # [B, num_classes]
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.cpu()              # 立刻回 CPU

            # labels：batch 中的分子对应 global valid_indices[start:end]
            # valid_indices 包含被 precompute 成功的全局索引
            for i_local, global_idx in enumerate(valid_indices[start:end]):
                if global_idx < len(test_active_values):
                    true_label = test_active_values[global_idx]
                    pred_label = str(predicted[i_local].item())
                    total += 1
                    if pred_label == true_label:
                        correct += 1

            # 释放本 batch 的大对象并清理显存
            del batch, h, g, out, predicted
            gc.collect()
            torch.cuda.empty_cache()

            if verbose:
                print(f"Evaluated {min(end, len(all_graphs))}/{len(all_graphs)} test graphs. current acc = {100*correct/total if total>0 else 0:.2f}%")

    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f"测试集总分子数: {len(test_active_values)}")
    print(f"成功预测分子数: {total}")
    print(f"预测正确数: {correct}")
    print(f"预测准确率: {accuracy:.2f}%")
    return accuracy

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
    print('图像完成')
#===================================================================================
def matrixmake(active_1_name, active_0_name, number_matrix, Matrix, graph_matrix, A, n, similer_nuber):
    # 构建 number_matrix
    for idx in range(len(active_1_name)):
        list0 = random.sample(active_0_name, n)
        list1 = random.sample(active_1_name, n)
        c0 = []
        c1 = []
        for i in list0:
            c0.append(extract_digits(i))
        for i in list1:
            c1.append(extract_digits(i))
        for j in range(len(c0)):
            number_matrix[idx, j] = c0[j]
        for j in range(len(c1)):
            number_matrix[idx, j+n+similer_nuber] = c1[j]
        for j in range(similer_nuber):
            number_matrix[idx, j+n] = Matrix[idx, j]
        for j in range(similer_nuber):
            number_matrix[idx, j+n*2+similer_nuber] = Matrix[idx, j+similer_nuber]
        number_matrix[idx, 2*(n+similer_nuber)] = Matrix[idx, 2*similer_nuber]
    
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
        # if idx % 10 == 0:
            # print(f'matrixmake进度:{idx}/{len(active_1_name)}')
            
        for j in range(len(number_matrix[idx])):
            index_val = number_matrix[idx, j]
            if index_val in index_to_graph:
                graph_matrix[idx, j] = index_to_graph[index_val]
            else:
                graph_matrix[idx, j] = None
                print("none")
    
    return number_matrix, graph_matrix
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
    MAX_GRAPH_PER_BATCH = 2000
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
    total_epochs = 400  # 总训练轮数
    lr = 1e-3  # 学习率
    temperature = 0.05  # 对比学习温度参数
    all_reduce_molecule_numbers = []

    # 早停参数
    early_stop_threshold = 90.0  # 测试准确率阈值
    patience = 30  # 耐心值
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
    classifier_hidden_dim = 64
    classifier = ClassifierNetwork(
        input_dim=encoder_output_dim,
        hidden_dim=classifier_hidden_dim,
        num_classes=2
    ).to(device)
    try:
        encoder = torch.compile(encoder, backend='inductor', mode='default', dynamic=True)
        classifier = torch.compile(classifier, backend='inductor', mode='default', dynamic=True)
        print("encoder 与 classifier 已使用 torch.compile 编译（dynamic=True）")
    except Exception as e:
        # 若编译失败则保留原模型，继续训练（打印警告）
        print("警告：torch.compile 编译失败，继续使用未编译模型。错误：", e)
    # 分别定义优化器
    # contrastive_optimizer = optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=1e-5)
    # classification_optimizer = optim.AdamW(classifier.parameters(), lr=1e-4, weight_decay=1e-5)
    contrastive_optimizer = optim.AdamW(encoder.parameters(), lr=1e-5, weight_decay=1e-5)
    classification_optimizer = optim.AdamW(classifier.parameters(), lr=1e-5, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    contrastive_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        contrastive_optimizer, T_max=total_epochs, eta_min=1e-6
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
            sample_graphs, _, _ = precompute_all_graphs([0], A)

        if sample_graphs and len(sample_graphs) > 0:
            sample_batch = Batch.from_data_list(sample_graphs).to(device)
            # 用 autocast 执行一次前向（如果希望也编译 backward，可加 scaled backward，但通常前向就能触发编译）
            with torch.amp.autocast('cuda'):
                h = encoder(sample_batch.x, sample_batch.edge_index)
                g = global_mean_pool(h, sample_batch.batch)
                _ = classifier(g)
            print("Warm-up 完成：已触发 torch.compile 的首轮编译。")
        else:
            print("Warm-up 跳过：未能获取有效 sample_graphs。")
    except Exception as e:
        print("Warm-up 阶段出现异常（可忽略）：", e)

        
    print("开始训练...")
    
    for epoch in range(total_epochs):
        if epoch == 0:
            epochtime=time.time()
        else:
            print(time.time()-epochtime)
            epochtime=time.time()
        print(f'Epoch: {epoch+1}/{total_epochs}')
        
        if epoch % 40 == 0:
            number_matrix,graph_matrix = matrixmake(active_1_name,active_0_name,number_matrix,Matrix,graph_matrix,A,n,similer_nuber)
        # 阶段1: 对比学习 - 使用修正的批量Circle Loss
        encoder.train()
        classifier.eval()

        print("  进行行级批量Circle Loss对比学习...")
        if len(active_1_name) < 600:
            # 使用修正的Circle Loss
            batch_circle_loss = FixedBatchCircleLoss(margin=0.25, gamma=64).to(device)

            batch_size = 64  # 减小批次大小确保稳定性

            # ---------- 替换开始：仅替换这里的对比学习循环部分 ----------

            # 1) 先把 graph_matrix 与 number_matrix 同步展平（保持顺序一致）
            flat_graphs = [g for row in graph_matrix for g in row]
            flat_numbers = [nm for row in number_matrix for nm in row]  # number_matrix 与 graph_matrix 一一对应

            if len(flat_graphs) != len(flat_numbers):
                raise RuntimeError(f"graph_matrix 与 number_matrix 长度不一致: {len(flat_graphs)} vs {len(flat_numbers)}")

            # 2) 用 number (或 name 字符串，如 'name4') 作为去重 key -> 构建 unique 列表
            unique_map = {}
            unique_graphs = []
            unique_numbers = []
            unique_labels = []

            for g, num in zip(flat_graphs, flat_numbers):
                # 确保 num 可哈希为字符串（统一）
                key = str(num)
                if key in unique_map:
                    continue
                idx = len(unique_graphs)
                unique_map[key] = idx
                unique_graphs.append(g)
                unique_numbers.append(key)
                unique_labels.append(number_to_label(key))

            if len(unique_graphs) < 2:
                raise RuntimeError(f"去重后分子太少：{len(unique_graphs)}，无法做对比学习。")
            jieyue = len(unique_graphs)+jieyue
            # 3) 建立 label -> 索引 列表，便于采样正/负
            label_to_indices = {}
            for i, lab in enumerate(unique_labels):
                label_to_indices.setdefault(int(lab), []).append(i)

            

            # ---------- 全量一次性 forward（替换你原来对比学习那段） ----------
            # encoder.train()
            # batch_circle_loss = FixedBatchCircleLoss(margin=0.25, gamma=64).to(device)
            print(f"去重后分子数：{len(unique_graphs)}")
            # 1) 把所有去重图合成一个大 Batch（unique_graphs 在前面已构造）
            full_batch = Batch.from_data_list(unique_graphs).to(device)

            # 2) 前向，得到 graph-level embeddings（不要 detach）
            with torch.amp.autocast('cuda'):
                h = encoder(full_batch.x, full_batch.edge_index)   # node-level
                embs = global_mean_pool(h, full_batch.batch)       # [N, D]

            # 3) 归一化（可选，但通常能稳定对比学习）
            embs = F.normalize(embs, p=2, dim=1)

            # 4) 构造每个 anchor 的 positives/negatives（按 unique_labels）
            labels_tensor = torch.tensor([int(x) for x in unique_labels], device=device)

            positives_lists = []
            negatives_lists = []
            N, D = embs.size()
            for i in range(N):
                same = (labels_tensor == labels_tensor[i]).nonzero(as_tuple=False).view(-1)
                same = same[same != i]  # 删掉 self
                diff = (labels_tensor != labels_tensor[i]).nonzero(as_tuple=False).view(-1)

                if same.numel() > 0:
                    positives_lists.append(embs[same])
                else:
                    positives_lists.append(embs.new_zeros((0, D)))  # 保持长度一致，loss 实现要容错

                if diff.numel() > 0:
                    negatives_lists.append(embs[diff])
                else:
                    negatives_lists.append(embs.new_zeros((0, D)))
            # # 5) 计算 loss & 优化
            # anchors = embs  # 所有样本都作为 anchor
            # loss = batch_circle_loss(anchors, positives_lists, negatives_lists)
            anchors = embs.float()
            anchors = F.normalize(anchors, p=2, dim=1)

            positives_lists = [p.float() for p in positives_lists]
            negatives_lists = [n.float() for n in negatives_lists]
            positives_lists = [F.normalize(p, p=2, dim=1) if p.numel()>0 else p for p in positives_lists]
            negatives_lists = [F.normalize(n, p=2, dim=1) if n.numel()>0 else n for n in negatives_lists]

            with torch.cuda.amp.autocast(enabled=False):
                loss = batch_circle_loss(anchors, positives_lists, negatives_lists)

            #=====================================================
            contrastive_optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(contrastive_optimizer)
            scaler.update()
            contrastive_loss = loss.detach()

        

            print("第二阶段")
            #=====================================================
            # 2) 训练：使用 encoder 和 classifier 训练模型
            encoder.eval()
            classifier.train()

            # 1) 准备：获取本次训练需要的所有唯一分子的索引（与前面对比学习里的 unique_graphs/unique_numbers 类似）
            #    这里我们从 number_matrix 中提取所有 unique id（str 形式）
            all_needed_numbers = []
            for row in number_matrix:
                for num in row:
                    all_needed_numbers.append(str(num))
            all_needed_numbers = list(dict.fromkeys(all_needed_numbers))  # 保持顺序且去重

            # 2) 如果之前已通过 matrixmake/precompute_all_graphs 得到图对象 -> 直接用这些图
            #    否则用 precompute_all_graphs(all_needed_numbers_as_ints, A) 并注意返回的 valid_indices
            #    下面用一种稳健方法：先试图用 graph_matrix 中已有对象（非 None），否则 fallback 去 precompute
            index_to_graph = {}
            # graph_matrix 的元素本来就是 graph 或 None（matrixmake 已填充）
            for i_row, row in enumerate(number_matrix):
                for j_col, num in enumerate(row):
                    key = str(num)
                    g = graph_matrix[i_row][j_col]
                    if g is not None and key not in index_to_graph:
                        index_to_graph[key] = g

            # 对仍然缺失的 key，调用 precompute_all_graphs 批量读取（以减少多次 IO）
            missing_keys = [k for k in all_needed_numbers if k not in index_to_graph]
            if len(missing_keys) > 0:
                # 尝试把 missing_keys 转为 int 索引列表（如果你的 key 是 'name4' 形式，请把它转换为 int）
                # 如果 key 原本就是 'name4'，下面 try/except 会尝试提取数字再调用 precompute
                try:
                    missing_indices = [int(k.replace('name','')) for k in missing_keys]
                except:
                    missing_indices = []
                    for k in missing_keys:
                        try:
                            missing_indices.append(int(k))
                        except:
                            pass
                if len(missing_indices) > 0:
                    # 批量预计算（分块以节省内存）
                    chunk = 1024
                    for s in range(0, len(missing_indices), chunk):
                        sub = missing_indices[s:s+chunk]
                        graphs, valid_idxs, _ = precompute_all_graphs(sub, A)  # 返回的 valid_idxs 对应 sub 里的原始索引
                        # 注意：precompute_all_graphs 在你的实现中返回的 valid_idxs 是实际有效的全局索引（int）
                        for g_idx, g in zip(valid_idxs, graphs):
                            key = str(g_idx)
                            if key in missing_keys and key not in index_to_graph:
                                index_to_graph[key] = g

            # 3) 现在应该有 index_to_graph 覆盖大部分 / 全部需要的分子；构建 index -> embedding 映射
            index_to_emb = {}
            # 批量构造 full_batch（分块以防显存不足）
            keys = list(index_to_graph.keys())
            batch_chunk = 1024  # 如果显存不足可以调小
            for s in range(0, len(keys), batch_chunk):
                subkeys = keys[s:s+batch_chunk]
                graphs_sub = [index_to_graph[k] for k in subkeys]
                # 确保 graph.x/edge_index 已经在 device，如果不是，移动它们（non_blocking 仅在 pinned memory 时有利）
                for g in graphs_sub:
                    if g.x.device != device:
                        g.x = g.x.to(device)
                        g.edge_index = g.edge_index.to(device)
                batch = Batch.from_data_list(graphs_sub).to(device)
                with torch.no_grad():
                    h = encoder(batch.x, batch.edge_index)
                    embs = global_mean_pool(h, batch.batch)  # [len(subkeys), D]
                # 存入映射
                for k, emb in zip(subkeys, embs):
                    index_to_emb[k] = emb.detach().clone()  # 保证 detach，训练 classifier 时不会回传给 encoder

            # 4) 构造用于分类器训练的 embeddings 列表与 labels 列表（和你原来保持一致：每个分子一个 embedding + label）
            all_embeddings = []
            all_classification_labels = []
            for row_idx in range(len(number_matrix)):
                row_nums = number_matrix[row_idx]
                # 为该行收集 embeddings 和 labels（跳过找不到 embedding 的分子）
                emb_list = []
                lab_list = []
                for j, num in enumerate(row_nums):
                    key = str(num)
                    if key not in index_to_emb:
                        # 可选：打印警告以便调试
                        # print(f"警告：索引 {key} 没有 embedding，跳过")
                        continue
                    emb_list.append(index_to_emb[key].unsqueeze(0))  # [1, D]
                    # 构造 label：这里你之前的逻辑是 row_labels[n+similer_nuber: ...] = 1 等
                    # 我们更稳健地使用 number_to_label(key) 或直接使用 active_values 映射（如果 active_values 使用原始 index）
                    # 下面尝试使用 number_to_label，如果失败则 fallback 用 active_values[int(key)]（需确保 key 是数字）
                    try:
                        lab = number_to_label(key)  # 你原先实现的函数
                    except Exception:
                        try:
                            lab = int(active_values[int(key)])
                        except Exception:
                            continue
                    lab_list.append(lab)
                if len(emb_list) == 0:
                    continue
                emb_tensor = torch.cat(emb_list, dim=0).to(device)  # [num_in_row, D]
                label_tensor = torch.tensor(lab_list, dtype=torch.long, device=device)
                all_embeddings.append(emb_tensor)
                all_classification_labels.append(label_tensor)

            # 5) 用这些预取的 embeddings 训练分类器（每个 emb_tensor 代表多个分子 embedding）
            classification_loss_total = 0.0
            correct_predictions = 0
            total_predictions = 0

            for embeddings, labels in zip(all_embeddings, all_classification_labels):
                embeddings = embeddings.detach().requires_grad_(False)
                classification_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    classifier_out = classifier(embeddings)  # [num_in_row, num_classes]
                    loss = criterion(classifier_out, labels)
                scaler.scale(loss).backward()
                scaler.step(classification_optimizer)
                scaler.update()

                classification_loss_total += loss.item()
                _, predicted = torch.max(classifier_out.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            classification_loss = classification_loss_total / (len(all_embeddings) if len(all_embeddings)>0 else 1)
            train_accuracy = 100 * correct_predictions / total_predictions if total_predictions > 0 else 0.0

            # 6) 评估测试集准确率（保留你原来的评估函数）
            test_accuracy = evaluate_test_accuracy_fixed(encoder, classifier, test_active_values, test_file_path, device)
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
            if train_accuracy > best_test_acc:
                best_test_acc = train_accuracy
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
                    
            # 早停条件
            if epoch >= 100:
                if test_accuracy >= early_stop_threshold or train_accuracy >= early_stop_threshold:
                    print(f"测试准确率达到{early_stop_threshold}%，触发早停！")
                    break
                        
                if patience_counter >= patience:
                    print(f"测试准确率连续{patience}轮未改善，触发早停！")
                    break
    
        elif len(active_1_name) > 600: 
            print("数据集过大使用老板")

            # # 使用修正的Circle Loss
            # batch_circle_loss = FixedBatchCircleLoss(margin=0.25, gamma=64).to(device)

            # batch_size = 128  # 减小批次大小确保稳定性

            # for batch_start in tqdm(range(0, len(active_1_name), batch_size)):
            #     batch_end = min(batch_start + batch_size, len(active_1_name))
            #     batch_indices = range(batch_start, batch_end)
                
            #     batch_anchors = []
            #     batch_positives_list = []
            #     batch_negatives_list = []
                
            #     # 预计算整个批次的嵌入
            #     all_row_embeddings = []
            #     for row_idx in batch_indices:
            #         row_graphs = graph_matrix[row_idx]
            #         if len(row_graphs) == 0:
            #             all_row_embeddings.append(None)
            #             continue

            #         batch = Batch.from_data_list(row_graphs).to(device)
            #         with torch.amp.autocast('cuda'):
            #             h = encoder(batch.x, batch.edge_index)
            #             row_embedding = global_mean_pool(h, batch.batch)
            #         all_row_embeddings.append(row_embedding)
                
            #     # 为每个样本构建正负样本
            #     for i, row_idx in enumerate(batch_indices):
            #         if all_row_embeddings[i] is None:
            #             continue
                        
            #         row_embedding = all_row_embeddings[i]
                    
            #         anchor = row_embedding[-1].unsqueeze(0)  # [1, embed_dim]
                    
            #         # 获取正样本：相似活性分子
            #         positive_start = n + similer_nuber
            #         positive_end = positive_start + similer_nuber
            #         positives = row_embedding[positive_start:positive_end]  # [num_pos, embed_dim]
                    
            #         # 获取负样本：相似非活性分子
            #         negatives = row_embedding[n:n+similer_nuber]  # [num_neg, embed_dim]
                    
            #         if positives.size(0) > 0 and negatives.size(0) > 0:
            #             batch_anchors.append(anchor.squeeze(0))  # [embed_dim]
            #             batch_positives_list.append(positives)
            #             batch_negatives_list.append(negatives)
                
            #     # 批量计算Circle Loss
            #     if batch_anchors:
            #         batch_anchors = torch.stack(batch_anchors)  # [batch_size, embed_dim]
                    
            #         with torch.amp.autocast('cuda'):
            #             contrastive_loss = batch_circle_loss(batch_anchors, batch_positives_list, batch_negatives_list)
                    
            #         # 优化步骤
            #         contrastive_optimizer.zero_grad(set_to_none=True)
            #         scaler.scale(contrastive_loss).backward()
                    
            #         # 添加梯度裁剪
            #         torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                    
            #         scaler.step(contrastive_optimizer)
            #         scaler.update()
            #     else:
            #         contrastive_loss = torch.tensor(0.0, device=device)
            # # 2. 分类学习阶段
            # # 2. 分类学习阶段（encoder 在 eval，embeddings 用 no_grad 获取）
            # encoder.eval()
            # classifier.train()

            # classification_loss_total = 0
            # correct_predictions = 0
            # total_predictions = 0

            # with torch.no_grad():
            #     all_embeddings = []
            #     all_classification_labels = []

            #     for row_idx in range(len(active_1_name)):
            #         row_graphs = graph_matrix[row_idx]
            #         if len(row_graphs) == 0:
            #             continue

            #         batch = Batch.from_data_list(row_graphs).to(device)
            #         # encoder 前向可以用 autocast（没有梯度）
            #         with torch.amp.autocast('cuda'):
            #             h = encoder(batch.x, batch.edge_index)
            #             row_embedding = global_mean_pool(h, batch.batch)

            #         all_embeddings.append(row_embedding)
            #         num_molecules = len(row_graphs)
            #         row_labels = torch.zeros(num_molecules, dtype=torch.long, device=device)
            #         row_labels[n+similer_nuber:n+similer_nuber+n+similer_nuber] = 1
            #         row_labels[-1] = 1
            #         all_classification_labels.append(row_labels)

            # # 阶段2: 分类学习 - 只训练分类器
            # # 用 scaler 对 classifier 进行训练（对每个 embeddings 单独优化）
            # for embeddings, labels in zip(all_embeddings, all_classification_labels):
            #     # 确保嵌入不计算梯度（双重保险）
            #     embeddings = embeddings.detach().requires_grad_(False)
            #     classification_optimizer.zero_grad(set_to_none=True)

            #     # classifier 前向和 loss 计算使用 autocast
            #     with torch.amp.autocast('cuda'):
            #         classifier_out = classifier(embeddings)
            #         loss = criterion(classifier_out, labels)

            #     # 缩放 loss 并反向
            #     scaler.scale(loss).backward()
            #     scaler.step(classification_optimizer)
            #     scaler.update()

            #     classification_loss_total += loss.item()

            #     # 计算准确率（注意这里 classifier_out 在 autocast 下，但 .data/.argmax 可以用）
            #     _, predicted = torch.max(classifier_out.data, 1)
            #     total_predictions += labels.size(0)
            #     correct_predictions += (predicted == labels).sum().item()

            
            # # 计算平均分类损失和训练准确率
            # classification_loss = classification_loss_total / len(all_embeddings) if all_embeddings else 0
            # train_accuracy = 100 * correct_predictions / total_predictions if total_predictions > 0 else 0
            
            # # 3. 评估测试集准确率
            # test_accuracy = evaluate_test_accuracy_fixed(encoder, classifier, test_active_values, test_file_path, device)
            
            # # 4. 记录指标
            # contrastive_losses.append(contrastive_loss.item())
            # classification_losses.append(classification_loss)
            # train_accuracies.append(train_accuracy)
            # test_accuracies.append(test_accuracy)
            
            
            # contrastive_scheduler.step()
            # classification_scheduler.step()

            
            # print(f"Epoch [{epoch + 1}/{total_epochs}]:")
            # print(f"  对比损失: {contrastive_loss.item():.4f}")
            # print(f"  分类损失: {classification_loss:.4f}")
            # print(f"  训练准确率: {train_accuracy:.2f}%")
            # print(f"  测试准确率: {test_accuracy:.2f}%")
            # if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
            #     print("警告：对比损失出现NaN或Inf！")
            #     break
            # # 6. 早停检查
            # if train_accuracy > best_test_acc:
            #     best_test_acc = train_accuracy
            #     patience_counter = 0
            #     # 保存最佳模型
            #     # best_model_path = A.replace('train', 'pth') +'randon_number-'+str(n)+'similer_nuber-'+str(n)+ '_best_model.pth'
            #     best_model_path = savepth
            #     torch.save({
            #         'encoder_state_dict': unwrap_state_dict(encoder),
            #         'classifier_state_dict': unwrap_state_dict(classifier),
            #         'epoch': epoch,
            #         'test_accuracy': test_accuracy
            #     }, best_model_path)
            #     print(f"  新的最佳模型已保存，测试准确率: {best_test_acc:.2f}%")
            # else:
            #     patience_counter += 1
            #     print(f"  早停计数器: {patience_counter}/{patience}")
            
            # # 早停条件
            # if epoch >= 100:
            #     if test_accuracy >= early_stop_threshold or train_accuracy >= early_stop_threshold:
            #         print(f"测试准确率达到{early_stop_threshold}%，触发早停！")
            #         break
                
            #     if patience_counter >= patience:
            #         print(f"测试准确率连续{patience}轮未改善，触发早停！")
            #         break
            # 使用修正的Circle Loss
            batch_circle_loss = FixedBatchCircleLoss(margin=0.25, gamma=64).to(device)

            batch_size = 64  # 减小批次大小确保稳定性

            # ---------- 替换开始：仅替换这里的对比学习循环部分 ----------

            # 1) 先把 graph_matrix 与 number_matrix 同步展平（保持顺序一致）
            flat_graphs = [g for row in graph_matrix for g in row]
            flat_numbers = [nm for row in number_matrix for nm in row]  # number_matrix 与 graph_matrix 一一对应

            if len(flat_graphs) != len(flat_numbers):
                raise RuntimeError(f"graph_matrix 与 number_matrix 长度不一致: {len(flat_graphs)} vs {len(flat_numbers)}")

            # 2) 用 number (或 name 字符串，如 'name4') 作为去重 key -> 构建 unique 列表
            unique_map = {}
            unique_graphs = []
            unique_numbers = []
            unique_labels = []

            for g, num in zip(flat_graphs, flat_numbers):
                # 确保 num 可哈希为字符串（统一）
                key = str(num)
                if key in unique_map:
                    continue
                idx = len(unique_graphs)
                unique_map[key] = idx
                unique_graphs.append(g)
                unique_numbers.append(key)
                unique_labels.append(number_to_label(key))

            if len(unique_graphs) < 2:
                raise RuntimeError(f"去重后分子太少：{len(unique_graphs)}，无法做对比学习。")
            jieyue = len(unique_graphs)+jieyue
            # 3) 建立 label -> 索引 列表，便于采样正/负
            label_to_indices = {}
            for i, lab in enumerate(unique_labels):
                label_to_indices.setdefault(int(lab), []).append(i)

            

            # ---------- 全量一次性 forward（替换你原来对比学习那段） ----------
            # encoder.train()
            # batch_circle_loss = FixedBatchCircleLoss(margin=0.25, gamma=64).to(device)

            # 1) 把所有去重图合成一个大 Batch（unique_graphs 在前面已构造）
            full_batch = Batch.from_data_list(unique_graphs).to(device)

            # 2) 前向，得到 graph-level embeddings（不要 detach）
            with torch.amp.autocast('cuda'):
                h = encoder(full_batch.x, full_batch.edge_index)   # node-level
                embs = global_mean_pool(h, full_batch.batch)       # [N, D]

            # 3) 归一化（可选，但通常能稳定对比学习）
            embs = F.normalize(embs, p=2, dim=1)

            # 4) 构造每个 anchor 的 positives/negatives（按 unique_labels）
            labels_tensor = torch.tensor([int(x) for x in unique_labels], device=device)

            positives_lists = []
            negatives_lists = []
            N, D = embs.size()
            for i in range(N):
                same = (labels_tensor == labels_tensor[i]).nonzero(as_tuple=False).view(-1)
                same = same[same != i]  # 删掉 self
                diff = (labels_tensor != labels_tensor[i]).nonzero(as_tuple=False).view(-1)

                if same.numel() > 0:
                    positives_lists.append(embs[same])
                else:
                    positives_lists.append(embs.new_zeros((0, D)))  # 保持长度一致，loss 实现要容错

                if diff.numel() > 0:
                    negatives_lists.append(embs[diff])
                else:
                    negatives_lists.append(embs.new_zeros((0, D)))
            # # 5) 计算 loss & 优化
            # anchors = embs  # 所有样本都作为 anchor
            # loss = batch_circle_loss(anchors, positives_lists, negatives_lists)
            anchors = embs.float()
            anchors = F.normalize(anchors, p=2, dim=1)

            positives_lists = [p.float() for p in positives_lists]
            negatives_lists = [n.float() for n in negatives_lists]
            positives_lists = [F.normalize(p, p=2, dim=1) if p.numel()>0 else p for p in positives_lists]
            negatives_lists = [F.normalize(n, p=2, dim=1) if n.numel()>0 else n for n in negatives_lists]

            with torch.cuda.amp.autocast(enabled=False):
                loss = batch_circle_loss(anchors, positives_lists, negatives_lists)

            #=====================================================
            contrastive_optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(contrastive_optimizer)
            scaler.update()
            contrastive_loss = loss.detach()

        

            print("第二阶段")
            #=====================================================
            # 2) 训练：使用 encoder 和 classifier 训练模型
            encoder.eval()
            classifier.train()

            # 1) 准备：获取本次训练需要的所有唯一分子的索引（与前面对比学习里的 unique_graphs/unique_numbers 类似）
            #    这里我们从 number_matrix 中提取所有 unique id（str 形式）
            all_needed_numbers = []
            for row in number_matrix:
                for num in row:
                    all_needed_numbers.append(str(num))
            all_needed_numbers = list(dict.fromkeys(all_needed_numbers))  # 保持顺序且去重

            # 2) 如果之前已通过 matrixmake/precompute_all_graphs 得到图对象 -> 直接用这些图
            #    否则用 precompute_all_graphs(all_needed_numbers_as_ints, A) 并注意返回的 valid_indices
            #    下面用一种稳健方法：先试图用 graph_matrix 中已有对象（非 None），否则 fallback 去 precompute
            index_to_graph = {}
            # graph_matrix 的元素本来就是 graph 或 None（matrixmake 已填充）
            for i_row, row in enumerate(number_matrix):
                for j_col, num in enumerate(row):
                    key = str(num)
                    g = graph_matrix[i_row][j_col]
                    if g is not None and key not in index_to_graph:
                        index_to_graph[key] = g

            # 对仍然缺失的 key，调用 precompute_all_graphs 批量读取（以减少多次 IO）
            missing_keys = [k for k in all_needed_numbers if k not in index_to_graph]
            if len(missing_keys) > 0:
                # 尝试把 missing_keys 转为 int 索引列表（如果你的 key 是 'name4' 形式，请把它转换为 int）
                # 如果 key 原本就是 'name4'，下面 try/except 会尝试提取数字再调用 precompute
                try:
                    missing_indices = [int(k.replace('name','')) for k in missing_keys]
                except:
                    missing_indices = []
                    for k in missing_keys:
                        try:
                            missing_indices.append(int(k))
                        except:
                            pass
                if len(missing_indices) > 0:
                    # 批量预计算（分块以节省内存）
                    chunk = 1024
                    for s in range(0, len(missing_indices), chunk):
                        sub = missing_indices[s:s+chunk]
                        graphs, valid_idxs, _ = precompute_all_graphs(sub, A)  # 返回的 valid_idxs 对应 sub 里的原始索引
                        # 注意：precompute_all_graphs 在你的实现中返回的 valid_idxs 是实际有效的全局索引（int）
                        for g_idx, g in zip(valid_idxs, graphs):
                            key = str(g_idx)
                            if key in missing_keys and key not in index_to_graph:
                                index_to_graph[key] = g

            # 3) 现在应该有 index_to_graph 覆盖大部分 / 全部需要的分子；构建 index -> embedding 映射
            index_to_emb = {}
            # 批量构造 full_batch（分块以防显存不足）
            keys = list(index_to_graph.keys())
            batch_chunk = 128  # 如果显存不足可以调小
            for s in range(0, len(keys), batch_chunk):
                subkeys = keys[s:s+batch_chunk]
                graphs_sub = [index_to_graph[k] for k in subkeys]
                # 确保 graph.x/edge_index 已经在 device，如果不是，移动它们（non_blocking 仅在 pinned memory 时有利）
                for g in graphs_sub:
                    if g.x.device != device:
                        g.x = g.x.to(device)
                        g.edge_index = g.edge_index.to(device)
                batch = Batch.from_data_list(graphs_sub).to(device)
                with torch.no_grad():
                    h = encoder(batch.x, batch.edge_index)
                    embs = global_mean_pool(h, batch.batch)  # [len(subkeys), D]
                # 存入映射
                for k, emb in zip(subkeys, embs):
                    index_to_emb[k] = emb.detach().cpu().clone()  # 保证 detach，训练 classifier 时不会回传给 encoder

            # 4) 构造用于分类器训练的 embeddings 列表与 labels 列表（和你原来保持一致：每个分子一个 embedding + label）
            all_embeddings = []
            all_classification_labels = []
            for row_idx in range(len(number_matrix)):
                row_nums = number_matrix[row_idx]
                # 为该行收集 embeddings 和 labels（跳过找不到 embedding 的分子）
                emb_list = []
                lab_list = []
                for j, num in enumerate(row_nums):
                    key = str(num)
                    if key not in index_to_emb:
                        # 可选：打印警告以便调试
                        # print(f"警告：索引 {key} 没有 embedding，跳过")
                        continue
                    emb_list.append(index_to_emb[key].unsqueeze(0))  # [1, D]
                    # 构造 label：这里你之前的逻辑是 row_labels[n+similer_nuber: ...] = 1 等
                    # 我们更稳健地使用 number_to_label(key) 或直接使用 active_values 映射（如果 active_values 使用原始 index）
                    # 下面尝试使用 number_to_label，如果失败则 fallback 用 active_values[int(key)]（需确保 key 是数字）
                    try:
                        lab = number_to_label(key)  # 你原先实现的函数
                    except Exception:
                        try:
                            lab = int(active_values[int(key)])
                        except Exception:
                            continue
                    lab_list.append(lab)
                if len(emb_list) == 0:
                    continue
                emb_tensor = torch.cat(emb_list, dim=0).to(device)  # [num_in_row, D]
                label_tensor = torch.tensor(lab_list, dtype=torch.long, device=device)
                all_embeddings.append(emb_tensor)
                all_classification_labels.append(label_tensor)

            # 5) 用这些预取的 embeddings 训练分类器（每个 emb_tensor 代表多个分子 embedding）
            classification_loss_total = 0.0
            correct_predictions = 0
            total_predictions = 0

            for embeddings, labels in zip(all_embeddings, all_classification_labels):
                embeddings = embeddings.detach().requires_grad_(False)
                classification_optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    classifier_out = classifier(embeddings)  # [num_in_row, num_classes]
                    loss = criterion(classifier_out, labels)
                scaler.scale(loss).backward()
                scaler.step(classification_optimizer)
                scaler.update()

                classification_loss_total += loss.item()
                _, predicted = torch.max(classifier_out.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            classification_loss = classification_loss_total / (len(all_embeddings) if len(all_embeddings)>0 else 1)
            train_accuracy = 100 * correct_predictions / total_predictions if total_predictions > 0 else 0.0

            # 6) 评估测试集准确率（保留你原来的评估函数）
            test_accuracy = evaluate_test_accuracy_fixed(encoder, classifier, test_active_values, test_file_path, device)
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
            if train_accuracy > best_test_acc:
                best_test_acc = train_accuracy
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
                    
            # 早停条件
            if epoch >= 100:
                if test_accuracy >= early_stop_threshold or train_accuracy >= early_stop_threshold:
                    print(f"测试准确率达到{early_stop_threshold}%，触发早停！")
                    break
                        
                if patience_counter >= patience:
                    print(f"测试准确率连续{patience}轮未改善，触发早停！")
                    break
    #===================================================================================
    print("训练完成！")
    print(f"最佳测试准确率: {best_test_acc:.2f}%")
    
    # 保存最终模型
    final_model_path = best_model_path
    
    print(f"最终模型已保存到{final_model_path}")
    
    # 绘制训练进度
    plot_training_progress(contrastive_losses, classification_losses, train_accuracies, test_accuracies, len(contrastive_losses), A, final_model_path)
    
    return epoch, all_reduce_molecule_numbers, len(active_values), test_accuracy
    #===================================================================================
if __name__ == '__main__':
    for chaocan in range(25):
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
            #这里是模型方式，0是20% 1是110
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
                # "train/BBBP",
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
                    baifenshu=min(len(active_0_name), len(active_1_name))//10
                            # if chaocan //2 == 0:
                            #     n = baifenshu * 4
                            #     similer_nuber = baifenshu * (chaocan + 4 )
                            # elif chaocan //2 == 1:
                            #     n = baifenshu * 5
                            #     similer_nuber = baifenshu * (chaocan + 2 )
                    randon_numberbili = (chaocan//5+1)
                    similer_numbbili = (chaocan %5+1)
                    print(f"随机的比例为{randon_numberbili*10}%,相似的比例为{similer_numbbili*10}%")
                    n = baifenshu * randon_numberbili
                    similer_nuber = baifenshu * similer_numbbili
                elif daimafangshi == 1:
                    n = similer_nuber=88
                aooepoch=aooepoch+1
                print(f"开始训练模型{file}，随机个数：{n}，相似个数：{similer_nuber},总体训练次数：{aooepoch}")
                if daimafangshi == 1:
                    print(f"这里为固定参数110")
                    epoch, all_reduce_molecule_numbers, listmo, test_accuracy = TRAIN(file, n, similer_nuber, bili,sim_baifenbi=0, randon_baifenbi=0,savepth=file.replace('train', 'pth') +'randon_number-'+'110'+'similer_nuber-'+'110'+ '_best_model.pth')
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