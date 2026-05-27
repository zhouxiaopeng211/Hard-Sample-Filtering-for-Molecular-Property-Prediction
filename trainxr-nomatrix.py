import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
from torch.nn import Linear
import os
import time
from tqdm import tqdm
import re
import random
import number
import shutil
from pathlib import Path
import torch.optim as opt
import torch as T
import pandas as pd
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from multiprocessing import Pool
import SDF_dispose
from rdkit import Chem
from rdkit import RDLogger
import test
from torch_geometric.utils import dropout_edge
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.metrics import roc_auc_score
import gc
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# 1. 升级：预计算图特征（加入 ECFP 指纹）与 trainpro.py 保持一致
# =====================================================================
def precompute_all_graphs(active_list, A, batch_size=256):
    supplier = Chem.SDMolSupplier(A+".sdf")
    graphs = []
    listuseful = []
    mol_names = []
    
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
                if mol is None: continue

                graph = SDF_dispose.molecule_to_pyg_graph(mol)
                
                # 计算并绑定 ECFP 指纹
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fp_array = np.zeros((1,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, fp_array)
                graph.ecfp = torch.tensor(fp_array, dtype=torch.float32).view(1, -1)
                
                if mol.HasProp("name"):
                    mol_name = mol.GetProp("name")
                    batch_mol_names.append(mol_name)

                batch_graphs.append(graph)
                batch_listuseful.append(idx)

            except Exception as e:
                continue
        
        graphs.extend(batch_graphs)
        listuseful.extend(batch_listuseful)
        mol_names.extend(batch_mol_names)

    return graphs, listuseful, mol_names

def save_vector_to_txt(vector, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in vector:
            file.write(f"{item}\n")

def extract_digits(s):
    return int(re.sub(r'^\D+', '', s))

# =====================================================================
# 2. 升级：GNN 编码器（加入动态扰动 drop_rate）
# =====================================================================
class GNN(nn.Module):
    def __init__(self, INPUT):
        super(GNN, self).__init__()
        self.INPUT = INPUT
        self.CHECKPOINT_DIR = self.INPUT["CHECKPOINT_DIR"]
        self.CHECKPOINT_FILE = os.path.join(self.CHECKPOINT_DIR, self.INPUT["NAME"])
        self.SIZE_LAYERS = self.INPUT["SIZE_LAYERS"]
        self.initial_conv = GATConv(self.SIZE_LAYERS[0], self.SIZE_LAYERS[1])
        self.conv1 = GATConv(self.SIZE_LAYERS[1], self.SIZE_LAYERS[2])
        self.conv2 = GATConv(self.SIZE_LAYERS[2], self.SIZE_LAYERS[2])
        self.linear = Linear(self.SIZE_LAYERS[2], self.SIZE_LAYERS[3])
        self.optimizer = opt.Adam(self.parameters(), lr=self.INPUT["LR"])
        self.criterion = nn.MSELoss()

    def forward(self, x, edge_index, drop_rate=0.0):  
        if drop_rate > 0.0 and self.training:
            edge_index, _ = dropout_edge(edge_index, p=drop_rate, force_undirected=True)
            num_nodes = x.size(0)
            mask_indices = torch.rand(num_nodes, device=x.device) < drop_rate
            x = x.clone()
        out = F.relu(self.initial_conv(x, edge_index=edge_index))
        out = F.relu(self.conv1(out, edge_index=edge_index))
        out = F.relu(self.conv2(out, edge_index=edge_index))
        return self.linear(out)

# =====================================================================
# 3. 升级：双塔多模态分类网络 (引入 ECFP)
# =====================================================================
class ClassifierNetwork(nn.Module):
    def __init__(self, gnn_dim=32, ecfp_dim=2048, num_classes=2):
        super(ClassifierNetwork, self).__init__()
        self.ecfp_branch = nn.Sequential(
            nn.Linear(ecfp_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3)
        )
        fusion_dim = gnn_dim + 64
        self.final_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, gnn_emb, ecfp_emb):
        ecfp_features = self.ecfp_branch(ecfp_emb)
        combined_features = torch.cat([gnn_emb, ecfp_features], dim=1)
        return self.final_classifier(combined_features)

class FixedBatchCircleLoss(nn.Module):
    def __init__(self, margin=0.25, gamma=64):
        super(FixedBatchCircleLoss, self).__init__()
        self.margin = margin; self.gamma = gamma
        self.O_p = 1 + margin; self.O_n = -margin
        self.delta_p = 1 - margin; self.delta_n = margin
        
    def forward(self, anchors, positives_list, negatives_list):
        batch_size = anchors.size(0)
        total_loss = 0.0
        valid_count = 0
        for i in range(batch_size):
            anchor = anchors[i].unsqueeze(0)
            positives = positives_list[i]
            negatives = negatives_list[i]
            if positives.size(0) == 0 or negatives.size(0) == 0: continue
                
            anchor_norm = F.normalize(anchor, p=2, dim=1)
            positives_norm = F.normalize(positives, p=2, dim=1)
            negatives_norm = F.normalize(negatives, p=2, dim=1)
            
            s_p = torch.mm(anchor_norm, positives_norm.t()).squeeze(0)
            s_n = torch.mm(anchor_norm, negatives_norm.t()).squeeze(0)
            s_p = torch.clamp(s_p, -1.0 + 1e-7, 1.0 - 1e-7)
            s_n = torch.clamp(s_n, -1.0 + 1e-7, 1.0 - 1e-7)
            
            with torch.no_grad():
                alpha_p = torch.clamp(self.O_p - s_p, min=0)
                alpha_n = torch.clamp(s_n - self.O_n, min=0)
            
            logit_p = -self.gamma * alpha_p * (self.delta_p - s_p)
            logit_n = self.gamma * alpha_n * (s_n - self.delta_n)
            
            loss = F.softplus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
            total_loss += loss
            valid_count += 1
        return total_loss / max(valid_count, 1)

# =====================================================================
# 4. 升级：评估函数 (支持双塔并计算 AUC)
# =====================================================================
def evaluate_test_accuracy_fixed(test_graphs_cache, test_valid_cache, encoder, classifier, test_active_values, device, batch_size=256):
    encoder.eval()
    classifier.eval()
    correct = 0; total = 0
    all_trues = []; all_probs = []

    if len(test_graphs_cache) == 0: return 0.0, 0.5

    with torch.no_grad():
        for start in range(0, len(test_graphs_cache), batch_size):
            end = min(start + batch_size, len(test_graphs_cache))
            sub_graphs = test_graphs_cache[start:end]
            batch = Batch.from_data_list(sub_graphs).to(device)

            h = encoder(batch.x, batch.edge_index, drop_rate=0.0)
            gnn_embeddings = global_mean_pool(h, batch.batch)
            ecfp_embeddings = batch.ecfp.view(-1, 2048)

            out = classifier(gnn_embeddings, ecfp_embeddings)
            _, predicted = torch.max(out.data, 1)
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            predicted = predicted.cpu()

            for i_local, global_idx in enumerate(test_valid_cache[start:end]):
                if global_idx < len(test_active_values):
                    true_label_str = test_active_values[global_idx]
                    pred_label_str = str(predicted[i_local].item())
                    total += 1
                    if pred_label_str == true_label_str: correct += 1
                    all_trues.append(int(true_label_str))
                    all_probs.append(probs[i_local])

            del batch, h, gnn_embeddings, ecfp_embeddings, out, predicted
            gc.collect()
            torch.cuda.empty_cache()

    accuracy = 100 * correct / total if total > 0 else 0.0
    try: auc_score = roc_auc_score(all_trues, all_probs)
    except: auc_score = 0.5
    
    print(f"测试集准确率: {accuracy:.2f}%, 测试集 ROC-AUC: {auc_score:.4f}")
    return accuracy, auc_score

def plot_training_progress(contrastive_losses, classification_losses, train_accuracies, test_accuracies, total_epochs, A, final_model_path):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1); plt.plot(range(total_epochs), contrastive_losses, color="purple"); plt.title("Contrastive Loss")
    plt.subplot(1, 4, 2); plt.plot(range(total_epochs), classification_losses, color="red"); plt.title("Classification Loss")
    plt.subplot(1, 4, 3); plt.plot(range(total_epochs), train_accuracies, color="blue"); plt.title("Train Accuracy")
    plt.subplot(1, 4, 4); plt.plot(range(total_epochs), test_accuracies, color="green"); plt.title("Test AUC")
    final_model_path = final_model_path.replace('.pth',".jpg")
    plt.savefig(final_model_path, dpi=300); plt.close()

def unwrap_state_dict(module):
    try:
        orig = getattr(module, "_orig_mod", None)
        if orig is not None: return orig.state_dict()
    except: pass
    return module.state_dict()

# =====================================================================
# 5. 消融核心逻辑：批次内 Random-CL 训练循环
# =====================================================================
def ABLATION_TRAIN(A, bili):
    scaler = torch.cuda.amp.GradScaler()
    RDLogger.DisableLog('rdApp.*')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    active_values, active_0_name, active_1_name = number.extract_active_property(A+".sdf")
    
    all_molecules = []
    all_labels = []
    for name in active_1_name:
        all_molecules.append(extract_digits(name))
        all_labels.append(1)
    for name in active_0_name:
        all_molecules.append(extract_digits(name))
        all_labels.append(0)
    
    test_file_path = A.replace('train/'+str(bili[0])+str(bili[1])+str(bili[2]), 'validation/')
    test_active_values, _, _ = number.extract_active_property(test_file_path+ ".sdf")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 动态扰动及多阶段超参数 (严格对齐 trainpro.py)
    total_epochs = 500
    perturb_start = 100
    perturb_ramp_end = 150
    target_drop_rate = 0.25
    second_phase_start = 200
    
    lr = 1e-3
    best_test_auc = 0.0
    patience_counter = 0
    patience = 50
    
    contrastive_losses = []; classification_losses = []; train_accuracies = []; test_accuracies = []

    graph,_,_ = precompute_all_graphs([0], A)
    num_node_features = graph[0].x.size(1)
    
    # 初始化模型
    gnn_input = {"LR": lr, "NAME": "GNN_Encoder", "CHECKPOINT_DIR": "", 
                 "SIZE_LAYERS": [num_node_features, 64, 64, 32]}
    encoder = GNN(gnn_input).to(device)
    classifier = ClassifierNetwork(gnn_dim=32, ecfp_dim=2048, num_classes=2).to(device)
    
    contrastive_optimizer = optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-5)
    classification_optimizer = optim.AdamW(classifier.parameters(), lr=1e-5, weight_decay=1e-5)
    
    # 计算加权交叉熵
    w0 = min(len(active_values) / (2 * len(active_0_name)), 10.0) if len(active_0_name) > 0 else 1.0
    w1 = min(len(active_values) / (2 * len(active_1_name)), 10.0) if len(active_1_name) > 0 else 1.0
    class_weights_tensor = torch.tensor([w0, w1], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    contrastive_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(contrastive_optimizer, T_max=total_epochs, eta_min=1e-5)
    classification_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(classification_optimizer, T_max=total_epochs, eta_min=1e-6)

    # 预加载测试集
    all_test_indices = list(range(len(test_active_values)))
    test_graphs_cache, test_valid_cache, _ = precompute_all_graphs(all_test_indices, test_file_path)

    print("开始消融实验 (Random-CL) 训练...")
    batch_circle_loss = FixedBatchCircleLoss(margin=0.25, gamma=64).to(device)
    
    for epoch in range(total_epochs):
        # 动态图扰动策略
        if epoch < perturb_start: current_drop_rate = 0.0
        elif perturb_start <= epoch < perturb_ramp_end: current_drop_rate = ((epoch - perturb_start) / (perturb_ramp_end - perturb_start)) * target_drop_rate
        else: current_drop_rate = target_drop_rate

        print(f'\nEpoch: {epoch+1}/{total_epochs}')
        indices = torch.randperm(len(all_molecules))
        
        # ======================================================================
        # 阶段 1：第一阶段对比学习 (批次内纯随机正负采样 Random-CL)
        # ======================================================================
        avg_contrastive_loss = 0.0
        if epoch < second_phase_start:
            encoder.train()
            classifier.eval()
            contrastive_loss_total = 0
            num_batches = 0
            batch_size = 512
            
            for batch_start in tqdm(range(0, len(indices), batch_size), desc="Stage 1: Random-CL"):
                batch_end = min(batch_start + batch_size, len(indices))
                batch_indices = indices[batch_start:batch_end]
                if len(all_molecules) == len(all_labels):
                    print("对齐数据集和标签集")
                else:
                    print("数据集和标签集不匹配")
                batch_molecules = [all_molecules[i] for i in batch_indices]
                batch_labels = [all_labels[i] for i in batch_indices]
                
                batch_graphs, _, _ = precompute_all_graphs(batch_molecules, A)
                batch_molecules = [all_molecules[i] for i in batch_indices]
                batch_labels = [all_labels[i] for i in batch_indices]
                
                batch_graphs, _, _ = precompute_all_graphs(batch_molecules, A)
                if len(batch_graphs) == 0: continue
                batch_data = Batch.from_data_list(batch_graphs).to(device)
                
                with torch.amp.autocast('cuda'):
                    h = encoder(batch_data.x, batch_data.edge_index, drop_rate=current_drop_rate)
                    embeddings = global_mean_pool(h, batch_data.batch)
                
                # 在 Batch 内部构建 Random Positive/Negative
                batch_anchors = []; batch_positives_list = []; batch_negatives_list = []
                for i in range(len(embeddings)):
                    anchor = embeddings[i].unsqueeze(0)
                    current_label = batch_labels[i]
                    
                    # 随机正样本 (同标签)
                    positive_indices = [j for j in range(len(embeddings)) if batch_labels[j] == current_label and j != i]
                    positives = embeddings[positive_indices] if positive_indices else torch.tensor([]).to(device)
                    
                    # 随机负样本 (异标签)
                    negative_indices = [j for j in range(len(embeddings)) if batch_labels[j] != current_label]
                    negatives = embeddings[negative_indices] if negative_indices else torch.tensor([]).to(device)
                    
                    if positives.size(0) > 0 and negatives.size(0) > 0:
                        batch_anchors.append(anchor.squeeze(0))
                        batch_positives_list.append(positives)
                        batch_negatives_list.append(negatives)
                
                if batch_anchors:
                    batch_anchors = torch.stack(batch_anchors)
                    with torch.amp.autocast('cuda'):
                        contrastive_loss = batch_circle_loss(batch_anchors, batch_positives_list, batch_negatives_list)
                    
                    contrastive_optimizer.zero_grad(set_to_none=True)
                    scaler.scale(contrastive_loss).backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                    scaler.step(contrastive_optimizer)
                    scaler.update()
                    
                    contrastive_loss_total += contrastive_loss.item()
                    num_batches += 1
            
            avg_contrastive_loss = contrastive_loss_total / max(num_batches, 1)

        # ======================================================================
        # 阶段 2：端到端分类微调 (双塔融合图特征与 ECFP)
        # ======================================================================
        avg_classification_loss = 0.0
        train_accuracy = 0.0
        
        if epoch >= second_phase_start:
            encoder.train()
            classifier.train()
            classification_loss_total = 0
            correct_predictions = 0
            total_predictions = 0
            cls_batch_size = 128
            
            for batch_start in tqdm(range(0, len(indices), cls_batch_size), desc="Stage 2: Fine-Tuning"):
                batch_end = min(batch_start + cls_batch_size, len(indices))
                batch_indices = indices[batch_start:batch_end]
                
                batch_molecules = [all_molecules[i] for i in batch_indices]
                batch_labels = torch.tensor([all_labels[i] for i in batch_indices], dtype=torch.long, device=device)
                
                batch_graphs, _, _ = precompute_all_graphs(batch_molecules, A)
                if len(batch_graphs) < 2: continue
                batch_data = Batch.from_data_list(batch_graphs).to(device)
                
                classification_optimizer.zero_grad(set_to_none=True)
                contrastive_optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    # 1. 提取图特征
                    h = encoder(batch_data.x, batch_data.edge_index, drop_rate=current_drop_rate)
                    gnn_embeddings = global_mean_pool(h, batch_data.batch)
                    
                    # 2. 提取 ECFP 指纹特征
                    ecfp_embeddings = batch_data.ecfp.view(-1, 2048)
                    
                    # 3. 双塔融合分类
                    outputs = classifier(gnn_embeddings, ecfp_embeddings)
                    classification_loss = criterion(outputs, batch_labels)
                
                scaler.scale(classification_loss).backward()
                scaler.step(classification_optimizer)
                scaler.step(contrastive_optimizer) # 联合更新
                scaler.update()

                classification_loss_total += classification_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += batch_labels.size(0)
                correct_predictions += (predicted == batch_labels).sum().item()

            avg_classification_loss = classification_loss_total / max((len(indices) // cls_batch_size), 1)
            train_accuracy = 100 * correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # === 评估与早停 (基于 AUC) ===
        test_accuracy, test_auc = evaluate_test_accuracy_fixed(test_graphs_cache, test_valid_cache, encoder, classifier, test_active_values, device)
        
        contrastive_losses.append(avg_contrastive_loss)
        classification_losses.append(avg_classification_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_auc * 100) # 画图存 AUC
        
        contrastive_scheduler.step()
        classification_scheduler.step()

        print(f"  对比损失: {avg_contrastive_loss:.4f} | 分类损失: {avg_classification_loss:.4f}")
        print(f"  训练准确率: {train_accuracy:.2f}% | 测试集 ROC-AUC: {test_auc:.4f}")
        
        # 早停检查 (以 AUC 为标准)
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            patience_counter = 0
            best_model_path = A.replace('train', 'pth') + 'xr_nomatrix.pth'
            torch.save({
                'encoder_state_dict': unwrap_state_dict(encoder),
                'classifier_state_dict': unwrap_state_dict(classifier),
                'epoch': epoch,
                'test_auc': test_auc
            }, best_model_path)
            print(f"  [*] 新的最佳模型已保存，测试 AUC: {best_test_auc:.4f}")
        else:
            patience_counter += 1
            if epoch >= second_phase_start:
                print(f"  早停计数器: {patience_counter}/{patience}")
    
    print("消融实验训练完成！")
    print(f"最佳测试 AUC: {best_test_auc:.4f}")
    
    plot_training_progress(contrastive_losses, classification_losses, train_accuracies, test_accuracies, len(contrastive_losses), A, best_model_path)
    return epoch, [], len(active_values), best_test_auc

if __name__ == '__main__':
    bili = [8, 1, 1]
    multiprocessing.set_start_method('spawn', force=True)
    
    filesss = [
        'train/BBBP',
        'train/bace',
        "train/clintox",
        'train/HIV',
    ]
    
    files = [fil.replace('train/','ours/train/'+str(bili[0])+str(bili[1])+str(bili[2])) for fil in filesss]
    
    print("开始执行 Random-CL (w/o HSM) 消融实验...")
    txt = [f'训练集,验证集,测试集比例为:{bili[0]}:{bili[1]}:{bili[2]}\n']
    
    for file in files:
        print(f"\n[{file}] 开始消融实验训练...")
        epoch, _, _, test_auc = ABLATION_TRAIN(file, bili)
        
        wenben = f"文件名：{file} | 变体: Random-CL (w/o HSM) | 最佳测试集 AUC: {test_auc:.4f}"
        txt.append(wenben)
        
        out_filename = file.replace('train', 'out') + "_ablation_random_cl.txt"
        save_vector_to_txt(txt, out_filename)