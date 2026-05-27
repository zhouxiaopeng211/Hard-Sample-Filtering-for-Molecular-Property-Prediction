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
import torch.cuda.amp as amp
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import dropout_edge
import gc
import math
import multiprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================================================
# 1. 核心数据加载工具 (带 ECFP 指纹)
# =====================================================================
import SDF_dispose # 确保你的同级目录下有 SDF_dispose.py

def extract_digits(s):
    return int(re.sub(r'^\D+', '', s))

def extract_active_property(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file)
    active_data = []; active_0_name = []; active_1_name = []
    for idx, mol in enumerate(supplier):
        if mol is None: continue
        if mol.HasProp("Active"):
            name = mol.GetProp("name")
            active = mol.GetProp("Active")
            active_data.append(active)
            if active == '0': active_0_name.append(name)
            elif active == '1': active_1_name.append(name)
    return active_data, active_0_name, active_1_name

def precompute_all_graphs(active_list, A, batch_size=256):  
    supplier = Chem.SDMolSupplier(A+".sdf")
    graphs = []; listuseful = []; mol_names = []
    
    for batch_start in range(0, len(active_list), batch_size):
        batch_end = min(batch_start + batch_size, len(active_list))
        batch_indices = active_list[batch_start:batch_end]
        batch_graphs = []; batch_listuseful = []; batch_mol_names = []
        
        for idx in batch_indices:
            try:
                mol = supplier[idx]
                if mol is None: continue

                graph = SDF_dispose.molecule_to_pyg_graph(mol)
                
                # 计算并绑定 ECFP 指纹
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fp_array = np.zeros((1,), dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, fp_array)
                graph.ecfp = torch.tensor(fp_array, dtype=torch.float32).view(1, -1)

                batch_graphs.append(graph)
                batch_listuseful.append(idx)
            except Exception as e:
                continue
        
        graphs.extend(batch_graphs)
        listuseful.extend(batch_listuseful)

    return graphs, listuseful, mol_names

# =====================================================================
# 2. 多模态双塔网络架构 (完全对齐 trainpro)
# =====================================================================
class GNN(nn.Module):
    def __init__(self, INPUT):
        super(GNN, self).__init__()
        self.SIZE_LAYERS = INPUT["SIZE_LAYERS"]
        self.initial_conv = GATConv(self.SIZE_LAYERS[0], self.SIZE_LAYERS[1])
        self.conv1 = GATConv(self.SIZE_LAYERS[1], self.SIZE_LAYERS[2])
        self.conv2 = GATConv(self.SIZE_LAYERS[2], self.SIZE_LAYERS[2])
        self.linear = Linear(self.SIZE_LAYERS[2], self.SIZE_LAYERS[3])

    def forward(self, x, edge_index, drop_rate=0.0):  
        if drop_rate > 0.0 and self.training:
            edge_index, _ = dropout_edge(edge_index, p=drop_rate, force_undirected=True)
            x = x.clone()
        out = F.relu(self.initial_conv(x, edge_index=edge_index))
        out = F.relu(self.conv1(out, edge_index=edge_index))
        out = F.relu(self.conv2(out, edge_index=edge_index))
        return self.linear(out)

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

# =====================================================================
# 3. 评估指标计算 (AUC)
# =====================================================================
def evaluate_test_accuracy_fixed(test_graphs_cache, test_valid_cache, encoder, classifier, test_active_values, device, batch_size=256):
    encoder.eval(); classifier.eval()
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
            gc.collect(); torch.cuda.empty_cache()

    accuracy = 100 * correct / total if total > 0 else 0.0
    try: auc_score = roc_auc_score(all_trues, all_probs)
    except: auc_score = 0.5
    return accuracy, auc_score

def unwrap_state_dict(module):
    try:
        orig = getattr(module, "_orig_mod", None)
        if orig is not None: return orig.state_dict()
    except: pass
    return module.state_dict()

# =====================================================================
# 4. 纯净版端到端训练逻辑 (完全剥离第一阶段对比学习)
# =====================================================================
def ABLATION_TRAIN_NOSTAGE1(A, bili, savepth):
    scaler = torch.cuda.amp.GradScaler()
    RDLogger.DisableLog('rdApp.*')
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # 提取训练集所有分子的真实 ID 和标签
    active_values, active_0_name, active_1_name = extract_active_property(A+".sdf")
    
    all_train_molecules = []
    mol_idx_to_label = {} # 防错位神器：ID到标签的精确映射
    
    for name in active_0_name:
        idx = extract_digits(name)
        all_train_molecules.append(idx)
        mol_idx_to_label[idx] = 0
        
    for name in active_1_name:
        idx = extract_digits(name)
        all_train_molecules.append(idx)
        mol_idx_to_label[idx] = 1
        
    test_file_path = A.replace('train/'+str(bili[0])+str(bili[1])+str(bili[2]), 'validation/')
    test_active_values, _, _ = extract_active_property(test_file_path+ ".sdf")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 训练节奏设定 (与主程序对齐)
    total_epochs = 500
    perturb_start = 100; perturb_ramp_end = 150; target_drop_rate = 0.25
    early_stop_patience = 50; best_test_auc = 0.0; patience_counter = 0

    # 预读一个图获取节点特征维度
    graph,_,_ = precompute_all_graphs([all_train_molecules[0]], A)
    num_node_features = graph[0].x.size(1)
    
    # 网络与优化器
    encoder = GNN({"LR": 1e-3, "NAME": "GNN", "CHECKPOINT_DIR": "", "SIZE_LAYERS": [num_node_features, 64, 64, 32]}).to(device)
    classifier = ClassifierNetwork(gnn_dim=32, ecfp_dim=2048, num_classes=2).to(device)
    
    contrastive_optimizer = optim.AdamW(encoder.parameters(), lr=1e-5, weight_decay=1e-5)
    classification_optimizer = optim.AdamW(classifier.parameters(), lr=1e-5, weight_decay=1e-5)
    
    # 自适应权重交叉熵
    w0 = min(len(active_values) / (2 * len(active_0_name)), 10.0) if len(active_0_name) > 0 else 1.0
    w1 = min(len(active_values) / (2 * len(active_1_name)), 10.0) if len(active_1_name) > 0 else 1.0
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([w0, w1], dtype=torch.float, device=device))
    
    contrastive_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(contrastive_optimizer, T_max=total_epochs, eta_min=1e-5)
    classification_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(classification_optimizer, T_max=total_epochs, eta_min=1e-6)

    # 预加载测试集
    print("\n预加载测试集...")
    all_test_indices = list(range(len(test_active_values)))
    test_graphs_cache, test_valid_cache, _ = precompute_all_graphs(all_test_indices, test_file_path)

    print(f"\n🚀 [No Stage 1 消融实验] 纯端到端微调启动 (总分子数: {len(all_train_molecules)})...")
    
    for epoch in range(total_epochs):
        # 动态图扰动
        if epoch < perturb_start: current_drop_rate = 0.0
        elif perturb_start <= epoch < perturb_ramp_end: current_drop_rate = ((epoch - perturb_start) / (perturb_ramp_end - perturb_start)) * target_drop_rate
        else: current_drop_rate = target_drop_rate
        
        encoder.train()
        classifier.train()
        
        # 每轮完全打乱训练集
        random.shuffle(all_train_molecules)
        
        classification_loss_total = 0.0
        cls_batch_size = 128
        
        # ================= 唯一阶段：端到端联合微调 =================
        for i in tqdm(range(0, len(all_train_molecules), cls_batch_size), desc=f"Epoch {epoch+1}/{total_epochs}"):
            batch_mol_indices = all_train_molecules[i : i+cls_batch_size]
            
            # 即时加载图结构和特征
            batch_graphs, valid_indices, _ = precompute_all_graphs(batch_mol_indices, A)
            if len(batch_graphs) < 2: continue
            
            # 防错位：只提取成功加载的图对应的标签
            valid_labels = [mol_idx_to_label[idx] for idx in valid_indices]
            batch_labels = torch.tensor(valid_labels, dtype=torch.long, device=device)
            batch_data = Batch.from_data_list(batch_graphs).to(device)
            
            classification_optimizer.zero_grad(set_to_none=True)
            contrastive_optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                # 1. 图网络编码
                h = encoder(batch_data.x, batch_data.edge_index, drop_rate=current_drop_rate)
                gnn_embeddings = global_mean_pool(h, batch_data.batch)
                
                # 2. 提取指纹并融合分类
                ecfp_embeddings = batch_data.ecfp.view(-1, 2048)
                classifier_out = classifier(gnn_embeddings, ecfp_embeddings)
                
                loss = criterion(classifier_out, batch_labels)
                
            scaler.scale(loss).backward()
            
            # 同时更新两个网络！
            scaler.step(classification_optimizer)
            scaler.step(contrastive_optimizer)
            scaler.update()
            
            classification_loss_total += loss.item()

        avg_loss = classification_loss_total / max((len(all_train_molecules) // cls_batch_size), 1)
        
        # 评估 AUC
        _, test_auc = evaluate_test_accuracy_fixed(test_graphs_cache, test_valid_cache, encoder, classifier, test_active_values, device)
        
        contrastive_scheduler.step(); classification_scheduler.step()
        
        print(f"  --> Loss: {avg_loss:.4f} | Test AUC: {test_auc:.4f}")
        
        # 早停与模型保存
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            patience_counter = 0
            torch.save({
                'encoder_state_dict': unwrap_state_dict(encoder),
                'classifier_state_dict': unwrap_state_dict(classifier),
                'test_auc': test_auc
            }, savepth)
            print(f"  [★] 新最佳模型，AUC: {best_test_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"连续 {early_stop_patience} 轮未提升，触发早停！")
                break
                
    print(f"训练完成！最佳测试 AUC: {best_test_auc:.4f}")
    return epoch, [], len(active_values), best_test_auc

# =====================================================================
# 5. 主循环启动
# =====================================================================
if __name__ == '__main__':
    bili = [8, 1, 1]
    multiprocessing.set_start_method('spawn', force=True)
    
    filesss = ['train/BBBP', 'train/bace', "train/clintox", 'train/HIV']
    files = [fil.replace('train/','ours/train/'+str(bili[0])+str(bili[1])+str(bili[2])) for fil in filesss]
    
    print("开始消融实验 (w/o Stage 1) 所有模型...")
    txt = [f'训练集,验证集,测试集比例为:{bili[0]}:{bili[1]}:{bili[2]}\n']
    
    for file in files:
        print(f"\n==========================================")
        print(f" 正在运行 Direct-FT (w/o Stage 1): {file}")
        print(f"==========================================")
        
        savepth = file.replace('train', 'pth') + 'xr_nostage1_best_model.pth'
        epoch, _, _, best_auc = ABLATION_TRAIN_NOSTAGE1(file, bili, savepth)
        
        wenben = f"文件名：{file} | 变体: Direct-FT (w/o Stage 1) | 最佳测试集 AUC: {best_auc:.4f}"
        txt.append(wenben)
        
        out_filename = file.replace('train', 'out') + "_ablation_nostage1.txt"
        with open(out_filename, 'w', encoding='utf-8') as f:
            for item in txt: f.write(f"{item}\n")