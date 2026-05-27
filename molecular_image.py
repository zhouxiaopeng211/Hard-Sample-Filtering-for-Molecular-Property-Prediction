import os
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from PIL import Image
import torch
import torch.nn as nn
import io
import csv
import os
from os import close
from rdkit import Chem
import numpy as np
import heapq
from rdkit import Chem
import re
# from tqdm import tqdm
from rdkit import RDLogger
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, GATConv
from torch.nn import Linear
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs
import SDF_dispose
from number import extract_active_property
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, GATConv
from torch.nn import Linear
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.svm import SVC  # 用于拟合决策边界
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs


RDLogger.DisableLog('rdApp.*')

# ==========================================
# 1. 重新定义模型结构 (确保与训练时完全一致)
# ==========================================
class GNN(nn.Module):
    def __init__(self, size_layers):
        super(GNN, self).__init__()
        self.initial_conv = GATConv(size_layers[0], size_layers[1])
        self.conv1 = GATConv(size_layers[1], size_layers[2])
        self.conv2 = GATConv(size_layers[2], size_layers[2])
        self.linear = Linear(size_layers[2], size_layers[3])

    def forward(self, x, edge_index):
        out = F.relu(self.initial_conv(x, edge_index=edge_index))
        out = F.relu(self.conv1(out, edge_index=edge_index))
        out = F.relu(self.conv2(out, edge_index=edge_index))
        return self.linear(out)

class ClassifierNetwork(nn.Module):
    def __init__(self, gnn_dim=32, ecfp_dim=2048, num_classes=2):
        super(ClassifierNetwork, self).__init__()
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
        fusion_dim = gnn_dim + 64
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
        # 注意这里：我们修改前向传播，让它不仅返回预测结果，还返回拼接后的融合特征！
        ecfp_features = self.ecfp_branch(ecfp_emb)
        combined_features = torch.cat([gnn_emb, ecfp_features], dim=1)
        logits = self.final_classifier(combined_features)
        return logits, combined_features


def build_decision_boundary_grid(X_2d, y_pred, grid_step=0.2, padding=5.0):
    """Fit a 2D SVM surrogate and return continuous scores for a boundary curve."""
    X_2d = np.asarray(X_2d)
    y_pred = np.asarray(y_pred)
    unique_classes = np.unique(y_pred)
    if len(unique_classes) != 2:
        return None

    clf_2d = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf_2d.fit(X_2d, y_pred)

    x_min, x_max = X_2d[:, 0].min() - padding, X_2d[:, 0].max() + padding
    y_min, y_max = X_2d[:, 1].min() - padding, X_2d[:, 1].max() + padding
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    scores = clf_2d.decision_function(grid_points).reshape(xx.shape)
    return xx, yy, scores


def load_labeled_sdf_molecules(sdf_file_path):
    """Load molecules and binary Active labels through Python's Unicode-aware file IO."""
    molecules = []
    labels = []
    with open(sdf_file_path, "rb") as sdf_handle:
        supplier = Chem.ForwardSDMolSupplier(sdf_handle)
        for idx, mol in enumerate(supplier):
            if mol is None:
                print(f"警告: 第 {idx + 1} 个分子无效，跳过")
                continue
            if not mol.HasProp("Active"):
                print(f"第 {idx + 1} 个分子缺少 <Active> 属性，跳过")
                continue
            try:
                label = int(mol.GetProp("Active"))
            except ValueError:
                print(f"第 {idx + 1} 个分子的 <Active> 属性不是整数，跳过")
                continue
            molecules.append(mol)
            labels.append(label)
    return molecules, labels


# ==========================================
# 2. 特征提取与 t-SNE 绘图核心逻辑
# ==========================================
def extract_features_and_plot_tsne(model_path, sdf_file_path, output_image_path, perplexity=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用计算设备: {device}")
    
    # 1. 获取真实标签
    print(f"[*] 正在解析 SDF 文件并提取标签...")
    molecules, labels = load_labeled_sdf_molecules(sdf_file_path)
    
    # 2. 准备图数据和指纹
    graphs = []
    valid_labels = []
    
    for mol, label in zip(molecules, labels):
        try:
            # 提取图特征
            graph = SDF_dispose.molecule_to_pyg_graph(mol)
            if graph is None:
                continue
            # 提取ECFP指纹
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_array = np.zeros((1,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            graph.ecfp = torch.tensor(fp_array, dtype=torch.float32).view(1, -1)
            
            graphs.append(graph)
            # 保存对应的真实标签
            valid_labels.append(label)
        except Exception as e:
            continue
            
    if len(graphs) == 0:
        print("[!] 错误：没有提取到有效的分子！")
        return
        
    print(f"[*] 成功提取 {len(graphs)} 个有效分子的拓扑图与指纹")

    # 3. 加载模型
    print(f"[*] 正在加载权重文件: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    num_node_features = graphs[0].x.size(1)
    encoder = GNN([num_node_features, 64, 64, 32]).to(device)
    classifier = ClassifierNetwork(gnn_dim=32, ecfp_dim=2048, num_classes=2).to(device)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    encoder.eval()
    classifier.eval()
    
    # 4. 前向传播提取特征
    all_features = []
    batch_size = 128
    
    print(f"[*] 正在通过双塔模型提取高维融合特征...")
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            end = min(start + batch_size, len(graphs))
            sub_graphs = graphs[start:end]
            batch = Batch.from_data_list(sub_graphs).to(device)
            
            # GNN提取结构特征
            h = encoder(batch.x, batch.edge_index)
            gnn_embeddings = global_mean_pool(h, batch.batch)
            
            # ECFP特征
            ecfp_embeddings = batch.ecfp.view(-1, 2048).to(device)
            
            # 传入修改后的分类器，获取中间融合特征 (96维度)
            _, combined_features = classifier(gnn_embeddings, ecfp_embeddings)
            
            all_features.append(combined_features.cpu().numpy())
            
    X = np.vstack(all_features)
    y = np.array(valid_labels)
    
    print(f"[*] 成功获取高维特征矩阵: 形状 {X.shape}")
    
    # 5. t-SNE 降维
    print(f"[*] 正在运行 t-SNE 降维 (Perplexity={perplexity})... 这可能需要一点时间...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    
    # 6. 可视化绘图
    print(f"[*] 正在绘制并保存散点图...")
    plt.figure(figsize=(10, 8))
    
    # 将标签映射为字符串，方便图例显示
    hue_labels = ["Negative (Class 0)" if label == 0 else "Positive (Class 1)" for label in y]
    
    # 使用 Seaborn 绘图，设定好看的颜色
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=hue_labels,
        palette={"Negative (Class 0)": "#3498db", "Positive (Class 1)": "#e74c3c"}, # 蓝色和红色
        alpha=0.7,      # 透明度防重叠
        edgecolor=None,
        s=40            # 点的大小
    )
    
    plt.title("t-SNE Visualization of Fused Molecular Representations", fontsize=15, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Activity Status", title_fontsize='13', fontsize='11', loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[*] 搞定！高质量的 t-SNE 图像已保存至: {output_image_path}")


def plot_tsne_with_boundary(model_path, sdf_file_path, output_image_path, perplexity=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用计算设备: {device}")
    
    # 1. 提取数据
    print(f"[*] 正在解析 SDF 文件并提取标签: {sdf_file_path}")
    molecules, labels = load_labeled_sdf_molecules(sdf_file_path)
    graphs = []
    y_true = []
    
    for mol, label in zip(molecules, labels):
        try:
            graph = SDF_dispose.molecule_to_pyg_graph(mol)
            if graph is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_array = np.zeros((1,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            graph.ecfp = torch.tensor(fp_array, dtype=torch.float32).view(1, -1)
            graphs.append(graph)
            y_true.append(label)
        except Exception:
            continue

    if len(graphs) == 0:
        raise RuntimeError("没有提取到有效分子，无法绘制 t-SNE。")
    if len(graphs) <= 1:
        raise RuntimeError("有效分子数量不足，无法绘制 t-SNE。")

    perplexity = min(perplexity, max(1, len(graphs) - 1))
    print(f"[*] 成功提取 {len(graphs)} 个有效分子，t-SNE perplexity={perplexity}")

    # 2. 加载模型并预测
    print(f"[*] 正在加载权重文件: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    num_node_features = graphs[0].x.size(1)
    encoder = GNN([num_node_features, 64, 64, 32]).to(device)
    classifier = ClassifierNetwork().to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    encoder.eval(); classifier.eval()

    all_features = []; all_preds = []
    with torch.no_grad():
        for start in range(0, len(graphs), 128):
            batch = Batch.from_data_list(graphs[start:start+128]).to(device)
            h = encoder(batch.x, batch.edge_index)
            gnn_emb = global_mean_pool(h, batch.batch)
            ecfp_emb = batch.ecfp.view(-1, 2048).to(device)
            logits, combined = classifier(gnn_emb, ecfp_emb)
            all_features.append(combined.cpu().numpy())
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())

    X = np.vstack(all_features)
    y_pred = np.concatenate(all_preds)
    y_true = np.array(y_true)

    # 3. t-SNE 降维
    print(f"[*] 正在运行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    X_2d = tsne.fit_transform(X)

    # 4. 训练代理模型绘制决策边界 (SVM)
    # 我们用 2D 坐标去拟合深度模型的预测分类结果 y_pred
    boundary = build_decision_boundary_grid(X_2d, y_pred, grid_step=0.2, padding=5.0)

    # 5. 绘图
    output_path = Path(output_image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 10))
    
    if boundary is not None:
        xx, yy, scores = boundary
        plt.contourf(xx, yy, scores, levels=20, alpha=0.12, cmap=plt.cm.RdBu_r)
        plt.contour(xx, yy, scores, levels=[0], colors='black', linewidths=1.2, linestyles='--')
    else:
        print("[!] 模型在 t-SNE 点上的预测只有一个类别，跳过决策边界曲线。")

    # 绘制散点
    df = pd.DataFrame({'x': X_2d[:, 0], 'y': X_2d[:, 1], 'Label': y_true})
    sns.scatterplot(data=df, x='x', y='y', hue='Label', 
                    palette={0: "#3498db", 1: "#e74c3c"}, alpha=0.8, s=50)

    plt.title(f"t-SNE with Decision Boundary\nModel: {os.path.basename(model_path)}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] t-SNE 决策边界图像已保存至: {output_path}")
    return str(output_path)
#=====================================================================

def draw_mol_from_sdf(sdf_input, index=0, size=(400, 400), save_path=None):
    """
    将 SDF 文件中的指定分子或直接传入的 Mol 对象绘制为二维结构图。

    参数:
    ----------
    sdf_input : str 或 rdkit.Chem.rdchem.Mol
        如果是字符串，则视为 SDF 文件路径；
        如果是 Mol 对象，则直接绘制该对象。
    index : int, 可选
        当 sdf_input 为文件路径时，指定要绘制第几个分子（从 0 开始），默认为 0。
    size : tuple, 可选
        输出图片的大小 (宽, 高)，默认为 (400, 400)。
    save_path : str, 可选
        图片保存路径（如 'molecule.png'）。如果为 None，则直接显示图片。

    返回:
    -------
    PIL.Image.Image 或 None
        返回 Pillow 图片对象；如果失败则返回 None。
    """
    mol = None

    # --- 1. 处理输入：获取 Mol 对象 ---
    if isinstance(sdf_input, str):
        # 输入是文件路径
        if not os.path.exists(sdf_input):
            print(f"错误: 文件未找到 - {sdf_input}")
            return None
        
        # 使用 SDMolSupplier 读取 SDF
        # removeHs=True (默认) 通常让 2D 图更整洁，只显示非氢原子
        supplier = Chem.SDMolSupplier(sdf_input, removeHs=True)
        
        if index >= len(supplier) or index < 0:
            print(f"错误: 索引 {index} 超出范围 (文件内共有 {len(supplier)} 个分子)。")
            return None
            
        mol = supplier[index]
        if mol is None:
            print(f"错误: 无法解析 SDF 文件中索引为 {index} 的分子。")
            return None
        print(f"成功读取 SDF 文件 '{sdf_input}' 中的第 {index} 个分子。")

    elif isinstance(sdf_input, Chem.rdchem.Mol):
        # 输入直接是 Mol 对象
        mol = sdf_input
        print("直接处理传入的 Mol 对象。")
    else:
        print("错误: input_data 必须是 SDF 文件路径(str) 或 RDKit Mol 对象。")
        return None

    # --- 2. 核心步骤：确保有 2D 坐标 ---
    # 虽然你的 SDF 可能是 2D 的，但 RDKit 的绘图引擎在显式生成
    # 2D 构象后，通常能产出更美观、键角更标准的图片。
    # 如果不关心美观，可以尝试注释掉下面这一行。
    AllChem.Compute2DCoords(mol)


    # --- 3. 绘制分子图 ---
    try:
        # MolToImage 返回一个 PIL (Pillow) 图片对象
        img = Draw.MolToImage(mol, size=size, kekulize=True)
        
        # --- 4. 处理输出 ---
        if save_path:
            img.save(save_path)
            print(f"分子图已保存至: {save_path}")
        else:
            # 如果是在 Jupyter Notebook 环境中，直接返回 img 即可显示
            # 如果是在 Linux 终端运行，这行代码不会有视觉输出，
            # 需要在支持 GUI 的环境下调用 img.show() 才能弹出窗口。
            # img.show() # 在纯终端 Linux 环境下可能会报错，建议使用 save_path
            pass
            
        return img

    except Exception as e:
        print(f"绘图过程中出现异常: {e}")
        return None

# =========================================

def jisuan(C,supplier,sz=20):
    from gesim import gesim
    from train import extract_active_property as train_extract_active_property

    active_values, active_0_name, active_1_name = train_extract_active_property(C)
    print(active_values[sz])
    if active_values[sz] == '0':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        RDLogger.DisableLog('rdApp.*')
        mol0=[]
        mol1=[]
        film = C
        
        i=sz
        for xxx in active_0_name:
            mol0.append(supplier[int(re.sub(r'^\D+', '', xxx))])
        for xxx in active_1_name:
            mol1.append(supplier[int(re.sub(r'^\D+', '', xxx))])
        molmaodian=supplier[i]
        sslist = gesim.graph_entropy_similarity_batch(molmaodian, mol0)
        # print(len(sslist),len(active_0_name),active_values[int(re.sub(r'^\D+', '', active_0_name[i]))],active_0_name[i])
        # 将列表转换为 NumPy 数组
        arr = np.array(sslist)
        # max_value = arr.max()
        # max_index = arr.argmax()
        #===============    ================    ================
        # 1. 找到第二大的数值
        unique_sorted = np.unique(arr)
        if len(unique_sorted) < 2:
            print("列表中不同的数字少于2个。")
        else:
            max_value = unique_sorted[-2]
            
            # 2. 找到原数组中所有等于第二大的数的索引
            max_index = np.where(arr == max_value)[0]
            max_index = max_index[0]
            # print(f"第二大的数是: {max_value}")
            # print(f"它在原数组的位置(索引)是: {max_index}")
        #===============    ================    ================
        # 2. 找到最小值和它的索引（位置）
        min_value = arr.min()
        min_index = arr.argmin()
        print(f"锚点分子: {int(re.sub(r'^\D+', '', active_0_name[sz]))}")
        print(f"最大值: {max_value}, 位置: {int(re.sub(r'^\D+', '', active_0_name[max_index]))}")
        print(f"最小值: {min_value}, 位置: {int(re.sub(r'^\D+', '', active_0_name[min_index]))}")
        a0 = int(i)
        a2 = int(re.sub(r'^\D+', '', active_0_name[max_index]))
        a4 = int(re.sub(r'^\D+', '', active_0_name[min_index]))

        #===============    ================    ================

        sslist = gesim.graph_entropy_similarity_batch(molmaodian, mol1)
        arr = np.array(sslist)
        max_value = arr.max()
        max_index = arr.argmax()
        min_value = arr.min()
        min_index = arr.argmin()
        # print(f"锚点分子: {int(re.sub(r'^\D+', '', active_0_name[sz]))}")
        print(f"最大值: {max_value}, 位置: {int(re.sub(r'^\D+', '', active_1_name[max_index]))}")
        print(f"最小值: {min_value}, 位置: {int(re.sub(r'^\D+', '', active_1_name[min_index]))}")
        a1 = int(re.sub(r'^\D+', '', active_1_name[max_index]))
        a3 = int(re.sub(r'^\D+', '', active_1_name[min_index]))

        
    elif active_values[sz] == '1':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        RDLogger.DisableLog('rdApp.*')
        mol0=[]
        mol1=[]
        film = C
        
        i=sz
        for xxx in active_0_name:
            mol0.append(supplier[int(re.sub(r'^\D+', '', xxx))])
        for xxx in active_1_name:
            mol1.append(supplier[int(re.sub(r'^\D+', '', xxx))])
        molmaodian=supplier[i]
        sslist = gesim.graph_entropy_similarity_batch(molmaodian, mol1)
        # print(len(sslist),len(active_0_name),active_values[int(re.sub(r'^\D+', '', active_0_name[i]))],active_0_name[i])
        # 将列表转换为 NumPy 数组
        arr = np.array(sslist)
        # max_value = arr.max()
        # max_index = arr.argmax()
        #===============    ================    ================
        # 1. 找到第二大的数值
        unique_sorted = np.unique(arr)
        if len(unique_sorted) < 2:
            print("列表中不同的数字少于2个。")
        else:
            max_value = unique_sorted[-2]
            
            # 2. 找到原数组中所有等于第二大的数的索引
            max_index = np.where(arr == max_value)[0]
            max_index = max_index[0]
            # print(f"第二大的数是: {max_value}")
            # print(f"它在原数组的位置(索引)是: {max_index}")
        #===============    ================    ================
        # 2. 找到最小值和它的索引（位置）
        min_value = arr.min()
        min_index = arr.argmin()
        print(f"锚点分子: {int(re.sub(r'^\D+', '', active_1_name[sz]))}")
        print(f"最大值: {max_value}, 位置: {int(re.sub(r'^\D+', '', active_1_name[max_index]))}")
        print(f"最小值: {min_value}, 位置: {int(re.sub(r'^\D+', '', active_1_name[min_index]))}")
        a0 = i
        a2 = int(re.sub(r'^\D+', '', active_1_name[max_index]))
        a4 = int(re.sub(r'^\D+', '', active_1_name[min_index]))

        #===============    ================    ================

        sslist = gesim.graph_entropy_similarity_batch(molmaodian, mol0)
        arr = np.array(sslist)
        max_value = arr.max()
        max_index = arr.argmax()
        min_value = arr.min()
        min_index = arr.argmin()
        # print(f"锚点分子: {int(re.sub(r'^\D+', '', active_0_name[sz]))}")
        print(f"最大值: {max_value}, 位置: {int(re.sub(r'^\D+', '', active_0_name[max_index]))}")
        print(f"最小值: {min_value}, 位置: {int(re.sub(r'^\D+', '', active_0_name[min_index]))}")
        a1 = int(re.sub(r'^\D+', '', active_0_name[max_index]))
        a3 = int(re.sub(r'^\D+', '', active_0_name[min_index]))

    return a0,a1,a2,a3,a4
def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    default_model = script_dir / "pth-end" / "对比训练" / "xr" / "811bacexr_nomatrix.pth"
    default_sdf = script_dir / "train" / "811bace.sdf"
    default_output_dir = project_root / "图片集" / "t-SNEw" / "oHSM"

    parser = argparse.ArgumentParser(description="Draw a t-SNE plot with a 2D decision boundary.")
    parser.add_argument("--model", default=str(default_model), help="Path to the .pth checkpoint.")
    parser.add_argument("--sdf", default=str(default_sdf), help="Path to the SDF file used for t-SNE.")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="Directory for the generated image.")
    parser.add_argument("--output-name", default="811bacexr_nomatrix_tsne_boundary.png", help="Generated image file name.")
    parser.add_argument("--perplexity", type=int, default=15, help="t-SNE perplexity.")
    return parser.parse_args()


# ==========================================
if __name__ == "__main__":
# ==========================================
    args = parse_args()
    model_path = Path(args.model)
    sdf_file_path = Path(args.sdf)
    output_image_path = Path(args.output_dir) / args.output_name

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not sdf_file_path.exists():
        raise FileNotFoundError(f"SDF 文件不存在: {sdf_file_path}")

    plot_tsne_with_boundary(
        str(model_path),
        str(sdf_file_path),
        str(output_image_path),
        perplexity=args.perplexity,
    )
