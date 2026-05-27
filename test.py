import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, GATConv
from torch.nn import Linear
import os
import number
import SDF_dispose  # 修正：使用正确的模块名
import re
from rdkit import Chem
from rdkit import RDLogger

# 定义GNN模型（与训练代码完全一致）
class GNN(nn.Module):
    def __init__(self, INPUT):
        super(GNN, self).__init__()
        self.INPUT = INPUT
        self.SIZE_LAYERS = self.INPUT["SIZE_LAYERS"]
        self.initial_conv = GATConv(self.SIZE_LAYERS[0], self.SIZE_LAYERS[1])
        self.conv1 = GATConv(self.SIZE_LAYERS[1], self.SIZE_LAYERS[2])
        self.conv2 = GATConv(self.SIZE_LAYERS[2], self.SIZE_LAYERS[2])
        self.linear = Linear(self.SIZE_LAYERS[2], self.SIZE_LAYERS[3])

    def forward(self, x, edge_index):
        out = F.relu(self.initial_conv(x, edge_index=edge_index))
        out = F.relu(self.conv1(out, edge_index=edge_index))
        out = F.relu(self.conv2(out, edge_index=edge_index))
        return self.linear(out)

# 定义分类器网络（与训练代码完全一致）
class ClassifierNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        super(ClassifierNetwork, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)

# 预处理图数据函数（与训练代码一致）
def precompute_all_graphs(active_list, A):
    """预加载指定分子列表的图数据和name"""
    supplier = Chem.SDMolSupplier(A+".sdf")
    graphs = []
    listuseful = []
    mol_names = []  # 存储name
    
    for idx in active_list:
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
                    mol_names.append(mol_name)

            graphs.append(graph)
            listuseful.append(idx)

        except Exception as e:
            print(f"处理分子 {idx} 时出错: {e}")
            continue

    return graphs, listuseful, mol_names

# 提取数字函数（与训练代码一致）
def extract_digits(s):
    return int(re.sub(r'^\D+', '', s))

def predict_test(model_path, test_file, device='auto'):
    """
    使用训练好的模型预测测试集
    参数:
        model_path: 训练好的模型路径 (.pth文件)
        test_file: 测试集文件路径 (不含.sdf扩展名)
        device: 使用的设备 ('cuda', 'cpu' 或 'auto')
    """
    # 关闭RDKit的警告信息
    RDLogger.DisableLog('rdApp.*')
    
    # 设备配置
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'使用设备: {device}')
    
    # 加载测试集数据
    print(f"加载测试集: {test_file}.sdf")
    try:
        test_active_values, _, _ = number.extract_active_property(test_file + ".sdf")
    except Exception as e:
        print(f"加载测试集失败: {e}")
        return
    
    # 加载模型
    print(f"加载模型: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # 从测试集获取节点特征维度
        sample_graphs, _, _ = precompute_all_graphs([0], test_file)
        if not sample_graphs:
            raise ValueError("无法从测试集生成图数据")
        num_node_features = sample_graphs[0].x.size(1)
        
        # 重建GNN编码器结构（与训练代码一致）
        gnn_input = {
            "SIZE_LAYERS": [num_node_features, 64, 64, 32]  # 与训练代码一致的结构
        }
        encoder = GNN(gnn_input).to(device)
        
        # 重建分类器结构（与训练代码一致）
        classifier = ClassifierNetwork(
            input_dim=32,  # encoder_output_dim
            hidden_dim=64,  # classifier_hidden_dim
            num_classes=2
        ).to(device)
        
        # 加载模型权重
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        print("模型加载成功!")
        print(f"模型训练轮数: {checkpoint.get('epoch', 'Unknown')}")
        print(f"模型最佳测试准确率: {checkpoint.get('test_accuracy', 'Unknown')}")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 设置模型为评估模式
    encoder.eval()
    classifier.eval()
    
    predictions = []
    correct = 0
    total = 0
    
    print(f"开始预测测试集 ({len(test_active_values)} 个分子)...")
    
    with torch.no_grad():
        for mol_idx in range(len(test_active_values)):
            try:
                # 为当前分子生成图数据
                graphs, _, _ = precompute_all_graphs([mol_idx], test_file)
                if not graphs:
                    predictions.append('0')  # 无法生成图默认预测为0
                    continue
                
                # 进行预测
                batch = Batch.from_data_list(graphs).to(device)
                
                # 通过GNN编码器
                h = encoder(batch.x, batch.edge_index)
                g = global_mean_pool(h, batch.batch)
                
                # 通过分类器
                out = classifier(g)
                
                # 获取预测结果
                _, predicted = torch.max(out.data, 1)
                predicted_label = str(predicted.item())
                predictions.append(predicted_label)
                
                # 计算准确率
                true_label = test_active_values[mol_idx]
                total += 1
                if predicted_label == true_label:
                    correct += 1
                    
                # 每100个分子打印一次进度
                if (mol_idx + 1) % 100 == 0:
                    print(f"已处理 {mol_idx + 1}/{len(test_active_values)} 个分子")
                    
            except Exception as e:
                print(f"处理分子 {mol_idx} 时出错: {e}")
                predictions.append('0')  # 出错时默认预测为0
                total += 1
                continue
    
    accuracy = 100 * correct / total if total > 0 else 0
    
    # 打印测试结果
    print("\n" + "="*50)
    print(f"测试集总分子数: {len(test_active_values)}")
    print(f"成功预测分子数: {total}")
    print(f"预测正确数: {correct}")
    print(f"测试集准确率: {accuracy:.2f}%")
    print("="*50)
    return accuracy, predictions

def test(model_test_mapa):
    # 定义模型和测试集映射
    model_test_map = model_test_mapa
    
    # 遍历所有模型进行预测
    results = {}
    all_predictions = {}
    
    for model_path, test_file in model_test_map.items():
        print("\n" + "="*80)
        print(f"处理: {os.path.basename(model_path)}")
        print("="*80)
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            continue
            
        if not os.path.exists(test_file + ".sdf"):
            print(f"测试集文件不存在: {test_file}.sdf")
            continue
            
        # 进行预测
        result = predict_test(model_path, test_file)
        if result is not None:
            accuracy, predictions = result
            results[os.path.basename(model_path)] = accuracy
            all_predictions[os.path.basename(model_path)] = predictions
    
    # 打印所有模型的结果摘要
    print("\n" + "="*80)
    print("所有模型测试结果摘要")
    print("="*80)
    print(f"{'模型':<30}{'准确率':<10}")
    print("-" * 40)
    for model, acc in results.items():
        print(f"{model:<30}{acc:.2f}%")
    
    # 计算平均准确率
    if results:
        avg_acc = sum(results.values()) / len(results)
        print("-" * 40)
        print(f"{'平均准确率':<30}{avg_acc:.2f}%")
    
    return results, all_predictions

if __name__ == '__main__':
    model_test_map = {
        "pth/nr-ahr.pth": "test/nr-ahr",
        "pth/nr-ar.pth": "test/nr-ar",  
        "pth/nr-ar-lbd.pth": "test/nr-ar-lbd",
        "pth/nr-aromatase.pth": "test/nr-aromatase",
        "pth/nr-er.pth": "test/nr-er",
        "pth/nr-er-lbd.pth": "test/nr-er-lbd",
        "pth/nr-ppar-gamma.pth": "test/nr-ppar-gamma",
        "pth/sr-are.pth": "test/sr-are",
        "pth/sr-atad5.pth": "test/sr-atad5",
        "pth/sr-hse.pth": "test/sr-hse",
        "pth/sr-mmp.pth": "test/sr-mmp",
        "pth/sr-p53.pth": "test/sr-p53",
    }
    results, predictions = test(model_test_map)
    print(results)