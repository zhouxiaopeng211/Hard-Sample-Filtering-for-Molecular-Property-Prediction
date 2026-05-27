import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, GATConv
from torch.nn import Linear
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import SDF_dispose
import os
import pandas as pd
# from tqdm import tqdm
from number import extract_active_property
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import seaborn as sns
import csv
from rdkit.Chem import AllChem
from rdkit import DataStructs

# 关闭RDKit的警告信息
RDLogger.DisableLog('rdApp.*')

class GNN(nn.Module):
    """GNN编码器模型"""
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

#================================================================================================
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
#===============无ECFP部分=======================================================================
# class ClassifierNetwork(nn.Module):
#     def __init__(self, gnn_dim=32, ecfp_dim=2048, num_classes=2):
#         super(ClassifierNetwork, self).__init__()
        
#         # 塔 A：ECFP 指纹专属降维通道 (已彻底屏蔽)
#         # self.ecfp_branch = ...
        
#         # 融合后的总维度：仅使用图结构特征 (32)
#         fusion_dim = gnn_dim 
        
#         # 融合决策塔：仅接收 32 维的图特征进行最终分类
#         self.final_classifier = nn.Sequential(
#             nn.Linear(fusion_dim, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
            
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Dropout(0.1),
            
#             nn.Linear(32, num_classes)
#         )
        
#     def forward(self, gnn_emb, ecfp_emb):
#         # 1. 彻底忽略 ecfp_emb
#         # 2. 取消特征拼接，直接将 GNN 特征送入分类器
#         combined_features = gnn_emb
        
#         # 3. 输出分类预测
#         return self.final_classifier(combined_features)
#================================================================================================

class MoleculePropertyPredictor:
    """分子性质预测器"""
    
    def __init__(self, model_path, device=None):
        """
        初始化预测器
        
        Args:
            model_path (str): 模型权重文件路径 (.pth文件)
            device: 计算设备，如果为None则自动选择
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # 加载模型
        self.encoder, self.classifier = self._load_model()
        print(f"模型加载完成，使用设备: {self.device}")
        
    def _load_model(self):
        """加载训练好的模型"""
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 加载模型权重
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 从checkpoint中获取模型架构信息
        # 需要根据训练时的参数重新构建模型
        # 这里使用默认参数，你可能需要根据实际情况调整
        encoder_output_dim = 32
        hidden_dim1 = 64
        hidden_dim2 = 64
        classifier_hidden_dim = 64
        
        # 创建编码器
        # 注意：这里需要知道特征维度，可能需要从第一个分子获取
        # 先创建一个临时的，后面会根据实际数据调整
        encoder = None
        # ==== 修改为新的双塔分类器初始化方式 ====
        classifier = ClassifierNetwork(
            gnn_dim=encoder_output_dim,
            ecfp_dim=2048,
            num_classes=2
        ).to(self.device)
        
        # 加载分类器权重
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        
        # 存储编码器状态字典，稍后加载
        self.encoder_state_dict = checkpoint['encoder_state_dict']
        
        return encoder, classifier
    
    def _initialize_encoder(self, num_node_features):
        """根据分子特征维度初始化编码器"""
        size_layers = [num_node_features, 64, 64, 32]
        encoder = GNN(size_layers).to(self.device)
        encoder.load_state_dict(self.encoder_state_dict)
        return encoder
    
    def _molecule_to_graph(self, mol):
        """将分子转换为图数据，并挂载ECFP指纹"""
        try:
            if mol is None:
                return None
            graph = SDF_dispose.molecule_to_pyg_graph(mol)
            
            # ==== 新增：计算并挂载 ECFP 指纹 ====
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_array = np.zeros((1,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            
            # 挂载属性
            graph.ecfp = torch.tensor(fp_array, dtype=torch.float32).view(1, -1)
            # ====================================
            
            return graph
        except Exception as e:
            print(f"分子转换为图时出错: {e}")
            return None
    
    def predict_single_molecule(self, mol):
        """预测单个分子的性质概率"""
        # 转换为图数据
        graph = self._molecule_to_graph(mol)
        if graph is None:
            return None
        
        # 如果编码器还未初始化，则根据特征维度初始化
        if self.encoder is None:
            num_node_features = graph.x.size(1)
            self.encoder = self._initialize_encoder(num_node_features)
            self.encoder.eval()
        
        self.classifier.eval()
        
        with torch.no_grad():
            # 转换为batch
            batch = Batch.from_data_list([graph]).to(self.device)
            
            # ==== 修改前向传播：双特征提取 ====
            # 1. 提 GNN 特征
            h = self.encoder(batch.x, batch.edge_index)
            gnn_embeddings = global_mean_pool(h, batch.batch)
            
            # 2. 提 ECFP 特征
            ecfp_embeddings = batch.ecfp.view(-1, 2048).to(self.device)
            
            # 3. 通过双塔分类器
            output = self.classifier(gnn_embeddings, ecfp_embeddings)
            # ==================================
            
            # 计算概率
            probabilities = F.softmax(output, dim=1)
            class_0_prob = probabilities[0][0].item()
            class_1_prob = probabilities[0][1].item()
            
            # 预测类别
            predicted_class = torch.argmax(output, dim=1).item()
            
            return {
                'class_0_prob': class_0_prob,
                'class_1_prob': class_1_prob,
                'predicted_class': predicted_class,
                'confidence': max(class_0_prob, class_1_prob)
            }
    
    def predict_sdf_file(self, sdf_file_path, output_file=None, batch_size=32):
        """
        预测SDF文件中所有分子的性质概率
        
        Args:
            sdf_file_path (str): SDF文件路径
            output_file (str): 输出文件路径，如果为None则自动生成
            batch_size (int): 批处理大小
            
        Returns:
            pd.DataFrame: 包含预测结果的DataFrame
        """
        if not os.path.exists(sdf_file_path):
            raise FileNotFoundError(f"SDF文件不存在: {sdf_file_path}")
        
        # 读取SDF文件
        supplier = Chem.SDMolSupplier(sdf_file_path)
        
        results = []
        valid_molecules = 0
        total_molecules = len(supplier)
        
        print(f"开始处理 {sdf_file_path}，共 {total_molecules} 个分子...")
        
        # 使用tqdm显示进度
        # for idx, mol in enumerate(tqdm(supplier, desc="预测进度")):
        for idx, mol in enumerate(supplier):
            try:
                if mol is None:
                    results.append({
                        'molecule_index': idx,
                        'molecule_name': 'Invalid',
                        'class_0_prob': np.nan,
                        'class_1_prob': np.nan,
                        'predicted_class': -1,
                        'confidence': np.nan,
                        'status': 'Invalid molecule'
                    })
                    continue
                
                # 获取分子名称
                mol_name = mol.GetProp("name") if mol.HasProp("name") else f"mol_{idx}"
                
                # 预测
                prediction = self.predict_single_molecule(mol)
                
                if prediction is not None:
                    results.append({
                        'molecule_index': idx,
                        'molecule_name': mol_name,
                        'class_0_prob': prediction['class_0_prob'],
                        'class_1_prob': prediction['class_1_prob'],
                        'predicted_class': prediction['predicted_class'],
                        'confidence': prediction['confidence'],
                        'status': 'Success'
                    })
                    valid_molecules += 1
                else:
                    results.append({
                        'molecule_index': idx,
                        'molecule_name': mol_name,
                        'class_0_prob': np.nan,
                        'class_1_prob': np.nan,
                        'predicted_class': -1,
                        'confidence': np.nan,
                        'status': 'Prediction failed'
                    })
                    
            except Exception as e:
                results.append({
                    'molecule_index': idx,
                    'molecule_name': f"mol_{idx}",
                    'class_0_prob': np.nan,
                    'class_1_prob': np.nan,
                    'predicted_class': -1,
                    'confidence': np.nan,
                    'status': f'Error: {str(e)}'
                })
                continue
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results)
        
        # 生成输出文件名
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(sdf_file_path))[0]
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            output_file = f"{base_name}_predictions_{model_name}.csv"
        
        # 保存结果
        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_results.to_csv(output_file, index=False)
        
        
        
        # 打印预测类别分布
        successful_predictions = df_results[df_results['status'] == 'Success']
        if len(successful_predictions) > 0:
            class_counts = successful_predictions['predicted_class'].value_counts()
            # print(f"\n预测类别分布:")
            # print(f"类别 0: {class_counts.get(0, 0)} 个分子")
            # print(f"类别 1: {class_counts.get(1, 0)} 个分子")
            
            avg_confidence = successful_predictions['confidence'].mean()
            print(f"平均置信度: {avg_confidence:.4f}")
            if avg_confidence <0.6:
                print(f"平均置信度低于0.6，模型可能有问题！")
            else:
                print(f"平均置信度高于0.6，模型表现良好！")
        return df_results,output_file

def calculate_roc_auc(y_true, y_prob, output_dir="roc_results"):
    """
    计算ROC曲线和AUC值
    
    Args:
        y_true: 真实标签 (list or array)
        y_prob: 预测的正类概率 (list or array)
        output_dir: 输出目录
        
    Returns:
        dict: 包含AUC值和其他评估指标的字典
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 转换为numpy数组
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # 计算AUC值
    auc_value = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 计算最佳阈值（Youden Index）
    youden_index = tpr - fpr
    best_threshold_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_idx]
    best_tpr = tpr[best_threshold_idx]
    best_fpr = fpr[best_threshold_idx]
    
    # 在ROC曲线上标记最佳阈值点
    plt.plot(best_fpr, best_tpr, 'ro', markersize=8, 
             label=f'Best threshold = {best_threshold:.3f}')
    plt.legend(loc="lower right")
    
    # 绘制阈值与TPR/FPR的关系
    plt.subplot(2, 2, 2)
    plt.plot(thresholds, tpr, label='True Positive Rate', color='green')
    plt.plot(thresholds, fpr, label='False Positive Rate', color='red')
    plt.axvline(x=best_threshold, color='orange', linestyle='--', 
                label=f'Best threshold = {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('TPR and FPR vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 使用最佳阈值进行预测
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制混淆矩阵
    plt.subplot(2, 2, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
    plt.title(f'Confusion Matrix (Threshold = {best_threshold:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 绘制概率分布直方图
    plt.subplot(2, 2, 4)
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='Class 0', color='blue')
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='Class 1', color='red')
    plt.axvline(x=best_threshold, color='orange', linestyle='--', 
                label=f'Best threshold = {best_threshold:.3f}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算其他评估指标
    tn, fp, fn, tp = cm.ravel()
    
    # 计算各种评估指标
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 准备结果字典
    results = {
        'auc': auc_value,
        'best_threshold': best_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }
    
    # 保存详细的分类报告
    y_pred_best = (y_prob >= best_threshold).astype(int)
    class_report = classification_report(y_true, y_pred_best, 
                                       target_names=['Class 0', 'Class 1'], 
                                       output_dict=True)
    
    # 打印详细结果
    print(f"\n{'='*50}")
    # print("ROC-AUC 分析结果")
    # print(f"{'='*50}")
    # print(f"AUC值: {auc_value:.4f}")
    # print(f"最佳阈值: {best_threshold:.4f}")
    # print(f"准确率: {accuracy:.4f}")
    # print(f"精确率: {precision:.4f}")
    # print(f"召回率: {recall:.4f}")
    # print(f"特异性: {specificity:.4f}")
    # print(f"F1分数: {f1_score:.4f}")
    # print(f"\n混淆矩阵:")
    # print(f"真负例 (TN): {tn}")
    # print(f"假正例 (FP): {fp}")
    # print(f"假负例 (FN): {fn}")
    # print(f"真正例 (TP): {tp}")
    
    # 保存结果到文件
    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(output_dir, 'roc_metrics.csv'), index=False)
    
    # 保存分类报告
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    print(f"\n结果已保存到: {output_dir}")
    
    return results

def main(model_path, sdf_file_path, output_file, roc_output_dir):
    """主函数示例"""
    # 配置参数
    # model_path = "pth/244sr-p53n-10similer_nuber-10_best_model.pth"  # 替换为你的模型路径
    # sdf_file_path = "test/244sr-p53.sdf"  # 替换为你的SDF文件路径
    # output_file = "out/predictions_output.csv"  # 输出文件路径
    
    try:
        # 创建预测器
        predictor = MoleculePropertyPredictor(model_path)
        
        # 执行预测
        results_df ,csvs= predictor.predict_sdf_file(
            sdf_file_path=sdf_file_path,
            output_file=output_file,
            batch_size=32
        )
        # 计算准确率（保留原代码）
        right = 0
        row = 0
        real_values, active_0_name, active_1_name = extract_active_property(sdf_file_path)
        
        for i in range(len(real_values)):
            if str(results_df['predicted_class'][i]) == real_values[i]:
                right += 1
            elif str(results_df['predicted_class'][i]) != real_values[i]:
                row += 1
        
        accuracy = right / len(real_values)
        print(f"准确率: {accuracy:.4f}")
        
        # ===== 新增：计算ROC曲线和AUC值 =====
        
        # 获取成功预测的数据
        successful_mask = results_df['status'] == 'Success'
        successful_results = results_df[successful_mask].copy()
        
        if len(successful_results) > 0:
            # 准备真实标签和预测概率
            y_true = []
            y_prob_class1 = []
            
            # 确保索引对应正确
            for idx in successful_results.index:
                if idx < len(real_values):
                    y_true.append(int(real_values[idx]))
                    y_prob_class1.append(successful_results.loc[idx, 'class_1_prob'])
            
            if len(y_true) > 0 and len(set(y_true)) > 1:  # 确保有两个类别
                print(f"\n开始计算ROC曲线和AUC值...")
                print(f"样本数量: {len(y_true)}")
                print(f"正样本数量: {sum(y_true)}")
                print(f"负样本数量: {len(y_true) - sum(y_true)}")
                
                # 计算ROC和AUC
                roc_results = calculate_roc_auc(
                y_true=y_true,
                y_prob=y_prob_class1,
                output_dir=roc_output_dir
                )
                print(f"\n权重文件{model_path}:")
                print(f"\n主要性能指标:")
                print(f"AUC: {roc_results['auc']:.4f}")
                print(f"最佳阈值: {roc_results['best_threshold']:.4f}")
                print(f"在最佳阈值下的准确率: {roc_results['accuracy']:.4f}")
                return {
                    'model_name': os.path.splitext(os.path.basename(model_path))[0],
                    'model_path': model_path,
                    'prediction_csv': output_file,
                    'roc_output_dir': roc_output_dir,
                    'num_samples': len(y_true),
                    'num_positive': int(sum(y_true)),
                    'num_negative': int(len(y_true) - sum(y_true)),
                    'auc': roc_results['auc'],
                    'best_threshold': roc_results['best_threshold'],
                    'accuracy': roc_results['accuracy'],
                    'precision': roc_results['precision'],
                    'recall': roc_results['recall'],
                    'specificity': roc_results['specificity'],
                    'f1_score': roc_results['f1_score'],
                    'raw_accuracy': accuracy
                }
                
            else:
                print("警告: 数据中只有一个类别或数据为空，无法计算ROC曲线")
                return {
                    'model_name': os.path.splitext(os.path.basename(model_path))[0],
                    'model_path': model_path,
                    'prediction_csv': output_file,
                    'roc_output_dir': roc_output_dir,
                    'num_samples': len(y_true),
                    'num_positive': int(sum(y_true)) if len(y_true) > 0 else 0,
                    'num_negative': int(len(y_true) - sum(y_true)) if len(y_true) > 0 else 0,
                    'auc': np.nan,
                    'best_threshold': np.nan,
                    'accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'specificity': np.nan,
                    'f1_score': np.nan,
                    'raw_accuracy': accuracy
                }
        else:
            print("警告: 没有成功的预测结果，无法计算ROC曲线")
            return {
                'model_name': os.path.splitext(os.path.basename(model_path))[0],
                'model_path': model_path,
                'prediction_csv': output_file,
                'roc_output_dir': roc_output_dir,
                'num_samples': 0,
                'num_positive': 0,
                'num_negative': 0,
                'auc': np.nan,
                'best_threshold': np.nan,
                'accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'specificity': np.nan,
                'f1_score': np.nan,
                'raw_accuracy': accuracy
            }

    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model_name': os.path.splitext(os.path.basename(model_path))[0],
            'model_path': model_path,
            'prediction_csv': output_file,
            'roc_output_dir': roc_output_dir,
            'num_samples': 0,
            'num_positive': 0,
            'num_negative': 0,
            'auc': np.nan,
            'best_threshold': np.nan,
            'accuracy': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'specificity': np.nan,
            'f1_score': np.nan,
            'raw_accuracy': np.nan
        }

if __name__ == "__main__":
    for bi in range(1):  
        # bi = 3
        if bi == 0:
            bili=[8,1,1]
        elif bi == 2:
            bili=[4,3,3]
        elif bi == 1:
            bili=[6,2,2]
        elif bi == 3:
            bili=[2,4,4]
        
        print(str(bili[0])+':'+str(bili[1])+':'+str(bili[2]))
        
        filess=[
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
            
            'ABBBP',
            'bace',
            "clintox",
            'HIV',
            ]
        
        for i in range(len(filess)):
            pth_root = "/root/autodl-tmp/13_zhouxiaopeng/ours/pth_ts"
            sdf_file_path = "/root/autodl-tmp/13_zhouxiaopeng/ours/test/"+filess[i]+".sdf"
            save_root = "/root/autodl-tmp/13_zhouxiaopeng/ours/auc_summary"

            folders = [filess[i]+"02",filess[i]+"03",filess[i]+"04"]

            os.makedirs(save_root, exist_ok=True)

            for folder in folders:
                folder_path = os.path.join(pth_root, folder)
                if not os.path.exists(folder_path):
                    print(f"不存在目录: {folder_path}")
                    continue

                rows = []

                model_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".pth")])

                for model_file in model_files:
                    model_path = os.path.join(folder_path, model_file)
                    model_stem = os.path.splitext(model_file)[0]

                    output_file = os.path.join(save_root, folder, "predictions", model_stem + ".csv")
                    roc_output_dir = os.path.join(save_root, folder, "roc", model_stem)

                    result = main(
                        model_path=model_path,
                        sdf_file_path=sdf_file_path,
                        output_file=output_file,
                        roc_output_dir=roc_output_dir
                    )

                    if result is not None:
                        rows.append(result)

                    print("=" * 50)

                if len(rows) == 0:
                    print(f"{folder} 没有可保存的结果")
                    continue

                df = pd.DataFrame(rows)

                # 按 AUC 从高到低排序
                df = df.sort_values(by="auc", ascending=False, na_position="last")

                excel_path = os.path.join(save_root, f"{folder}_auc.xlsx")

                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                    # Sheet 1: 完整汇总
                    df.to_excel(writer, index=False, sheet_name="summary")

                    # Sheet 2: 主要指标
                    df[[
                        "model_name", "auc", "accuracy", "raw_accuracy",
                        "best_threshold", "precision", "recall",
                        "specificity", "f1_score",
                        "num_samples", "num_positive", "num_negative"
                    ]].to_excel(writer, index=False, sheet_name="metrics")

                    # Sheet 3: 文件路径
                    df[[
                        "model_name", "model_path", "prediction_csv", "roc_output_dir"
                    ]].to_excel(writer, index=False, sheet_name="files")

                print(f"已保存 Excel: {excel_path}")