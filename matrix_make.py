import csv
import os
from os import close
from rdkit import Chem
import numpy as np
import pandas as pd
import train
import heapq
from rdkit import Chem
from gesim import gesim
import re
# from tqdm import tqdm
from rdkit import RDLogger
import torch

def different_active_matrix(active_values, active_0_name, active_1_name, A, C, k):
    # 启用GPU设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    RDLogger.DisableLog('rdApp.*')
    
    matrix = A
    number0 = []
    name0 = []
    number1 = []
    name1 = []
    similarity = []
    film = C
    supplier = Chem.SDMolSupplier(film+".sdf")
    
    # 提取分子编号
    for i in active_0_name:
        number0.append(int(re.sub(r'^\D+', '', i)))
    for i in active_1_name:
        number1.append(int(re.sub(r'^\D+', '', i)))
    
    # 定义CSV文件路径
    sim_0_csv_path = f"{film.replace('train/', 'matrix/')}_similarity_active_0.csv"  # active_1 vs active_0 相似度矩阵
    sim_1_csv_path = f"{film.replace('train/', 'matrix/')}_similarity_active_1.csv"  # active_1 vs active_1 相似度矩阵
    
    # 预加载所有需要的分子到内存，减少重复读取
    print("预加载分子数据...")
    mol0_cache = {}
    mol1_cache = {}
    
    for j in number0:
        mol0_cache[j] = supplier[j]
    for j in number1:
        mol1_cache[j] = supplier[j]
    
    # 检查CSV文件是否存在
    sim_0_matrix = None
    sim_1_matrix = None
    
    if os.path.exists(sim_0_csv_path) and os.path.exists(sim_1_csv_path):
        print("找到相似度CSV文件，直接加载...")
        try:
            # 加载相似度矩阵
            sim_0_matrix = pd.read_csv(sim_0_csv_path, index_col=0).values
            sim_1_matrix = pd.read_csv(sim_1_csv_path, index_col=0).values
            
            # 验证矩阵维度是否正确
            if (sim_0_matrix.shape[0] == len(number1) and sim_0_matrix.shape[1] == len(number0) and
                sim_1_matrix.shape[0] == len(number1) and sim_1_matrix.shape[1] == len(number1)):
                print("CSV文件维度匹配，使用缓存数据")
            else:
                print("CSV文件维度不匹配，重新计算相似度")
                sim_0_matrix = None
                sim_1_matrix = None
        except Exception as e:
            print(f"加载CSV文件失败: {e}，重新计算相似度")
            sim_0_matrix = None
            sim_1_matrix = None
    
    # 如果CSV不存在或加载失败，计算相似度矩阵
    if sim_0_matrix is None or sim_1_matrix is None:
        print("开始计算相似度矩阵...")
        
        # 初始化相似度矩阵
        sim_0_matrix = np.zeros((len(number1), len(number0)))
        sim_1_matrix = np.zeros((len(number1), len(number1)))
        
        # 计算active_1 vs active_0相似度矩阵
        print("计算active_1 vs active_0相似度矩阵...")
        for i in range(len(number1)):
            target_mol = mol1_cache[number1[i]]
            mol0_list = [mol0_cache[j] for j in number0]
            
            try:
                sslist = gesim.graph_entropy_similarity_batch(target_mol, mol0_list, device=device)
            except TypeError:
                sslist = gesim.graph_entropy_similarity_batch(target_mol, mol0_list)
            
            sim_0_matrix[i] = sslist
        
        # 计算active_1 vs active_1相似度矩阵  
        print("计算active_1 vs active_1相似度矩阵...")
        for i in range(len(number1)):
            target_mol = mol1_cache[number1[i]]
            mol1_list = [mol1_cache[j] for j in number1]
            
            try:
                sslist = gesim.graph_entropy_similarity_batch(target_mol, mol1_list, device=device)
            except TypeError:
                sslist = gesim.graph_entropy_similarity_batch(target_mol, mol1_list)
            
            sim_1_matrix[i] = sslist
        
        # 保存相似度矩阵到CSV
        print("保存相似度矩阵到CSV文件...")
        try:
            # 保存active_1 vs active_0矩阵
            df_0 = pd.DataFrame(sim_0_matrix, 
                               index=[f"active1_{i}" for i in number1],
                               columns=[f"active0_{i}" for i in number0])
            df_0.to_csv(sim_0_csv_path)
            
            # 保存active_1 vs active_1矩阵
            df_1 = pd.DataFrame(sim_1_matrix,
                               index=[f"active1_{i}" for i in number1],
                               columns=[f"active1_{i}" for i in number1])
            df_1.to_csv(sim_1_csv_path)
            
            print(f"相似度矩阵已保存到: {sim_0_csv_path} 和 {sim_1_csv_path}")
        except Exception as e:
            print(f"保存CSV文件失败: {e}")
    
    print("开始构建最终矩阵...")
    
    for i in range(len(number1)):
        # 处理active_1 vs active_0相似度（取相似度最大的K个 - 最相似的）
        sim_scores_0 = sim_0_matrix[i]
        
        if torch.cuda.is_available():
            try:
                sim_tensor = torch.tensor(sim_scores_0, device=device)
                values, indices = torch.topk(sim_tensor, k, largest=True)  # 取最大的K个（最相似的）
                positions = indices.cpu().numpy().tolist()
            except:
                # GPU失败回退到CPU
                indexed = list(enumerate(sim_scores_0))
                indexed.sort(key=lambda x: x[1], reverse=True)  # 降序排序
                top_k = indexed[:k]
                positions = [idx for idx, val in top_k]
        else:
            # CPU排序
            indexed = list(enumerate(sim_scores_0))
            indexed.sort(key=lambda x: x[1], reverse=True)  # 降序排序
            top_k = indexed[:k]
            positions = [idx for idx, val in top_k]
        
        name0 = []
        for j in positions:
            name0.append(number0[j])
        for j in range(len(name0)):
            matrix[i, j] = name0[j]
        
        # 处理active_1 vs active_1相似度（排除自身后取相似度最小的K个 - 最不相似的）
        sim_scores_1 = sim_1_matrix[i].copy()
        sim_scores_1[i] = float('inf')  # 将自身设为最大值，确保不会被选中（因为我们要取最小的）
        
        if torch.cuda.is_available():
            try:
                sim_tensor = torch.tensor(sim_scores_1, device=device)
                values, indices = torch.topk(sim_tensor, k, largest=False)  # 取最小的K个（最不相似的）
                positions = indices.cpu().numpy().tolist()
            except:
                # GPU失败回退到CPU
                indexed = list(enumerate(sim_scores_1))
                indexed.sort(key=lambda x: x[1], reverse=False)  # 升序排序
                top_k = indexed[:k]
                positions = [idx for idx, val in top_k]
        else:
            # CPU排序
            indexed = list(enumerate(sim_scores_1))
            indexed.sort(key=lambda x: x[1], reverse=False)  # 升序排序
            top_k = indexed[:k]
            positions = [idx for idx, val in top_k]
        
        name1 = []
        for j in positions:
            name1.append(number1[j])
        for j in range(len(name1)):
            matrix[i, j+k] = name1[j]
        matrix[i, 2*k] = number1[i]
    
    return matrix


def matrix(A,k):
    # k = 50
    active_values, active_0_name, active_1_name = train.extract_active_property(A+".sdf")
    Matrix = np.zeros((len(active_1_name), 2*k+1), dtype=list)  # 矩阵380*(8965+380)
    Matrix = different_active_matrix(active_values, active_0_name, active_1_name, Matrix, A, k)
    # matrix的相关结构为前k个是相反性质的0（最相似的），后K个是相同性质的1（最不相似的），最后一个是自身
    return Matrix, k


if __name__ == '__main__':
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.get_device_name()}")
    else:
        print("CUDA不可用，将使用CPU")
    files=[]
    for i in range(1):
        # for i in bi:
            # i=3
            if i == 0:
                bili=[8,1,1]
            elif i == 1:
                bili=[6,2,2]
            elif i == 2:
                bili=[4,3,3]
            elif i == 3:
                bili=[2,4,4]
            filesss=[
                "train/nr-ar",
                'train/nr-ahr',
                'train/nr-ar-lbd',
                'train/nr-aromatase',
                'train/nr-er',
                'train/nr-er-lbd',
                'train/nr-ppar-gamma',
                'train/sr-are',
                'train/sr-atad5',
                'train/sr-hse',
                'train/sr-mmp',
                "train/sr-p53",
                "train/BBBP",
                "train/bace",
                "train/clintox",
                'train/HIV',
                ]
            
            for fil in filesss:
                files.append(fil.replace('train/','ours/train/'+str(bili[0])+str(bili[1])+str(bili[2])))
    print(files)
    for fil in files:
        M, K = matrix(fil,1)
    # a = M[5, :]
    # print(a)