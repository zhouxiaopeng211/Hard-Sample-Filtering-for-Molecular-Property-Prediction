import SDF_dispose
from rdkit import Chem
from tqdm import tqdm
import os
import random
from number import extract_active_property 

def is_valid_molecule(mol):
    """检查分子是否有效"""
    if mol is None:
        return False
    graph = SDF_dispose.molecule_to_pyg_graph(mol)
    if len(graph.edge_index) == 0:
        return False
    try:
        # 检查分子是否可净化（化学合理性）
        Chem.SanitizeMol(mol)
        # 确保分子至少包含一个原子
        return mol.GetNumAtoms() > 0
    except:
        return False

def filter_sdf(input_file, output_file):
    """过滤无效分子并保存到新文件"""
    # 读取SDF文件
    suppl = Chem.SDMolSupplier(input_file)
    # 创建输出文件写入器
    writer = Chem.SDWriter(output_file)

    skipped_count = 0
    total_count = 0

    for idx, mol in enumerate(suppl):
        total_count += 1
        try:
            if is_valid_molecule(mol):
                writer.write(mol)
            else:
                skipped_count += 1
                print(f"跳过无效分子 #{idx + 1}")

        except Exception as e:
            skipped_count += 1
            print(f"处理分子 #{idx + 1} 时发生错误: {str(e)}")

    writer.close()
    print(f"\n处理完成！共处理 {total_count} 个分子")
    print(f"保留有效分子: {total_count - skipped_count}")
    print(f"删除无效分子: {skipped_count}")

def remove_nth_molecule(input_file, output_file, n):
    """删除指定索引的分子（索引从 0 开始）"""
    supplier = Chem.SDMolSupplier(input_file)
    writer = Chem.SDWriter(output_file)

    for idx, mol in enumerate(supplier):
        if idx != n:  # 跳过第n个分子
            if mol:  # 同时检查有效性
                writer.write(mol)

    writer.close()

def add_name_property_to_sdf(input_sdf, output_sdf):
    """
    为SDF文件中的每个分子添加name属性，按顺序命名为name0, name1, ...

    参数:
    input_sdf -- 输入SDF文件路径
    output_sdf -- 输出SDF文件路径
    """
    # 读取SDF文件
    suppl = Chem.SDMolSupplier(input_sdf)
    molecules = [mol for mol in suppl if mol is not None]

    print(f"从文件 {input_sdf} 中读取了 {len(molecules)} 个有效分子")

    # 添加name属性
    writer = Chem.SDWriter(output_sdf)

    for idx, mol in tqdm(enumerate(molecules), total=len(molecules), desc="添加name属性"):
        mol_name = f"name{idx}"
        mol.SetProp("name", mol_name)
        writer.write(mol)

    writer.close()
    print(f"已将修改后的分子保存到 {output_sdf}")

from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import json
import os
import random
from rdkit import Chem

def split_sdf_dataset(input_sdf, out_sdf):
    """
    使用与 MolCLR 完全一致的 Murcko Scaffold Split 逻辑进行 8:1:1 划分。
    """
    seed = 55
    # 设置 valid 和 test 的比例，对齐 MolCLR
    valid_size = 0.1
    test_size = 0.1
    train_size = 1.0 - valid_size - test_size

    # 1) 读取分子
    suppl = Chem.SDMolSupplier(input_sdf)
    molecules = [mol for mol in suppl if mol is not None]
    total = len(molecules)
    print(f"[scaffold split] 读取文件 {input_sdf}，有效分子数 = {total}")
    if total == 0:
        raise ValueError("输入 SDF 中没有有效分子 (None)")

    # 2) 生成 scaffolds，完全对齐 dataset_test.py
    scaffolds = {}
    for ind, mol in enumerate(molecules):
        try:
            # 尝试生成 SMILES 格式的骨架
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            # 容错处理
            scaffold = Chem.MolToSmiles(mol, canonical=True)
            
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # 3) 排序：按集合大小降序，若大小相同则按第一个索引升序 (Tie-breaker)
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # 4) 分配阈值，完全对齐 dataset_test.py 的累积判断逻辑
    train_cutoff = train_size * total
    valid_cutoff = (train_size + valid_size) * total
    train_idx, val_idx, test_idx = [], [], []

    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(val_idx) + len(scaffold_set) > valid_cutoff:
                test_idx += scaffold_set
            else:
                val_idx += scaffold_set
        else:
            train_idx += scaffold_set

    # 为了写入时的顺序性进行排序
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    print(f"[scaffold split] 划分结果: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    # 5) 保存 split 索引文件
    split_name = out_sdf.replace('.sdf', '')
    os.makedirs('splits', exist_ok=True)
    save_path = f'splits/{split_name}_scaffold_split_seed{seed}.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({
            "seed": seed,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "counts": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
        }, f, indent=2, ensure_ascii=False)

    # 6) 根据划分写出 SDF
    train_file = '/root/autodl-tmp/13_zhouxiaopeng/ours/train/811' + out_sdf
    val_file = '/root/autodl-tmp/13_zhouxiaopeng/ours/validation/' + out_sdf
    test_file = '/root/autodl-tmp/13_zhouxiaopeng/ours/test/' + out_sdf

    for file_path, indices in zip([train_file, val_file, test_file], [train_idx, val_idx, test_idx]):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        writer = Chem.SDWriter(file_path)
        for idx in indices:
            writer.write(molecules[idx])
        writer.close()

    print(f"[scaffold split] 已保存: {train_file}, {val_file}, {test_file}")
    # return train_idx, val_idx, test_idx

# def split_sdf_dataset(input_sdf,out_sdf):
#     """
#     将SDF数据集按8:1:1的比例随机划分为训练集、验证集和测试集
#     """
#     # 读取所有分子
#     suppl = Chem.SDMolSupplier(input_sdf)
#     molecules = [mol for mol in suppl if mol is not None]
#     print(molecules[0])
#     total = len(molecules)
#     print(f"总共有 {total} 个分子")
    
#     # 随机打乱分子顺序
#     random.shuffle(molecules)
    
#     # 计算每个部分的数量
#     test_count = (total // 10)      # 十分之一给测试集
#     val_count = (total // 10)       # 十分之一给验证集
#     train_count811 = total - test_count - val_count  # 剩下的给训练集
#     train_count244 = train_count811 // 4  
#     train_count433 = train_count811 // 2
#     train_count622 = train_count811 - train_count244
#     print(f"训练集811: {train_count811} 个")
#     # print(f"训练集622: {train_count622} 个")
#     # print(f"训练集433: {train_count433} 个")
#     # print(f"训练集244: {train_count244} 个")
#     print(f"验证集: {val_count} 个")  
#     print(f"测试集: {test_count} 个")
    
#     # 分割分子列表（现在是随机的）
#     val_molecules = molecules[0: val_count]
#     test_molecules = molecules[  val_count:  val_count + test_count]
#     train_molecules811 = molecules[val_count + test_count:]
#     train_molecules244 = molecules[val_count + test_count:val_count + test_count + train_count244]
#     train_molecules433 = molecules[val_count + test_count:val_count + test_count + train_count433]
#     train_molecules622 = molecules[val_count + test_count:val_count + test_count + train_count622]

#     # 保存训练集
#     train_file = '/root/autodl-tmp/13_zhouxiaopeng/ours/train/811'+out_sdf
#     if os.path.exists(train_file):
#         os.remove(train_file)
#     writer = Chem.SDWriter(train_file)
#     for mol in train_molecules811:
#         writer.write(mol)
#     writer.close()
#     # train_file = '/root/autodl-tmp/13_zhouxiaopeng/ours/train/244'+out_sdf
#     # if os.path.exists(train_file):
#     #     os.remove(train_file)
#     # writer = Chem.SDWriter(train_file)
#     # for mol in train_molecules244:
#     #     writer.write(mol)
#     # writer.close()
#     # train_file = '/root/autodl-tmp/13_zhouxiaopeng/ours/train/433'+out_sdf
#     # if os.path.exists(train_file):
#     #     os.remove(train_file)
#     # writer = Chem.SDWriter(train_file)
#     # for mol in train_molecules433:
#     #     writer.write(mol)
#     # writer.close()
#     # train_file = '/root/autodl-tmp/13_zhouxiaopeng/ours/train/622'+out_sdf
#     # if os.path.exists(train_file):
#     #     os.remove(train_file)
#     # writer = Chem.SDWriter(train_file)
#     # for mol in train_molecules622:
#     #     writer.write(mol)
#     # writer.close()
#     # 保存验证集
#     val_file = '/root/autodl-tmp/13_zhouxiaopeng/ours/validation/'+out_sdf
#     if os.path.exists(val_file):
#         os.remove(val_file)
#     writer = Chem.SDWriter(val_file)
#     for mol in val_molecules:
#         writer.write(mol)
#     writer.close()
    
#     # 保存测试集
#     test_file = '/root/autodl-tmp/13_zhouxiaopeng/ours/test/'+out_sdf
#     if os.path.exists(test_file):
#         os.remove(test_file)
#     writer = Chem.SDWriter(test_file)
#     for mol in test_molecules:
#         writer.write(mol)
#     writer.close()
    
#     # print(f"已保存: {train_file}, {val_file}, {test_file}")

def sdfmake(files):
    
    for i in range(len(files)):
        print(f"\n处理数据集 {i+1}/{len(files)}: {files[i]}")
        
        input_sdf ='/root/autodl-tmp/13_zhouxiaopeng/date/'+ files[i] + '.sdf'        # 原始输入文件
        filtered_sdf = "/root/autodl-tmp/13_zhouxiaopeng/date/B.sdf"               # 过滤后的临时文件
        final_sdf = input_sdf                # 最终添加name属性的输出文件（覆盖原始）
        
        # 步骤1: 过滤无效分子
        print("步骤1: 过滤无效分子")
        filter_sdf(input_sdf, filtered_sdf)

        # 步骤2: 划分数据集
        print("步骤2: 划分数据集")
        split_sdf_dataset(filtered_sdf,files[i]+ '.sdf')

        # 步骤3: 添加name属性
        print("步骤3: 添加name属性")
        add_name_property_to_sdf('/root/autodl-tmp/13_zhouxiaopeng/ours/train/811'+files[i] + '.sdf','/root/autodl-tmp/13_zhouxiaopeng/ours/train/811'+files[i]  + '.sdf')
        # add_name_property_to_sdf('/root/autodl-tmp/13_zhouxiaopeng/ours/train/622'+files[i] + '.sdf','/root/autodl-tmp/13_zhouxiaopeng/ours/train/622'+files[i]  + '.sdf')
        # add_name_property_to_sdf('/root/autodl-tmp/13_zhouxiaopeng/ours/train/433'+files[i] + '.sdf','/root/autodl-tmp/13_zhouxiaopeng/ours/train/433'+files[i]  + '.sdf')
        # add_name_property_to_sdf('/root/autodl-tmp/13_zhouxiaopeng/ours/train/244'+files[i] + '.sdf','/root/autodl-tmp/13_zhouxiaopeng/ours/train/244'+files[i]  + '.sdf')

        add_name_property_to_sdf('/root/autodl-tmp/13_zhouxiaopeng/ours/validation/'+files[i] + '.sdf','/root/autodl-tmp/13_zhouxiaopeng/ours/validation/'+files[i]  + '.sdf')
        add_name_property_to_sdf('/root/autodl-tmp/13_zhouxiaopeng/ours/test/'+files[i] + '.sdf','/root/autodl-tmp/13_zhouxiaopeng/ours/test/'+files[i] + '.sdf')
        
        # 清理临时文件
        if os.path.exists(filtered_sdf):
            os.remove(filtered_sdf)
        
        print(f"数据集 {files[i]} 处理完成！")
        print("-" * 50)


#!/usr/bin/env python3
"""
检查并删除SDF文件内部重复分子
"""

import sys
import os
from collections import defaultdict
from rdkit import Chem

#==============================================
def invert_sdf_properties(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
                # 检查是否是只包含0或1的行（可能前后有空格）
            stripped_line = line.strip()
                
            if stripped_line == '0':
                f_out.write('1\n')
            elif stripped_line == '1':
                f_out.write('0\n')
            else:
                f_out.write(line)
#==============================================

def remove_sdf_duplicates(sdf_file):
    """
    删除SDF文件内部的重复分子并保存新文件
    
    Args:
        sdf_file: SDF文件路径
    
    Returns:
        tuple: (是否有重复分子, 去重后的分子数量)
    """
    # 读取SDF文件
    supplier = Chem.SDMolSupplier(sdf_file)
    
    # 使用InChIKey作为唯一标识符
    inchikey_dict = defaultdict(list)
    total_count = 0
    valid_count = 0
    
    print(f"检查文件: {sdf_file}")
    print("=" * 60)
    
    # 处理每个分子
    for idx, mol in enumerate(supplier):
        total_count += 1
        
        if mol is None:
            continue
            
        valid_count += 1
        
        # 获取分子名称
        mol_name = f"mol_{idx}"
        if mol.HasProp("_Name"):
            mol_name = mol.GetProp("_Name")
        elif mol.HasProp("name"):
            mol_name = mol.GetProp("name")
        
        # 生成InChIKey
        try:
            inchikey = Chem.MolToInchiKey(mol)
            if inchikey:
                inchikey_dict[inchikey].append((idx, mol_name, mol))
        except:
            # 如果InChIKey失败，使用SMILES作为备选
            try:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                inchikey_dict[smiles].append((idx, mol_name, mol))
            except:
                pass
    
    # 统计结果
    print(f"总分子数: {total_count}")
    print(f"有效分子: {valid_count}")
    print(f"唯一分子: {len(inchikey_dict)}")
    
    # 检查重复并确定要保留的分子
    duplicates_found = False
    duplicate_groups = 0
    total_duplicates = 0
    molecules_to_keep = []  # 存储要保留的分子
    
    for inchikey, molecules in inchikey_dict.items():
        if len(molecules) > 1:
            duplicates_found = True
            duplicate_groups += 1
            total_duplicates += len(molecules) - 1
            
            print(f"\n重复组 {duplicate_groups}:")
            print(f"标识符: {inchikey}")
            
            # 保留第一个分子，删除其他重复分子
            keep_idx, keep_name, keep_mol = molecules[0]
            molecules_to_keep.append(keep_mol)
            
            print(f"  ✓ 保留: 索引 {keep_idx}: {keep_name}")
            for idx, name, mol in molecules[1:]:
                print(f"  ✗ 删除: 索引 {idx}: {name}")
        else:
            # 没有重复的分子直接保留
            idx, name, mol = molecules[0]
            molecules_to_keep.append(mol)
    
    # 输出总结
    print("\n" + "=" * 60)
    if duplicates_found:
        print(f"❌ 发现重复分子!")
        print(f"重复组数: {duplicate_groups}")
        print(f"删除的重复分子数: {total_duplicates}")
        print(f"保留的唯一分子数: {len(molecules_to_keep)}")
        
        # 创建去重后的SDF文件
        output_file = sdf_file.replace('2.sdf', '.sdf')
        writer = Chem.SDWriter(output_file)
        
        for mol in molecules_to_keep:
            writer.write(mol)
        writer.close()
        
        print(f"✅ 已创建去重后的文件: {output_file}")
        return True, len(molecules_to_keep)
    else:
        print("✅ 没有发现重复分子!")
        # 即使没有重复，也创建一份副本
        output_file = sdf_file.replace('2.sdf', '.sdf')
        writer = Chem.SDWriter(output_file)
        
        for inchikey, molecules in inchikey_dict.items():
            idx, name, mol = molecules[0]
            writer.write(mol)
        writer.close()
        
        print(f"✅ 已创建无重复文件: {output_file}")
        return False, len(molecules_to_keep)

if __name__ == "__main__":
    xinzhifanzhuan = False  # 是否反转SDF文件中的性质
    if xinzhifanzhuan :
        input_sdf = "date/BBBP.sdf"    # 输入SDF文件
        output_sdf = "date/BBBP1.sdf"  # 输出SDF文件
        invert_sdf_properties(input_sdf, output_sdf)
        print(f"性质反转完成！输出文件: {output_sdf}")

        os.path.exists(input_sdf) and os.remove(input_sdf)
        os.path.exists(output_sdf) and os.rename(output_sdf, input_sdf)
    files = [
            "nr-ahr",
            'nr-ar',
            'nr-ar-lbd',
            'nr-aromatase',
            'nr-er',
            'nr-er-lbd',
            'nr-ppar-gamma',
            'sr-are',
            'sr-atad5',
            'sr-hse',
            'sr-mmp',
            "sr-p53",
            # 'BBBP',
            # 'HIV',
            # "bace",
            # "clintox",
             ]
    print("开始处理SDF文件去重...")
    print("=" * 80)
    for file in files:
        sdf_file = 'date/'+file + '2.sdf'
        if os.path.exists(sdf_file):
            has_duplicates, unique_count = remove_sdf_duplicates(sdf_file)
            print(f"\n处理完成: {sdf_file}")
            print("-" * 80)
        else:
            print(f"❌ 文件不存在: {sdf_file}")
    
    print("\n所有文件处理完成!")
    sdfmake(files)




