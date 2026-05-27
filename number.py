from rdkit import Chem

def count_sdf_molecules_accurate(sdf_file):
    supplier = Chem.SDMolSupplier(sdf_file)
    valid_count = 0
    for mol in supplier:
        if mol is not None:  # 仅统计有效分子
            valid_count += 1
    return valid_count

#-------------------------------

# def count_sdf_molecules_active(sdf_file):
#     count = 0
#     with open(sdf_file, 'r') as f:
#         # for line in f:
#         #     if line.strip() == '>  <Active>':
#         #         count += 1
#     return count


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


if __name__ == "__main__":
    for i in range(1):
        if i == 0:
            bili=[8,1,1]
        elif i == 1:
            bili=[6,2,2]
        elif i == 2:
            bili=[4,3,3]
        elif i == 3:
            bili=[2,4,4]
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
                # 'train/HIV',
                'train/bace',
                # "train/BBBP",
                # "train/clintox",
                # 'train/bace_no_balance1',
                # 'train/bace_no_balance2',
                # 'train/bace_no_balance3',
                ]
        files=[]
        for fil in filess:
            files.append(fil.replace('train/','ours/train/'+str(bili[0])+str(bili[1])+str(bili[2])))
            # files.append(fil.replace('train/','ours/test/'))
            # files.append(fil.replace('train/','ours/validation/'))
        for file in files:
            active_values,active_0_name,active_1_name = extract_active_property(file+'.sdf')
            print('='*20)
            print(file)
            print(len(active_values))
            print("Active 属性列表:", active_values[:22])
            print("其中标记为0的有",len(active_0_name),'个标记为1的有',len(active_1_name))
            # print("1/n+1/m=",((len(active_0_name)/len(active_1_name)+1)/len(active_1_name))/2)
            
    # # 使用示例
    # # writer = Chem.SDWriter('nr-ar2.sdf')
    # active_values,active_0_name,active_1_name = extract_active_property("date/bace.sdf")
    # print(len(active_values))
    # print("Active 属性列表:", active_values)
    # print("其中标记为0的有",len(active_0_name),'个标记为1的有',len(active_1_name))#0有8965,1有380
    # print('标记为0的有',active_0_name,'\n标记为1的有',active_1_name)
    # # print(active_0_munber+active_1_munber)
    # # 使用示例
    # num_valid = count_sdf_molecules_accurate("date/bace.sdf")
    # print(f"SDF 文件包含 {num_valid} 个有效分子")