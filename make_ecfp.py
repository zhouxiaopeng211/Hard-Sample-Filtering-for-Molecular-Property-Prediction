import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from tqdm import tqdm

def generate_and_save_ecfp(sdf_file, output_pt_file, radius=2, nBits=2048):
    supplier = Chem.SDMolSupplier(sdf_file)
    ecfp_dict = {}
    
    print(f"正在为 {sdf_file} 计算 ECFP 指纹...")
    for idx, mol in enumerate(tqdm(supplier)):
        if mol is None: continue
        
        # 获取分子的唯一标识 (与你代码里的命名逻辑保持一致)
        if mol.HasProp("name"):
            mol_name = mol.GetProp("name")
        else:
            mol_name = str(idx)
            
        # 计算 Morgan 指纹
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        fp_array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        
        # 存入字典，转换为 Tensor
        ecfp_dict[mol_name] = torch.tensor(fp_array, dtype=torch.float32)
        
    # 保存为 .pt 文件
    torch.save(ecfp_dict, output_pt_file)
    print(f"成功保存 {len(ecfp_dict)} 个分子的 ECFP 指纹至 {output_pt_file}")

# 使用示例
if __name__ == "__main__":
    files = [
            # "nr-ahr",
            # 'nr-ar',
            # 'nr-ar-lbd',
            # 'nr-aromatase',
            # 'nr-er',
            # 'nr-er-lbd',
            # 'nr-ppar-gamma',
            # 'sr-are',
            # 'sr-atad5',
            # 'sr-hse',
            # 'sr-mmp',
            # "sr-p53",
            'ABBBP',
            # 'HIV',
            # "bace",
            # "clintox",
             ]

    for file in files:
        datesate = file
        generate_and_save_ecfp("ours/train/811"+datesate+".sdf", "ours/train/811"+datesate+".pt")
        generate_and_save_ecfp("ours/test/"+datesate+".sdf", "ours/test/"+datesate+".pt")
        generate_and_save_ecfp("ours/validation/"+datesate+".sdf", "ours/validation/"+datesate+".pt")
        # import torch, numpy as np

        # fp_pt = "ours/train/811bace.pt"   # <- 改成你实际的 path
        # d = torch.load(fp_pt)
        # print("type(d)=", type(d))
        # keys = list(d.keys())[:8]
        # print("sample keys:", keys)
        # # show shapes / sum
        # for k in keys:
        #     v = d[k]
        #     try:
        #         a = v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
        #         print(k, "shape:", a.shape, "sum:", int(a.sum()))
        #     except Exception as e:
        #         print("sample", k, "-> cannot inspect:", e)
        # # check typical size
        # first = next(iter(d.values()))
        # try:
        #     first_len = first.numel() if hasattr(first, 'numel') else len(first)
        #     print("first fingerprint length:", first_len)
        # except:
        #     print("cannot get length of first fingerprint, inspect above")