# 这个是为了将分子信息转化为对象的代码


from rdkit import Chem
from torch_geometric.data import Data
import torch
import numpy as np
#-----------------------------------------------------------------------------------------------------------------------
#turn to smile
def sdf_to_smiles(sdf_path):
    # 读取 SDF 文件
    supplier = Chem.SDMolSupplier(sdf_path)
    smiles_list = []

    # 遍历所有分子
    for mol in supplier:
        if mol is not None:  # 跳过无效分子
            # 生成标准 SMILES（默认规范化）
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        else:
            print("Warning: 跳过无法解析的分子")

    return smiles_list

#-----------------------------------------------------------------------------------------------------------------------
#turn to graph


# 常见原子表（one-hot 用），超出范围归为 'other'
COMMON_ATOMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
def atom_one_hot(atomic_num, vocab=COMMON_ATOMS):
    vec = [1.0 if atomic_num == a else 0.0 for a in vocab]
    if sum(vec) == 0:
        # other bucket
        vec.append(1.0)
    else:
        vec.append(0.0)
    return vec  # length = len(vocab)+1

def one_hot_index(x, choices):
    vec = [1.0 if x == c else 0.0 for c in choices]
    if sum(vec) == 0:
        vec.append(1.0)  # unknown
    else:
        vec.append(0.0)
    return vec

def molecule_to_pyg_graph(mol, atom_vocab=COMMON_ATOMS):
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        feats = []
        # 1) atomic number one-hot + other
        feats.extend(atom_one_hot(atom.GetAtomicNum(), vocab=atom_vocab))

        # 2) degree one-hot (0..5, rest->other)
        feats.extend(one_hot_index(atom.GetDegree(), list(range(6))))

        # 3) formal charge (int) as scalar (can be negative)
        feats.append(float(atom.GetFormalCharge()))

        # 4) number of implicit/explicit Hs (0..4)
        feats.extend(one_hot_index(atom.GetTotalNumHs(), list(range(5))))

        # 5) hybridization one-hot
        hyb = atom.GetHybridization()
        hyb_choices = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.S
        ]
        feats.extend(one_hot_index(hyb, hyb_choices))

        # 6) aromatic (bool), in ring (bool), chiral tag (one-hot), mass normalized
        feats.append(1.0 if atom.GetIsAromatic() else 0.0)
        feats.append(1.0 if atom.IsInRing() else 0.0)

        chiral = atom.GetChiralTag()
        chiral_choices = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
        ]
        feats.extend(one_hot_index(chiral, chiral_choices))

        feats.append(atom.GetMass() / 100.0)  # scale mass

        atom_features.append(feats)

    x = torch.tensor(atom_features, dtype=torch.float)  # [num_nodes, num_node_features]

    # edges
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        # bond type one-hot: single,double,triple,aromatic
        bt = bond.GetBondType()
        bt_choices = [Chem.rdchem.BondType.SINGLE,
                      Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE,
                      Chem.rdchem.BondType.AROMATIC]
        bt_onehot = [1.0 if bt == c else 0.0 for c in bt_choices]
        if sum(bt_onehot) == 0:
            bt_onehot.append(1.0)  # other
        else:
            bt_onehot.append(0.0)

        # other bond features
        bond_feats = []
        bond_feats.extend(bt_onehot)
        bond_feats.append(1.0 if bond.GetIsConjugated() else 0.0)
        bond_feats.append(1.0 if bond.IsInRing() else 0.0)
        # stereo: one-hot
        stereo = bond.GetStereo()
        stereo_choices = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE
        ]
        bond_feats.extend(one_hot_index(stereo, stereo_choices))

        # add both directions
        edge_index.append((a1, a2))
        edge_index.append((a2, a1))
        edge_attr.append(bond_feats)
        edge_attr.append(bond_feats)

    if len(edge_index) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.empty((0, len(bond_feats)), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)





