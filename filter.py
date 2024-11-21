import pandas as pd
from rdkit.Chem import Descriptors,AllChem,QED
from rdkit import Chem
from tqdm import tqdm
import numpy as np
from glob import glob
import os
#paths = glob("log_5zyk_2d_8/log_data/data*/present")
paths = glob("log_5zdp_2d_3/log_data/data*/present")

def _morganFP(m):
    return AllChem.GetMorganFingerprintAsBitVect(m,2,2048)

#df_temp = pd.read_csv("Data/input/actives_20230516.smi",header=None,names=["smi"])
df_temp = pd.read_csv("Data/input/actives_5zdp_20230711.smi",header=None,names=["smi"])
_li_mol = list(map(Chem.MolFromSmiles,df_temp.smi))
li_fp = list(map(_morganFP,_li_mol))
def _qed_reward(smiles):
    _mol = Chem.MolFromSmiles(smiles)
    if _mol is None:
        return 0
    
    _q = QED.qed(_mol)
    return _q
def _arr_tanimoto(__m):
    __fp = _morganFP(__m)
    __li_tanimoto = max(list(map(Chem.DataStructs.TanimotoSimilarity,li_fp, [__fp for _ in li_fp])))
    return __li_tanimoto
#paths = glob("log_5zdp_2d_2/data*/present")
for path in tqdm(paths):
    with open(f"{path}/more0.8.txt","w") as f:
            f.write(f"")
            f.flush()
    if not os.path.exists(f"{path}/ligands.txt"):
        print(f"404: Not Found: {path}/ligands.txt")
        continue
    df1 = pd.read_csv(f"{path}/ligands.txt",header=None, names=["smi"])
    df2 = pd.read_csv(f"{path}/scores.txt",header=None, names=["d", "q"])
    #df2.drop("t",axis=1)
    cols = ["d"]#,"q"]
    df2["q"] = list(map(_qed_reward,df1.smi))
    df3 = pd.concat([df1,df2],axis=1)
    #print(df3.loc[(df3[cols]>0.8).all(axis=1)].head)
    df4 = df3.loc[(df3[cols]>0.8).all(axis=1)]
    df4 = df4.drop_duplicates(subset=["smi"])
    for i, ent in df4.iterrows():
        m = Chem.MolFromSmiles(ent.smi)
        if m is None:
            continue
        wt = Descriptors.ExactMolWt(m)
        # MOL Weight Constraint
        if wt < 250 or wt > 750:
            continue
        # MOL substructure 1 constraint
        q = Chem.MolFromSmarts("CS")
        if m.HasSubstructMatch(q):
            continue
        q = Chem.MolFromSmarts("NN")
        if m.HasSubstructMatch(q):
            continue
        q = Chem.MolFromSmarts("NO")
        if m.HasSubstructMatch(q):
            continue
        # TANIMOTO SIM Constraint
        if _arr_tanimoto(m) > 0.7:
            continue
        # MOL RingSize Constraint
        info = m.GetRingInfo()
        ringsize = []
        for ring in info.BondRings():
            ringsize.append(len(ring))
        if len(ringsize) > 0 and np.max(np.array(ringsize)) >= 8:
            continue
        elif len(ringsize) > 0 and np.min(np.array(ringsize)) <= 4:
            continue
        else:
            with open(f"{path}/more0.8.txt","a") as f:
                f.write(f"{ent.smi},{ent.d},{ent.q}\n")
                f.flush()
    #df4.to_csv(f"{prot}.more0.9.txt",index=False)#,header=None,index=None)