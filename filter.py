import pandas as pd
from rdkit.Chem import Descriptors
from rdkit import Chem
from tqdm import tqdm
import numpy as np
from glob import glob

paths = glob("log_8gcy_2d_2/data*/present")
for path in tqdm(paths):
    df1 = pd.read_csv(f"{path}/ligands.txt",header=None, names=["smi"])
    df2 = pd.read_csv(f"{path}/scores.txt",header=None, names=["d", "q"])
    #df2.drop("t",axis=1)
    df3 = pd.concat([df1,df2],axis=1)
    cols = ["d"]#, "q"]
    #print(df3.loc[(df3[cols]>0.8).all(axis=1)].head)
    df4 = df3.loc[(df3[cols]>0.8).all(axis=1)]
    df4 = df4.drop_duplicates(subset=["smi"])
    for i, ent in df4.iterrows():
        m = Chem.MolFromSmiles(ent.smi)
        wt = Descriptors.ExactMolWt(m)
        if wt < 250 or wt > 750:
            continue
        info = m.GetRingInfo()
        ringsize = []
        for ring in info.BondRings():
            ringsize.append(len(ring))
        if np.max(np.array(ringsize)) >= 7:
            continue
        elif np.min(np.array(ringsize)) <= 4:
            continue
        else:
            with open(f"{path}/more0.8.txt","a") as f:
                f.write(f"{ent.smi},{ent.d},{ent.q}\n")
                f.flush()
    #df4.to_csv(f"{prot}.more0.9.txt",index=False)#,header=None,index=None)