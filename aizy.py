## must run on aizynth-env
from aizynthfinder.aizynthfinder import AiZynthFinder
import pandas as pd
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataDir')
parser.add_argument('-n',type=int, default=0)
args = parser.parse_args()
dataDir = args.dataDir
n = args.n
# set aizynthfinder
find = AiZynthFinder(configfile="../aizynth/config.yml")
find.stock.select("namiki")
find.expansion_policy.select("uspto")
find.filter_policy.select("uspto")

#ind: float = 3.5

""" if not os.path.exists(f"{dataDir}/present/merge.csv"):
    df1 = pd.read_csv(f"{dataDir}/present/ligands.txt",header=None, names=["smi"])
    df2 = pd.read_csv(f"{dataDir}/present/scores.txt",header=None, names=["d", "q"])
    df3 = pd.concat([df1,df2],axis=1)
    cols=["d","q"]
    df3.loc[(df3[cols]>0.8).all(axis=1)].to_csv(f"{dataDir}/present/merge.csv",index=False)
 """
df = pd.read_csv(f"{dataDir}/present/more0.8.txt",header=None,names=["smi","d","q"])
#print(df.head)
#dfA = pd.DataFrame()
if df.shape[0] < n:
    exit(0)
for i, ent in tqdm(df.iterrows()):
    if i < n:
        continue
    if i == 150:
        del find
        find = AiZynthFinder(configfile="../aizynth/config.yml")
        find.stock.select("namiki")
        find.expansion_policy.select("uspto")
        find.filter_policy.select("uspto")
    find.target_smiles=ent.smi
    find.tree_search()#show_progress=True)
    find.build_routes()
    if find.extract_statistics()["is_solved"]:
        print(ent.smi)
        with open(f"{dataDir}/present/aizy.pass.csv",'a') as fo:
            fo.write(f"{ent.smi}, {ent.d}, {ent.q}\n")
            fo.flush()
        #dfA = pd.concat([dfA, ent])
        #break
#print(dfA.head)
#dfA.to_csv(f"sa{ind}.aizy.pass.csv",index=False)