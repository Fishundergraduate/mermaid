import pandas as pd
import glob,re,os
from tqdm import tqdm
paths = glob.glob("log_5zdp_2d_2/concat*/present/more0.8.txt")
for dataDir in tqdm(paths):
    if False:
        df = pd.read_csv(dataDir+"/present/ligands.txt",names=["smi"])
        df2 = pd.read_csv(dataDir+"/present/scores.txt",names=["D","Q","T"])
        df3 = pd.concat([df,df2],axis=1)
        df4=df3.drop_duplicates()
        df4.to_csv(dataDir+"/present/dropLigs.csv",index=False)

    if False:
        df = pd.read_csv(dataDir+"/present/pareto.csv",names=["D","Q","T"])
        df = df.sort_values(["D"])
        df = df.drop_duplicates(subset=["Q","T"],keep="last")
        df = df.sort_index()
        df.to_csv(dataDir+"/present/dup_pareto.csv",index=False,header=False)
    if True:
        df = pd.read_csv(dataDir,header=None,names=["smi","d","q"])
        df = df.sort_values(["d"])
        df = df.drop_duplicates(subset=["smi"],keep="last")
        df = df.sort_index()
        df.to_csv(dataDir,index=False,header=True)