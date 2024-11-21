import pandas as pd
df = pd.read_csv("log_8gcy_2d/8gcy.more0.8.txt",header=None,names=["smi","d","q"])

for i in range(4):
    df.iloc[i*2890:(i+1)*2890].to_csv(f"log_8gcy_2d/concat{i+1}/present/merge.csv",index=False)
