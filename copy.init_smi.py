import pandas as pd
from glob import glob
from tqdm import tqdm
df = pd.read_csv("log_5zdp_2d_2/concat1/present/glide.passed_0708.csv",header=0)

paths = glob("log_5zdp_2d_3/log_data/data*/input/next.smi")

for i in tqdm(range(len(paths))):
    with open(f"log_5zdp_2d_3/log_data/data{i}/input/next.smi","w") as f:
        f.write(df.smi[i//10])
