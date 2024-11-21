import subprocess
import re
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED
import os
parser = argparse.ArgumentParser(description='search molecular')


parser.add_argument('smi',help='smiles')
#parser.add_argument('--skipLines',default=0)
args = parser.parse_args()

smi = args.smi
#skipLines = args.skipLines

proteinName="5ZYK"
GLIDE_PIPELINE="/gs/hs0/tga-science/shared/LigPrep-Glide-Pipeline"
#index = str(re.sub(r"\/", "", smi))


#df2 = pd.read_csv(smi+"./present/filtered.csv", names=["smi","d","q"])
#passedSmis = list(df2.smi)
subprocess.run(f"echo \"{smi}\"  > tmp.smi",shell=True)
subprocess.run(f"cp tmp.smi {GLIDE_PIPELINE}/input/input_tmp.smi",shell=True)
#print(f"OK\t{i}\t{smi}")
#break
index = 100
#subprocess.run(f"mkdir $GLIDE_PIPELINE/{index}",shell=True)
subprocess.run(f"bash {GLIDE_PIPELINE}/src/pipeline_mult.sh input_tmp.smi {str(index)}",shell=True)
#subprocess.run(f"mv {GLIDE_PIPELINE}/{index}/glide-{proteinName}dw_constrainted.csv .",shell=True)
df = pd.read_csv(f"{GLIDE_PIPELINE}/{index}/glide-{proteinName}dw_constrainted.csv")
m = float(df["r_i_docking_score"][0])
print(f"Docking Score__{m}")
if m < -5:
    subprocess.run(f"sleep 20",shell=True)
    subprocess.run(f"cp {GLIDE_PIPELINE}/result/glide-{proteinName}dw_constrainted_pv.maegz .",shell=True)
#break
