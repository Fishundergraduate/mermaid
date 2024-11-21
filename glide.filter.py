import subprocess
import re
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import QED
import os
parser = argparse.ArgumentParser(description='search molecular')


parser.add_argument('dataDir',help='path to data dir')
parser.add_argument('skipLines',default=0)
parser.add_argument('dataIndex')
args = parser.parse_args()

dataDir = args.dataDir
skipLines = args.skipLines
n=args.dataIndex
proteinName="5ZYK"
GLIDE_PIPELINE="/gs/hs0/tga-science/shared/LigPrep-Glide-Pipeline"
index = str(re.sub(r"\/", "", dataDir))

#if not os.path.exists(dataDir+"./present/filtered.csv"):
#    with open(dataDir+"./present/filtered.csv", "w") as f:
#        f.write("smi,d,q\n")
#df2 = pd.read_csv(dataDir+"./present/filtered.csv", names=["smi","d","q"])
#passedSmis = list(df2.smi)
if not os.path.exists(f"mkdir {GLIDE_PIPELINE}/{index}"):
    subprocess.run(f"mkdir {GLIDE_PIPELINE}/{index}",shell=True)

if not os.path.exists(dataDir+"./present/glide.passed.csv"):
    with open(dataDir+"./present/glide.passed.csv", "w") as f:
        f.write("smi,g,d,q\n")    
with open(f"{dataDir}./present/lig{n}.concat.txt","r") as f:
    for i in range(len(skipLines)):
        f.readline()
    for i,smi in tqdm(enumerate(f.readlines())):
        if smi in passedSmis:
            continue
        with open(dataDir+'./workspace/ligand.smi','w') as f1:
            f1.write(smi)
        subprocess.run(f"cp {dataDir}./workspace/ligand.smi {GLIDE_PIPELINE}/input/input_{str(index)}.smi",shell=True)
        #print(f"OK\t{i}\t{smi}")
        #break
        #subprocess.run(f"bash {GLIDE_PIPELINE}/src/pipeline_mult.sh input_{str(index)}.smi {str(index)}",shell=True)
        #subprocess.run(f"mv {GLIDE_PIPELINE}/{index}/glide-{proteinName}dw_{str(index)}.csv {dataDir}./workspace/",shell=True)
        #df = pd.read_csv(f"{dataDir}./workspace/glide-{proteinName}dw_{str(index)}.csv")
        #m = float(df["r_i_docking_score"][0])
        subprocess.run(f"bash {GLIDE_PIPELINE}/src/pipeline_mult.sh input_{str(index)}.smi {str(index)}",shell=True)
    #subprocess.run(f"cp {GLIDE_PIPELINE}/{index}/glide-{proteinName}dw_{str(index)}.csv {dataDir}./workspace/glide-{proteinName}dw_{str(index)}.csv",shell=True)
        subprocess.run(f"mv {GLIDE_PIPELINE}/{index}/glide-{proteinName}dw_constrainted_{str(index)}.csv {dataDir}./workspace/",shell=True)
        df = pd.read_csv(f"{dataDir}./workspace/glide-{proteinName}dw_constrainted_{str(index)}.csv")
        m = float(df["r_i_docking_score"][0])
        if m < -5:
            mol = Chem.MolFromSmiles(smi)

            qed = QED.qed(mol)
            with open(dataDir+"./present/filtered.csv","a") as f2:
                f2.write(smi.strip()+","+str(m)+","+str(qed)+"\n")
                passedSmis.append(smi)
            df_append = pd.DataFrame(data=[[smi,m,qed]],columns=["smi","d","q"])
            df2 = pd.concat([df2,df_append],ignore_index=True)
        #break
