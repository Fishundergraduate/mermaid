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
parser.add_argument('-skipLines',default=0,type=int)
args = parser.parse_args()

dataDir = args.dataDir
skipLines = args.skipLines
proteinName="5ZDP"
CONSTRAINT = "" if False else "_constrainted"
GLIDE_PIPELINE="/gs/hs0/tga-science/shared/LigPrep-Glide-Pipeline"
index = str(re.sub(r"\/", "", dataDir))


#if not os.path.exists(dataDir+"./present/glide.passed.csv"):
#    with open(dataDir+"./present/glide.passed.csv", "w") as f:
#        f.write("smi,g,d,q\n")
if not os.path.exists(dataDir+"./present/poses"):
    os.mkdir(dataDir+"./present/poses")

#df2 = pd.read_csv(dataDir+"./present/filtered2.csv", names=["smi","d","q"])
#df2 = pd.read_csv(dataDir+"./present/glide.passed.csv", names=["smi","g","d","q"])
#df_score = pd.read_csv(dataDir+"./present/scores.txt",names=["d","q","t"])
#passedSmis = list(df2.smi)
df1 = pd.read_csv(dataDir+"./present/glide.passed.csv", header=0)

#with open(f"{dataDir}./present/ligands.txt","r") as f:
for i, ent in tqdm(df1.iterrows()):
    if i < skipLines:
        #f.readline()
        continue
    #for i,smi in tqdm(enumerate(f.readlines())):
    """ if ent.smi in passedSmis:
        continue """
    #if list(df_score.d)[i] < 0.500:
    #    continue
    with open(dataDir+'./workspace/ligand.smi','w') as f1:
        f1.write(ent.smi)
    subprocess.run(f"cp {dataDir}./workspace/ligand.smi {GLIDE_PIPELINE}/input/input_{str(index)}.smi",shell=True)
    #print(f"OK\t{i}\t{smi}")
    #break
    #subprocess.run(f"mkdir $GLIDE_PIPELINE/{index}",shell=True)
    if not os.path.exists(f"{GLIDE_PIPELINE}/{index}"):
        os.mkdir(f"{GLIDE_PIPELINE}/{index}")
    subprocess.run(f"bash {GLIDE_PIPELINE}/src/pipeline_mult_{proteinName}{CONSTRAINT}.sh input_{str(index)}.smi {str(index)} 4",shell=True)
    #subprocess.run(f"cp {GLIDE_PIPELINE}/{index}/glide-{proteinName}dw_{str(index)}.csv {dataDir}./workspace/glide-{proteinName}dw_{str(index)}.csv",shell=True)
    subprocess.run(f"mv {GLIDE_PIPELINE}/{index}/glide-{proteinName}dw{CONSTRAINT}_{str(index)}.csv {dataDir}./workspace/",shell=True)
    df = pd.read_csv(f"{dataDir}./workspace/glide-{proteinName}dw{CONSTRAINT}_{str(index)}.csv")
    m = float(df["r_i_docking_score"][0])
    if m < -4:
        #mol = Chem.MolFromSmiles(ent.smi)

        #qed = QED.qed(mol)
        #with open(dataDir+"./present/glide.passed_2.csv","a") as f2:
        #    f2.write(ent.smi.strip()+","+str(m)+","+str(ent.d)+","+str(qed)+"\n")
            #passedSmis.append(ent.smi)
        #df_append = pd.DataFrame(data=[[ent.smi,m,ent.d,qed]],columns=["smi","g","d","q"])
        #df2 = pd.concat([df2,df_append],ignore_index=True)
        subprocess.run(f"bash {GLIDE_PIPELINE}/src/ligand_pose.sh input_{str(index)}.smi {str(index)}",shell=True)
        #subprocess.run(f"sleep 30",shell=True)
        if not os.path.exists(f"{dataDir}./present/poses/dock_{i}"):
            os.mkdir(f"{dataDir}./present/poses/dock_{i}")
        subprocess.run(f"mv {GLIDE_PIPELINE}/{index}/out*.png {dataDir}./present/poses/dock_{i}",shell=True)
        #break
    #elif m > -4:
    #    subprocess.run("sleep 5")