import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("dataDir")
parser.add_argument("begin",type=int)
parser.add_argument("end",type=int)
parser.add_argument("outDir")

args = parser.parse_args()

dataDir = args.dataDir
begin = args.begin
end = args.end
outDir = args.outDir

dfA = pd.read_csv(dataDir+str(begin)+"/present/aizy.pass.csv",header=None,names=["smi","d","q"])

for i in range(begin+1, end):
    dfB = pd.read_csv(dataDir+str(i)+"/present/aizy.pass.csv",header=None,names=["smi","d","q"])
    dfA = pd.concat([dfA,dfB])
dfA = dfA.drop_duplicates(subset=["smi"],keep="last")
dfA.to_csv(outDir+"/present/merged.csv",index=False,header=None)