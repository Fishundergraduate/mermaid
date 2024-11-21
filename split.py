from tqdm import tqdm
import subprocess
with (open("Data/input/sample_data.smi","r") as f):
    for i,smi in tqdm(enumerate(f.readlines())):
        with open(f"molOpt/data{100+i+1}/input/sample_given{101+i}.smi", "w") as fo:
            fo.write(smi)        
        subprocess.run(f"rm -r molOpt/data{101+i}/data_templete/", shell=True)
        #break