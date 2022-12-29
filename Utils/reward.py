import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import warnings
warnings.filterwarnings('ignore')

import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.six.moves import cPickle
from rdkit.Chem import AllChem, QED, DataStructs, Descriptors

from Utils.sascore import calculateScore

import time
from joblib import load
import hydra
from omegaconf import DictConfig, OmegaConf
from config.config import Config
#TO Docking
import traceback
from rdkit import rdBase
import pandas as pd
import subprocess
import numpy as np

import pdb

def getReward(name):
    if name == "QED":
        return QEDReward()
    elif name == "PLogP":
        return PenalizedLogPReward()
    elif name == "Docking":
        return DockingReward()
    elif name == "SigmoidDocking":
        return SigmoidDockingReward()
    elif name == "NonNormalizeDocking":
        return NonNormalizedDockingReward()
    elif name == "Toxicity":
        return ToxicityReward()
    else:
        raise NotImplementedError()

def getRewards(nameList: list):
    return [getReward(name) for name in nameList]

class Reward:
    def __init__(self):
        self.vmin = -100
        self.max_r = -10000
        return

    def reward(self, smiles):
        raise NotImplementedError()


class PenalizedLogPReward(Reward):
    def __init__(self, *args, **kwargs):
        super(PenalizedLogPReward, self).__init__(*args, **kwargs)
        self.vmin = -100
        return

    def reward(self, smiles):
        """
            This code is obtained from https://drive.google.com/drive/folders/1FmYWcT8jDrwZlzPbmMpRhulb9OKTDWJL
            , which is a part of GraphAF program done by Chence Shi.
            Reward that consists of log p penalized by SA and # long cycles,
            as described in (Kusner et al. 2017). Scores are normalized based on the
            statistics of 250k_rndm_zinc_drugs_clean.smi dataset
            :param mol: rdkit mol object
            :return: float
            """
        # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        SA_mean = -3.0525811293166134
        SA_std = 0.8335207024513095
        cycle_mean = -0.0485696876403053
        cycle_std = 0.2860212110245455

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                log_p = Descriptors.MolLogP(mol)
                SA = -calculateScore(mol)

                # cycle score
                cycle_list = nx.cycle_basis(nx.Graph(
                    Chem.rdmolops.GetAdjacencyMatrix(mol)))
                if len(cycle_list) == 0:
                    cycle_length = 0
                else:
                    cycle_length = max([len(j) for j in cycle_list])
                if cycle_length <= 6:
                    cycle_length = 0
                else:
                    cycle_length = cycle_length - 6
                cycle_score = -cycle_length

                normalized_log_p = (log_p - logP_mean) / logP_std
                normalized_SA = (SA - SA_mean) / SA_std
                normalized_cycle = (cycle_score - cycle_mean) / cycle_std
                score = normalized_log_p + normalized_SA + normalized_cycle
            except ValueError:
                score = self.vmin
        else:
            score = self.vmin

        return score


class QEDReward(Reward):
    def __init__(self, *args, **kwargs):
        super(QEDReward, self).__init__(*args, **kwargs)
        self.vmin = 0

    def reward(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        try:
            if mol is not None:
                score = QED.qed(mol)
            else:
                score = -1
        except:
            score = -1

        return score

class DockingReward(Reward):
    def __init__(self, *args, **kwargs):
        super(DockingReward, self).__init__(*args, **kwargs)
        self.vmin = -20
        self.dataDir = hydra.utils.get_original_cwd()+OmegaConf.structured(Config)["mcts"]["data_dir"]
        self.proteinName = OmegaConf.structured(Config)["reward"]["protein_name"]
        self.proteinFile = hydra.utils.get_original_cwd()+OmegaConf.structured(Config)["reward"]["protein_dir"]

    def _normalize(self, score: float) -> float:
        base_dock_score = 0
        return -round(((score - base_dock_score)*0.1)/(1+abs((score - base_dock_score)*0.1)),3)
    def reward(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            #pdb.set_trace()
            print("Docking: Vmin")
            score = self.vmin
            return score
        del mol
        # create SMILES file
        with open(self.dataDir+'./workspace/ligand.smi','w') as f:
            f.write(smiles)
        # save produced ligands
        with open(self.dataDir+'./output/allLigands.txt','a', newline="\n") as f:
            f.write(smiles+"\n") 

         # convert SMILES > PDBQT
        # --gen3d: the option for generating 3D coordinate
        #  -h: protonation
        
        try:
            cvt_log = open(self.dataDir+"workspace/cvt_log.txt","w")
            cvt_cmd = ["obabel", self.dataDir+"workspace/ligand.smi" ,"-O",self.dataDir+"workspace/ligand.pdbqt" ,"--gen3D","-p"]
            subprocess.run(cvt_cmd, stdin=None, input=None, stdout=cvt_log, stderr=None, shell=False, timeout=300, check=False, universal_newlines=False)
            cvt_log.close()
        except:
            f = open(self.dataDir+"present/error_output.txt", 'a')
            print("cvt_error: ", time.asctime( time.localtime(time.time()) ),file=f)
            print(traceback.print_exc(),file=f)
            f.close()
        # docking simulation
        try:
            vina_log = open(self.dataDir+"workspace/log_docking.txt","w")
            docking_cmd =["vina --config "+self.proteinFile+self.proteinName+"_vina_config.txt --num_modes=1 --receptor="+self.proteinFile+self.proteinName+".pdbqt --ligand="+self.dataDir+"./workspace/ligand.pdbqt"]#TODO: direct acess to protein file
            subprocess.run(docking_cmd, stdin=None, input=None, stdout=vina_log, stderr=None, shell=True, timeout=600, check=False, universal_newlines=False)
            vina_log.close()
            data = pd.read_csv(self.dataDir+'workspace/log_docking.txt', sep= "\t",header=None)
            score = round(float(data.values[-2][0].split()[1]),2)
        except ValueError as ve:
            print("Value Error"+smiles)
            return self.vmin
        except:
            f = open(self.dataDir+"./present/error_output.txt", 'a')
            print("vina_error: ", time.asctime( time.localtime(time.time()) ),file=f)
            print(traceback.print_exc(),file=f)
            f.close()
            score = self.vmin
            return score
        assert score < 10**10

        base_dock_score = 0
        return self._normalize(score)

class SigmoidDockingReward(DockingReward):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _normalize(self, score:float)-> float:
        threshold = -5
        return 1 / (1+ np.exp(score - threshold))

class NonNormalizedDockingReward(DockingReward):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _normalize(self, score: float)-> float:
        return -score

class ToxicityReward(Reward):
    def __init__(self, *args, **kwargs):
        super(ToxicityReward, self).__init__(*args, **kwargs)
        self.vmin = 0
        self.model = load(hydra.utils.get_original_cwd()+ OmegaConf.structured(Config)["reward"]["etoxpred_model"])

    def reward(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1
        
        mol = Chem.AddHs(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_string = fp.ToBitString()
        tmpX = np.array(list(fp_string),dtype=float)
        tox_score = self.model.predict_proba(tmpX.reshape((1,1024)))[:,1]
        return 1 - tox_score[0]