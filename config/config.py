import math

from hydra.core.config_store import ConfigStore

from dataclasses import dataclass, field
from typing import List


@dataclass
class PreProcess:
    datapath: str = "/Data/input/sample_data.smi"
    outdir: str = "/Data/preprocessed/"
    ratio: float = 0.8
    max_len: int = 20


@dataclass
class ModelConfig:
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dr_rate: float = 0.2
    fr_len: int = 25


@dataclass
class TrainConfig:
    lr: float = 0.0001
    epoch: int = 100
    datapath: str = "/Data/preprocessed/fragments.smi"
    ratio: float = 0.2
    batch_size: int = 128
    seq_len: int = 25
    shuffle: bool = True
    drop_last: bool = True
    start_epoch: int = 0
    eval_step: int = 1
    save_step: int = 10
    model_dir: str = "/ckpt/"
    log_dir: str = "/logs/"
    ckptdir: str = "/ckpt/"
    log_filename: str = "log.txt"


@dataclass
class MCTSConfig:
    n_step: int = 1000
    n_iter: int = 3
    seq_len: int = 25
    in_smiles_file: str = "/Data/input/init_smiles.smi" #TODO: merge to data_dir
    rep_file: str = ""
    modeL_dir: str = "/ckpt/"
    out_dir: str = "/Data/output/" # TODO: merge to data_dir
    ucb_c: float = 1/math.sqrt(2)
    model_ver: int = 100
    model_dir: str = "/ckpt/"
    reward_name: str = "PLogP"
    data_dir: str = "/data_templete/"
    isLoadTree: bool = False
    time_limit_sec: int = 10*60

@dataclass
class RewardConfig:
    protein_name: str = "6lu7"+"_prepared"
    reward_list: List[str] = field(default_factory=lambda: ['Docking', 'QED', 'Toxicity'])# choose from QED, PLogP, Docking, Toxicity
    etoxpred_model: str = "/Utils/etoxpred_best_model.joblib"
    


@dataclass
class Config:
    prep: PreProcess = PreProcess()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    mcts: MCTSConfig = MCTSConfig()
    reward: RewardConfig = RewardConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

