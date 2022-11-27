2022.11.06
## Utils/reward.py
1. extend dockingReward -> DockingRewardClass
2. extend eToxPred-> ToxicityRewardClass
3. extend other Rewards(approx 2)
4. dataDir catch from hydra manager => OmegaConf getter

## jobsubmittor.sh
1. ArrayJob
2. output dir must be "day counted"

## Genrator/mcts.py
1. extend multi-objective optimization(line: 205) -> たぶんできた
2. change select(): Line78 
3. output score file and ligands file (LOGGAR) => OK
4. (optional) torch to tf
5. change to RAscore from OZAWA senpai.

## Model/model
1. (optional) torch to tf

## config/config.py
1. move to Data Dir
2. write on vina_config