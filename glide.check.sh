#!/bin/bash
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00
source /etc/profile.d/modules.sh
cat glide.check.sh
module load cuda/11.2.146 cudnn/8.1 python/3.10.2
date
#bash init.sh
#source .venv/bin/activate
#source ~/.zshrc
#conda activate mermaid
#export HYDRA_FULL_ERROR=1
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>./log_5zyk_2d_3/concat/gpu.log &
#python Generator/mcts.py mcts.data_dir="/data$SGE_TASK_ID/" mcts.isLoadTree=False mcts.time_limit_sec=$((23*60*60+30*60)) reward.reward_list="['Docking', 'QED', 'Toxicity']" reward.protein_name="5zyk_prepared" reward.center="['-25.406', '9.601', '-2.276']" reward.box="['98','80','126']" reward.spacing=0.514 mcts.in_smiles_file="/Data/input/CC.smi" mcts.n_iter=3 mcts.sascore_threshold=10
python3 glide.check.py "CC(=O)N(C)C(=O)NCc1ccccc1N1CCS(=O)(=O)CC1" #6349 #181 #11200 #12770
#echo $SGE_TASK_ID
#conda deactivate
#deactivate
#if [ $SGE_TASK_ID -eq 1 ]; then
#curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH/with/key/bD0xPz0Ajd2SWn6JVoww3w/
#fi
echo finish
date
#mv ligand_design.0.job.sh.* log$1d_normal