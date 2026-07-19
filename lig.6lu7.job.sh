#!/bin/bash
#$ -cwd
#$ -t 1-5:1
#$ -l node_q=1
#$ -l h_rt=24:00:00
source ~/.bashrc
module load miniconda cuda/12.1

eval  "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate mermaid-cuda121
export PROT="6lu7"
python Generator/mcts.py mcts.data_dir="/data_$PROT/data$SGE_TASK_ID/" mcts.isLoadTree=True mcts.time_limit_sec=$((23*60*60+30*60)) mcts.n_iter=1 reward.reward_list="['Docking', 'QED', 'Toxicity']" reward.scalor=10 mcts.in_smiles_file="/data_$PROT/data$SGE_TASK_ID/input/next.smi" reward.protein_name=$1"_prepared" reward.center="[-5.086, 14.329, 69.900]" reward.box="[16,20,20]" reward.spacing=1.000
conda deactivate
