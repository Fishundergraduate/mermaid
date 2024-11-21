#!/bin/zsh
#$ -cwd
#$ -l q_node=1
#$ -t 1-10:1
#$ -l h_rt=00:10:00
source /etc/profile.d/modules.sh
module load cuda/11.2.146 cudnn/8.1

#bash init.sh
#source .venv/bin/activate
echo $SGE_TASK_ID 
source ~/.zshrc
#conda activate mermaid
export HYDRA_FULL_ERROR=1
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
#python Generator/mcts.py mcts.data_dir="/data5/" mcts.isLoadTree=False reward.reward_list="['SigmoidDocking', 'QED', 'Toxicity']"

#conda deactivate
#deactivate
#curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH/with/key/bD0xPz0Ajd2SWn6JVoww3w/
echo finish