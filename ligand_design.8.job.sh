#!/bin/zsh
#$ -cwd
#$ -t 1-1:1
#$ -l q_node=1
#$ -l h_rt=24:00:00
source /etc/profile.d/modules.sh
module load cuda/11.2.146 cudnn/8.1
date
#bash init.sh
#source .venv/bin/activate
source ~/.zshrc
conda activate mermaid
export HYDRA_FULL_ERROR=1
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
export TASK=`expr $SGE_TASK_ID + 0`
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
# |6|101 to 105|6lu7|sigmoid|next.smi|dqt:sa=10|
for i in `seq 1 5`
do
    python Generator/mcts.py mcts.data_dir="/log_tpsa_plogp/data$TASK/" mcts.isLoadTree=False mcts.time_limit_sec=$((23*60*60+30*60)) reward.reward_list="['PLogP', 'TPSA']" mcts.in_smiles_file="/log_tpsa_plogp/data$TASK/input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=3.5 > ./log_tpsa_plogp/data$TASK/out.log &
    PID=$!
    export TASK=`expr $TASK + 1`
done
wait $PID
#echo $SGE_TASK_ID
conda deactivate
#deactivate
if [ $SGE_TASK_ID -eq 1 ]; then
curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH_2/with/key/bD0xPz0Ajd2SWn6JVoww3w/?value1=zdp5
fi
echo finish
date
#mv ligand_design.0.job.sh.* log$1d_normal
