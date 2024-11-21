#!/bin/zsh
#$ -cwd
#$ -t 1-420:10
#$ -l h_node=1
#$ -l h_rt=24:00:00
source /etc/profile.d/modules.sh
module load cuda/11.2.146 cudnn/8.1
date
#bash init.sh
#source .venv/bin/activate
source ~/.zshrc
conda activate mermaid
export HYDRA_FULL_ERROR=1
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>./data$SGE_TASK_ID/gpu.log &
vmstat -S M -t 10 >./data$SGE_TASK_ID/cpu.log & 

export TASK=`expr $SGE_TASK_ID + 0`
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
for i in `seq 1 25`
do
    python Generator/mcts.py mcts.data_dir="/data$TASK/" mcts.isLoadTree=False mcts.time_limit_sec=$((23*60*60+30*60)) reward.reward_list="['SigmoidDocking','Const']" reward.protein_name="5zyk_prepared" reward.center="['-25.406', '9.601', '-2.276']" reward.box="['98','80','126']" reward.spacing=0.514 mcts.in_smiles_file="/data$TASK/input/next.smi" mcts.n_iter=1 mcts.sascore_threshold=3.5 mcts.tanimoto_threshold=0.7> ./data$TASK/out.log &
    PID=$!
    export TASK=`expr $TASK + 1`
done
wait $PID

conda deactivate
#deactivate
if [ $SGE_TASK_ID -eq 131 ]; then
curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH/with/key/bD0xPz0Ajd2SWn6JVoww3w/
fi
echo finish
date
#mv ligand_design.0.job.sh.* log$1d_normal