#!/bin/zsh
#$ -cwd
#$ -t 1-30:30
#$ -l f_node=1
#$ -l h_rt=24:00:00
source /etc/profile.d/modules.sh
module load cuda/11.2.146 cudnn/8.1
date
#bash init.sh
#source .venv/bin/activate
source ~/.zshrc
conda activate mermaid
export HYDRA_FULL_ERROR=1
export TASK=`expr $SGE_TASK_ID + 0`
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
for i in `seq 1 20`
do
    python Generator/mcts.py mcts.data_dir="/data_8gcy/data$TASK/" mcts.isLoadTree=True mcts.time_limit_sec=$((23*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'Const']"  mcts.n_iter=1 reward.protein_name="8gcy_prepared" reward.center="[6.0,3.5,24.7]" reward.box="[62,60,54]" reward.spacing=1.000 mcts.in_smiles_file="/data_8gcy/data$TASK/input/next.smi" mcts.sascore_threshold=3.5 mcts.tanimoto_threshold=0.5 > ./data_8gcy/data$TASK/out.log &
    PID=$!
    export TASK=`expr $TASK + 1`
done
wait $PID
#echo $SGE_TASK_ID
conda deactivate
#deactivate
if [ $SGE_TASK_ID -eq 6 ]; then
curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH_2/with/key/bD0xPz0Ajd2SWn6JVoww3w/?value1=8gcy
fi
echo finish
date
#mv ligand_design.0.job.sh.* log$1d_normal
