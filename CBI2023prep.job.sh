#!/bin/zsh
#$ -cwd
#$ -t 1-3:1
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
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
export TASK=`expr $SGE_TASK_ID - 1`
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
# |CBI| 1-5 | 3ZOS | Sigmoid, QED, Tox | next |
if [ $SGE_TASK_ID  -eq 1 ]; then
    for i in `seq 1 5`
    do
        export logdir="/log_cbi/data$i/"
        python Generator/mcts.py mcts.data_dir=$logdir mcts.isLoadTree=False mcts.time_limit_sec=$((23*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED', 'Toxicity']" mcts.in_smiles_file=$logdir"input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=3.5 reward.protein_name="3zosA_prepared" reward.center="[-7.5,2.5,-40]" reward.box="[24,20,20]" > .$logdir./out.log &
        PID=$!
    done
    wait $PID
# |CBI| 6-10 | 3ZOS | Sigmoid, QED, ConstTox | next | TOX Thre: 0.5 |
elif [ $SGE_TASK_ID  -eq 2 ]; then
    for i in `seq 6 10`
    do
        export logdir="/log_cbi/data$i/"
        python Generator/mcts.py mcts.data_dir=$logdir mcts.isLoadTree=False mcts.time_limit_sec=$((23*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED', 'ConstToxicity']" mcts.in_smiles_file=$logdir"input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=3.5 reward.protein_name="3zosA_prepared" reward.center="[-7.5,2.5,-40]" reward.box="[24,20,20]" reward.toxicity_threshold=1.0 > .$logdir./out.log &
        PID=$!
    done
    wait $PID
# |CBI| 11-15 | 3ZOS | Sigmoid, QED, ConstTox | next | TOX Thre: 0.25 |
elif [ $SGE_TASK_ID  -eq 3 ]; then
    for i in `seq 11 15`
    do
        export logdir="/log_cbi/data$i/"
        python Generator/mcts.py mcts.data_dir=$logdir mcts.isLoadTree=False mcts.time_limit_sec=$((23*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED', 'ConstToxicity']" mcts.in_smiles_file=$logdir"input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=3.5 reward.protein_name="3zosA_prepared" reward.center="[-7.5,2.5,-40]" reward.box="[24,20,20]" reward.toxicity_threshold=0.25 > .$logdir./out.log &
        PID=$!
    done
    wait $PID
else
    echo error
fi
wait $PID
#echo $SGE_TASK_ID
conda deactivate
#deactivate
if [ $SGE_TASK_ID -eq 101 ]; then
curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH_2/with/key/bD0xPz0Ajd2SWn6JVoww3w/?value1=lig6\&value2=express
fi
echo finish
date
#mv ligand_design.0.job.sh.* log$1d_normal
