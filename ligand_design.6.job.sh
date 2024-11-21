#!/bin/zsh
#$ -cwd
#$ -t 1-30:30
#$ -l f_node=1
#$ -l h_rt=12:00:00
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
# |6|101 to 105|6lu7|sigmoid|next.smi|dqt:sa=10|
for i in `seq 1 5`
do
    python Generator/mcts.py mcts.data_dir="/log_sigbio_jun_3/data$TASK/" mcts.isLoadTree=False mcts.time_limit_sec=$((11*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED', 'Toxicity']" mcts.in_smiles_file="/log_sigbio_jun_3/data$TASK/input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=10 > ./log_sigbio_jun_3/data$TASK/out.log &
    PID=$!
    export TASK=`expr $TASK + 1`
done
# |6|106 to 110|3zos|sigmoid|next.smi|dqt:sa=10|
for i in `seq 1 5`
do
    python Generator/mcts.py mcts.data_dir="/log_sigbio_jun_3/data$TASK/" mcts.isLoadTree=False mcts.time_limit_sec=$((11*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED', 'Toxicity']" mcts.in_smiles_file="/log_sigbio_jun_3/data$TASK/input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=10 reward.protein_name="3zosA_prepared" reward.center="[-7.5,2.5,-40]" reward.box="[24,20,20]" > ./log_sigbio_jun_3/data$TASK/out.log &
    PID=$!
    export TASK=`expr $TASK + 1`
done
# |6|111 to 115|6lu7|sigmoid|next.smi|dqt:sa=3.5|
for i in `seq 1 5`
do
    python Generator/mcts.py mcts.data_dir="/log_sigbio_jun_3/data$TASK/" mcts.isLoadTree=False mcts.time_limit_sec=$((11*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED', 'Toxicity']" mcts.in_smiles_file="/log_sigbio_jun_3/data$TASK/input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=3.5 > ./log_sigbio_jun_3/data$TASK/out.log &
    PID=$!
    export TASK=`expr $TASK + 1`
done
# |6|116 to 120|3zos|sigmoid|next.smi|dqt:sa=3.5|
for i in `seq 1 5`
do
    python Generator/mcts.py mcts.data_dir="/log_sigbio_jun_3/data$TASK/" mcts.isLoadTree=False mcts.time_limit_sec=$((11*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED', 'Toxicity']" mcts.in_smiles_file="/log_sigbio_jun_3/data$TASK/input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=3.5 reward.protein_name="3zosA_prepared" reward.center="[-7.5,2.5,-40]" reward.box="[24,20,20]" > ./log_sigbio_jun_3/data$TASK/out.log &
    PID=$!
    export TASK=`expr $TASK + 1`
done
# |6|121 to 125|6lu7|sigmoid|next.smi|dq:t->後で足切り|
for i in `seq 1 5`
do
    python Generator/mcts.py mcts.data_dir="/log_sigbio_jun_3/data$TASK/" mcts.isLoadTree=False mcts.time_limit_sec=$((11*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED']" mcts.in_smiles_file="/log_sigbio_jun_3/data$TASK/input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=3.5 > ./log_sigbio_jun_3/data$TASK/out.log &
    PID=$!
    export TASK=`expr $TASK + 1`
done
# |6|126 to 130|3zos|sigmoid|next.smi|dq:t->後で足切り|
for i in `seq 1 5`
do
    python Generator/mcts.py mcts.data_dir="/log_sigbio_jun_3/data$TASK/" mcts.isLoadTree=False mcts.time_limit_sec=$((11*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED']" mcts.in_smiles_file="/log_sigbio_jun_3/data$TASK/input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=3.5 reward.protein_name="3zosA_prepared" reward.center="[-7.5,2.5,-40]" reward.box="[24,20,20]" > ./log_sigbio_jun_3/data$TASK/out.log &
    PID=$!
    export TASK=`expr $TASK + 1`
done
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