#!/bin/zsh
#$ -cwd
#$ -t 1:1
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
vmstat -S M -t 10 >./data_test/cpu.log & 
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
#if [ $SGE_TASK_ID -eq 1 ];then
#python Generator/mcts.py mcts.data_dir="/data_test/" mcts.isLoadTree=False mcts.time_limit_sec=$((23*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED']" reward.protein_name="5zyk_prepared" reward.center="['-25.406', '9.601', '-2.276']" reward.box="['98','80','126']" reward.spacing=0.514 mcts.in_smiles_file="/data_test/input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=3.5 mcts.tanimoto_threshold=0.7 >> ./data_test/out.log 
#fi
#if [ $SGE_TASK_ID -eq 2 ];then 
python Generator/mcts.py mcts.data_dir="/data_test2/" mcts.isLoadTree=False reward.reward_list="['SigmoidDocking', 'QED']" mcts.time_limit_sec=$((7*60)) reward.protein_name="3zosA_prepared" reward.center="[-7.5,2.5,-40]" reward.box="[24,20,20]"
#fi

conda deactivate
#deactivate
curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH_2/with/key/bD0xPz0Ajd2SWn6JVoww3w/?value1=LIGANDTEST\&value2=jobfinish
echo finish
date