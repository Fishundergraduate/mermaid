#!/bin/zsh
#$ -cwd
#$ -t 2-4:1
#$ -l q_node=1
#$ -l h_rt=3:00:00
source /etc/profile.d/modules.sh
module load cuda/11.2.146 cudnn/8.1
date
#bash init.sh
#source .venv/bin/activate
source ~/.zshrc
conda activate aizynth
aizynthcli --config ../aizynth/config.yml --smiles log_5zdp_2d_3/5zdp.0804.$SGE_TASK_ID.smi --stocks namiki --output output.$SGE_TASK_ID.json.gz
curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH_2/with/key/bD0xPz0Ajd2SWn6JVoww3w/?value1=aizynth\&value2=_5zdp\&value3=_$SGE_TASK_ID
