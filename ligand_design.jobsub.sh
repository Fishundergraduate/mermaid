#!/bin/bash

NUM_JOB=$2  # ジョブの数
PROT=$1
job_id_prefix="j"$PROT
first_job_id=$job_id_prefix"001"
# 最初のジョブを発行
qsub -g tga-takuetsu -M $MAIL -N $first_job_id ./lig.$PROT.job.sh $PROT
prev_job_id=$first_job_id
echo "Submitted j8gcy001 with ID: $prev_job_id"

# 残りのジョブを発行
for ((i=2; i<=NUM_JOB; i++)); do
    job_name=$(printf $job_id_prefix"%03d" $i )
    hold_option="-hold_jid $prev_job_id"  # 直前のジョブに依存    
    echo $job_name $hold_optioon
    qsub -g tga-takuetsu -M $MAIL -N $job_name $hold_option ./lig.$PROT.job.sh $PROT
    prev_job_id=$job_name
    echo "Submitted $job_name with ID: $prev_job_id"
done
