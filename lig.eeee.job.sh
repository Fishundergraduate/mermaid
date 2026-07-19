#!/bin/bash
#$ -cwd
#$ -t 1-5:1
#$ -l node_q=1
#$ -l h_rt=0:10:00
source ~/.bashrc
echo hoge
sleep 10
echo end
