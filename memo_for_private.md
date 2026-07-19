# how to open mermaid virtual environment
(bash)> zsh
(zsh)> source /etc/profile.d/modules.sh
(zsh)> module load cuda/11.2.146 cudnn/8.1
(zsh)(base)> conda activate mermaid
(zsh)(mermaid)> hoge
||
(zsh)(mermaid)> conda deactivate

# TO PARALLEL RUN
python Generator/mcts.py mcts.data_dir="/data_2"

datas
|submittor| data_dir|protein|dock reward|
|---|---|---|---|
|0 |1 to 5|6lu7|normal|
|1| 5 to 10|6lu7 | sigmoid|
|2 | 11 to 15|6lu7 | nonormal|
|3 | 16 to 20| 3zos| normal|
|4| 21 to 25| 3zos| sigmoid|
|5| 26 to 30| 3zos| nonormal|

bachlor
|submittor| data_dir|protein|dock reward|
|---|---|---|---|
|1|molOpt1|6lu7|normal|
|2|molOpt2|6lu7 | sigmoid|
|3|molOpt3|6lu7 | nonormal|
|4|molOpt4| 3zos| normal|
|5|molOpt5| 3zos| sigmoid|
|6|molOpt6| 3zos| nonormal|

# TO check nonormal
23th Jan to 26th Jan
|submittor| data_dir|protein|dock reward|smiles|
|---|---|---|---|
|1|1 to 5|6lu7|nonormal / 10|init_smiles.smi|
|2|6 to 10|3zos|nonormal/ 10|init_smiles.smi|
|5|21,22,23,24|6lu7|normal, sigmoid, nonormal, nonormal/10: profile|init_smiles.smi|
|6|25,26,27,28|3zos|normal, sigmoid, nonormal, nonormal/10: profile|init_smiles.smi|

# TO 2 search

8th May to 10th May
|submittor| data_dir|protein|dock reward|smiles|opt|
|---|---|---|---|
|2|1 to 100|5zyk|sigmoid|next.smi|dq|
|6|101 to 105|6lu7|sigmoid|next.smi|dqt:sa=10|
|6|106 to 110|3zos|sigmoid|next.smi|dqt:sa=10|
|6|111 to 115|6lu7|sigmoid|next.smi|dqt:sa=3.5|
|6|116 to 120|3zos|sigmoid|next.smi|dqt:sa=3.5|
|6|121 to 125|6lu7|sigmoid|next.smi|dq:t->後で足切り|
|6|126 to 130|3zos|sigmoid|next.smi|dq:t->後で足切り|

mcts.data_dir="/data$SGE_TASK_ID/" mcts.isLoadTree=False mcts.time_limit_sec=$((23*60*60+30*60)) reward.reward_list="['SigmoidDocking', 'QED']" reward.protein_name="5zyk_prepared" reward.center="['-25.406', '9.601', '-2.276']" reward.box="['98','80','126']" reward.spacing=0.514 mcts.in_smiles_file="/data$SGE_TASK_ID/input/next.smi" mcts.n_iter=3 mcts.sascore_threshold=4
python -m cProfile -o prof$SGE_TASK_ID.prof -s tottime Generator/mcts.py mcts.data_dir="/data$SGE_TASK_ID/" mcts.isLoadTree=False mcts.time_limit_sec=$((23*60*60+30*60)) reward.reward_list="['SquareDocking', 'QED', 'Toxicity']"  mcts.n_iter=1 mcts.sascore_threshold=3.5 reward.protein_name="3zosA_prepared" reward.center="[-7.5,2.5,-40]" reward.box="[24,20,20]"

# single search

|submittor| data_dir|protein|dock reward|smiles|opt|
|---|---|---|---|
|1|1 to 30|8gcy|next.smi|dq|tanimoto>0.5|
|5zyk|1 to 420|~~|d|tanimoto>0.7|
|7|5zdp| ~~ | 

# From Jun 22th to 25th June

|submittor| data_dir|protein|dock reward|smiles|opt|
|---|---|---|---|
|gen|mso|||
|7|->TAO|||
|5zyk|->novare|||
|6|->sigbio data|||
|8|->sigbio plogp-tpsa|||

# For CBI
|submittor| data_dir|protein|dock reward|smiles|opt|
|---|---|---|---|
|CBI| 1-5 | 3ZOS | Sigmoid, QED, Tox | next |
|CBI| 6-10 | 3ZOS | Sigmoid, QED, ConstTox | next | TOX Thre: 0.5 |
|CBI| 11-15 | 3ZOS | Sigmoid, QED, ConstTox | next | TOX Thre: 0.25 |