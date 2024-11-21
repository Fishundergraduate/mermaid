for i in `seq 1 34`
do
    #cat log_5zyk_2d_8/log_data/data$i[0-9]/present/more0.8.txt > log_5zyk_2d_8/concat$i/present/more0.8.txt
    cat log_5zyk_2d_8/concat*/present/aizy.pass.csv > log_5zyk_2d_8/aizy.pass.all.csv
done