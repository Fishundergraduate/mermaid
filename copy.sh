for i in `seq $1 $2`
do
    #cp -pr data_templete/ data_8gcy/data$i
    cp -pr data_templete/ log_cbi/data$i
done