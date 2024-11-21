for i in {0..43}
#for i in {0..40}
do
    #cat log_datas/data$i/present/aizy.pass.csv | grep 'c3ccccn3'
    cat 5zyk_hit/concat$i/present/aizy.pass.csv | grep 'S(=O)(=O)'
    echo $i
done
