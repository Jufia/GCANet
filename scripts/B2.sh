algorithm=ConvTran
gcu=cuda:1
for data in xjtu mcc5
do
    for length in 1024 512
    do
        python main.py --algorithm $algorithm --length $length --GPU $gcu --log_name ${length}_${data}_${algorithm}.log  --use_data $data
    done
done