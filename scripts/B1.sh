algorithm=ConvTran
length=1024
gcu=cuda:1
for data in mcc5 xjtu
do
    python main.py --algorithm $algorithm --length $length --GPU $gcu --log_name ${length}_${data}_${algorithm}.log --use_data $data
done
