algorithm=biLSTM
length=512
gcu=cuda:1
for data in xjtu mcc5
do
    python main.py --algorithm $algorithm --length $length --GPU $gcu --log_name ${length}_${data}_${algorithm}.log --use_data $data
done
