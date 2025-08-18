algorithm=GCA
gcu=cuda:1
for data in mcc5
do
    for length in 512
    do
        python main.py --algorithm $algorithm --length $length --GPU $gcu --log_name ${length}_${data}_${algorithm}.log  --use_data $data --optimizer adam --batch_size 128
    done
done