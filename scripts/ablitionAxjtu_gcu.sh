data=mcc5
device=cuda:1

for length in 1024
do
    for snr_val in -1 -3 -6
    do
        python -u main.py \
            --log_name ablitionA_${length}_${data}_${snr_val}_gcu_duplicated.log \
            --use_data ${data} \
            --algorithm GCA \
            --length ${length} \
            --snr ${snr_val} \
            --epochs 100 \
            --GPU ${device} \
            --optimizer adam
    done
done