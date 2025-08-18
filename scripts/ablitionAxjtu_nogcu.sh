data=mcc5
device=cuda:0

for length in 1024
do
    for snr_val in -1 -3 -6
    do
        python -u main.py \
            --log_name ablitionA_${length}_${data}_${snr_val}_nogcu_duplicated.log \
            --use_data ${data} \
            --algorithm GCA \
            --length ${length} \
            --snr ${snr_val} \
            --epochs 100 \
            --gcug none \
            --GPU ${device} \
            --optimizer adam
    done
done