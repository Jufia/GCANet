data=mcc5
device=cuda:0

for length in 512 1024
do
    python -u main.py \
        --log_name ablitionA_${length}_${data}_raw_nogcu.log \
        --use_data ${data} \
        --algorithm GCA \
        --length ${length} \
        --epochs 100 \
        --gcug none \
        --GPU ${device} \
        --optimizer adam

    for snr_val in 1 -1 -3 -6
    do
        python -u main.py \
            --log_name ablitionA_${length}_${data}_${snr_val}_nogcu.log \
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