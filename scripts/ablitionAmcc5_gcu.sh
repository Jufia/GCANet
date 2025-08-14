data=mcc5
device=cuda:1

for length in 512 1024
do
    python -u main.py \
        --log_name ablitionA_${length}_${data}_raw_gcu.log \
        --use_data ${data} \
        --algorithm GCA \
        --length ${length} \
        --epochs 100 \
        --GPU ${device} \
        --optimizer adam

    for snr_val in 1 -1 -3 -6
    do
        python -u main.py \
            --log_name ablitionA_${length}_${data}_${snr_val}_gcu.log \
            --use_data ${data} \
            --algorithm GCA \
            --length ${length} \
            --snr ${snr_val} \
            --epochs 100 \
            --GPU ${device} \
            --optimizer adam
    done
done