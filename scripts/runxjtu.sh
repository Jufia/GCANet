# run: bash ./scripts/runxjtu.sh
# device=cuda:7
length=1024
data=xjtu
snr=-6
name=6_

python -u main.py \
    --log_name ${length}_${data}_${name}GCA.log \
    --use_data ${data} \
    --algorithm GCA \
    --length ${length} \
    --snr ${snr} \
    --epochs 100 \
    --optimizer adam

python -u main.py \
    --log_name ${length}_${data}_${name}GCA_noatt.log \
    --use_data ${data} \
    --algorithm GCA \
    --length ${length} \
    --snr ${snr} \
    --epochs 100 \
    --optimizer adam \
    --att none

python -u main.py \
    --log_name ${length}_${data}_${name}patchtst.log \
    --use_data ${data} \
    --algorithm PatchTST \
    --length ${length} \
    --snr ${snr} \
    --epochs 20 \
    --optimizer radam

python -u main.py \
    --log_name ${length}_${data}_${name}softshape.log \
    --use_data ${data} \
    --algorithm softshape \
    --length ${length} \
    --snr ${snr} \
    --epochs 50 \
    --lr 1e-3 \
    --optimizer adam
    # --batch_size 16 \

python -u main.py \
    --log_name ${length}_${data}_${name}TimesNet.log \
    --use_data ${data} \
    --algorithm TimesNet \
    --lr 0.001 \
    --length ${length} \
    --snr ${snr} \
    --epochs 100 \
    --optimizer radam
    # --batch_size 16 \

python -u main.py \
    --log_name ${length}_${data}_${name}ModernTCN.log \
    --use_data ${data} \
    --algorithm ModernTCN \
    --lr 1e-4 \
    --epochs 100 \
    --length ${length} \
    --snr ${snr} \
    --optimizer adam \
    --lambda_l2 0
    # --batch_size 128 \

python -u main.py \
    --log_name ${length}_${data}_${name}TimeMixer.log \
    --use_data ${data} \
    --algorithm TimeMixer \
    --lr 0.001 \
    --length ${length} \
    --snr ${snr} \
    --epochs 100 \
    --optimizer radam
    # --batch_size 16 \

python -u main.py \
    --log_name ${length}_${data}_${name}LightTS.log \
    --use_data ${data} \
    --algorithm LightTS \
    --lr 0.001 \
    --length ${length} \
    --snr ${snr} \
    --epochs 100 \
    --optimizer radam
    # --batch_size 16 \

python -u main.py \
    --log_name ${length}_${data}_${name}InceptionTime.log \
    --use_data ${data} \
    --algorithm InceptionTime \
    --lr 0.001 \
    --length ${length} \
    --snr ${snr} \
    --epochs 100 \

python -u main.py \
    --log_name ${length}_${data}_${name}TCN.log \
    --use_data ${data} \
    --algorithm TCN \
    --lr 0.001 \
    --length ${length} \
    --snr ${snr} \
    --epochs 100 \