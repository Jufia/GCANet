# run: bash ./scripts/runhit.sh
# device=cuda:7
length=1024
data=hit

python -u main.py \
    --log_name ${length}_${data}_GCA.log \
    --use_data ${data} \
    --algorithm GCA \
    --length ${length} \
    --epochs 100 \
    --optimizer adam

python -u main.py \
    --log_name ${length}_${data}_GCA_noatt.log \
    --use_data ${data} \
    --algorithm GCA \
    --length ${length} \
    --epochs 100 \
    --optimizer adam \
    --att none

python -u main.py \
    --log_name ${length}_${data}_patchtst.log \
    --use_data ${data} \
    --algorithm PatchTST \
    --length ${length} \
    --epochs 20 \
    --optimizer radam

python -u main.py \
    --log_name ${length}_${data}_softshape.log \
    --use_data ${data} \
    --algorithm softshape \
    --length ${length} \
    --epochs 50 \
    --lr 1e-3 \
    --optimizer adam
    # --batch_size 16 \

python -u main.py \
    --log_name ${length}_${data}_TimesNet.log \
    --use_data ${data} \
    --algorithm TimesNet \
    --lr 0.001 \
    --length ${length} \
    --epochs 100 \
    --optimizer radam
    # --batch_size 16 \

python -u main.py \
    --log_name ${length}_${data}_ModernTCN.log \
    --use_data ${data} \
    --algorithm ModernTCN \
    --lr 1e-4 \
    --epochs 100 \
    --length ${length} \
    --optimizer adam \
    --lambda_l2 0
    # --batch_size 128 \

python -u main.py \
    --log_name ${length}_${data}_TimeMixer.log \
    --use_data ${data} \
    --algorithm TimeMixer \
    --lr 0.001 \
    --length ${length} \
    --epochs 100 \
    --optimizer radam
    # --batch_size 16 \

python -u main.py \
    --log_name ${length}_${data}_LightTS.log \
    --use_data ${data} \
    --algorithm LightTS \
    --lr 0.001 \
    --length ${length} \
    --epochs 100 \
    --optimizer radam
    # --batch_size 16 \

python -u main.py \
    --log_name ${length}_${data}_InceptionTime.log \
    --use_data ${data} \
    --algorithm InceptionTime \
    --lr 0.001 \
    --length ${length} \
    --epochs 100 \

python -u main.py \
    --log_name ${length}_${data}_TCN.log \
    --use_data ${data} \
    --algorithm TCN \
    --lr 0.001 \
    --length ${length} \
    --epochs 100 \