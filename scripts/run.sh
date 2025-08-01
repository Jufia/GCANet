# run: bash ./scripts/run.sh
device=cuda:7

python -u main.py \
    --log_name 512_mcc5_patchtst.log \
    --use_data mcc5 \
    --algorithm PatchTST \
    --epochs 20 \
    --GPU $device \
    --optimizer radam

# python -u main.py \
#     --log_name 512_mcc5_softshape.log \
#     --use_data mcc5 \
#     --algorithm softshape \
#     --epochs 50 \
#     --batch_size 16 \
#     --lr 1e-3 \
#     --GPU $device \
#     --optimizer adam

# python -u main.py \
#     --log_name 512_mcc5_TimesNet.log \
#     --use_data mcc5 \
#     --algorithm TimesNet \
#     --batch_size 16 \
#     --lr 0.001 \
#     --epochs 100 \
#     --GPU $device \
    # --optimizer radam

python -u main.py \
    --log_name 512_mcc5_ModernTCN.log \
    --use_data mcc5 \
    --algorithm ModernTCN \
    --lr 1e-4 \
    --batch_size 128 \
    --epochs 100 \
    --GPU $device \
    --optimizer adam \
    --lambda_l2 0

python -u main.py \
    --log_name 512_mcc5_TimeMixer.log \
    --use_data mcc5 \
    --algorithm TimeMixer \
    --batch_size 16 \
    --lr 0.001 \
    --epochs 100 \
    --GPU $device \
    --optimizer radam

python -u main.py \
    --log_name 512_mcc5_LightTS.log \
    --use_data mcc5 \
    --algorithm LightTS \
    --batch_size 16 \
    --lr 0.001 \
    --epochs 100 \
    --GPU $device \
    --optimizer radam

python -u main.py \
    --log_name 512_mcc5_InceptionTime.log \
    --use_data mcc5 \
    --algorithm InceptionTime \
    --lr 0.001 \
    --epochs 100 \
    --GPU $device \

python -u main.py \
    --log_name 512_mcc5_TCN.log \
    --use_data mcc5 \
    --algorithm TCN \
    --lr 0.001 \
    --epochs 100 \
    --GPU $device \