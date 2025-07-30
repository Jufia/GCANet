# according to the paper:
# length = 512
# epoch = 20
# optimizer = 'radam'

python main.py --algorithm PatchTST --GPU cuda:7 --log_name PatchTST --use_data hit --epochs 20
python main.py --algorithm PatchTST --GPU cuda:7 --log_name PatchTST --use_data xjtu --epochs 20
python main.py --algorithm PatchTST --GPU cuda:7 --log_name PatchTST --use_data mcc5 --epochs 20
