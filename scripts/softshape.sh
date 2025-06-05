# according to paper, the batch size is min(x_train[0], 16),,, 16太慢了...
# learning rate is 0.001 no decay
# max train epochs is 500

python main.py --algorithm softshape --batch_size 256 --lr 1e-3 --lr_decay 1 --GPU cuda:7 --log_name softshape --use_data hit --epochs 500
python main.py --algorithm softshape --batch_size 256 --lr 1e-3 --lr_decay 1 --GPU cuda:7 --log_name softshape --use_data xjtu --epochs 500
python main.py --algorithm softshape --batch_size 256 --lr 1e-3 --lr_decay 1 --GPU cuda:0 --log_name softshape --use_data mcc5 --epochs 500