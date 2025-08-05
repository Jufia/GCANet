
python main.py --algorithm GCA --use_data mcc5 --log_name ablationB_gcu_lr_1_rand_1_2.log --random_state 1 --lr 1 --blocker '' --gcug True --att agca --gcub True --head 2 --epochs 100 --batch_size 128

python main.py --algorithm GCA --use_data mcc5 --log_name ablationB_gcu_lr_1_rand_0_2.log --random_state 0 --lr 1 --blocker '' --gcug True --att agca --gcub True --head 2 --epochs 100 --batch_size 128

python main.py --algorithm GCA --use_data mcc5 --log_name ablationB_gcu_lr_1_rand_215_2.log --random_state 215 --lr 1 --blocker '' --gcug True --att agca --gcub True --head 2 --epochs 100 --batch_size 128

python main.py --algorithm GCA --use_data mcc5 --log_name ablationB_gcu_lr_1_rand_613_2.log --random_state 613 --lr 1 --blocker '' --gcug True --att agca --gcub True --head 2 --epochs 100 --batch_size 128