# 一切正常
python main.py --algorithm GCA --use_data xjtu --log_name xjtu_GCA.log --lr 0.001 --blocker T --gcug T --att agca --gcub True --head 2 --epochs 10 --GPU cuda:7 --batch_size 128
python main.py --algorithm GCA --use_data hit --log_name hit_GCA.log --lr 0.001 --blocker T --gcug T --att agca --gcub True --head 2 --epochs 10 --GPU cuda:7 --batch_size 128
python main.py --algorithm GCA --use_data mcc5 --log_name mcc5_GCA.log --lr 0.001 --blocker T --gcug T --att agca --gcub True --head 2 --epochs 10 --GPU cuda:7 --batch_size 128


# 消融实验B
python main.py --algorithm GCA --use_data mcc5 --log_name ablationB_gcu_lr_1_rand_3407_1.log --lr 1 --blocker T --gcug True --att agca --gcub True --head 2 --epochs 100 --batch_size 128 --random_state 3407

python main.py --algorithm GCA --use_data mcc5 --log_name ablationB_gcu_lr_1_rand_3407_2.log --lr 1 --blocker '' --gcug True --att agca --gcub True --head 2 --epochs 100 --batch_size 128 --random_state 3407

python main.py --algorithm GCA --use_data mcc5 --log_name ablationB_gcu_lr_1_rand_3407_3.log --lr 1 --blocker T --gcug True --att none --gcub True --head 2 --epochs 100 --batch_size 128 --random_state 3407

