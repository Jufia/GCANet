# 一切正常
python main.py --algorithm GCA --use_data xjtu --log_name xjtu_GCA.log --lr 0.001 --blocker T --gcug T --att agca --gcub True --head 2 --epochs 10 --GPU cuda:7 --batch_size 128
python main.py --algorithm GCA --use_data hit --log_name hit_GCA.log --lr 0.001 --blocker T --gcug T --att agca --gcub True --head 2 --epochs 10 --GPU cuda:7 --batch_size 128
python main.py --algorithm GCA --use_data mcc5 --log_name mcc5_GCA.log --lr 0.001 --blocker T --gcug T --att agca --gcub True --head 2 --epochs 10 --GPU cuda:7 --batch_size 128

# 消融实验GBL, 
# 正常：把全局卷积中的GCU去掉
python main.py --algorithm GCA --use_data mcc5 --log_name ablationB_1_lr_1.log --lr 1 --blocker T --gcug '' --att agca --gcub True --head 2 --epochs 10 --GPU cuda:7 --batch_size 128
# 不用 blocker
python main.py --algorithm GCA --use_data mcc5 --log_name ablationB_2_lr_1.log --lr 1 --blocker '' --gcug '' --att agca --gcub True --head 2 --epochs 10 --GPU cuda:7 --batch_size 128
# 不使用通道注意力机制中的 GCU
python main.py --algorithm GCA --use_data mcc5 --log_name ablationB_3_lr_1.log --lr 1 --blocker T --gcug '' --att agca --gcub '' --head 2 --epochs 10 --GPU cuda:7 --batch_size 128
