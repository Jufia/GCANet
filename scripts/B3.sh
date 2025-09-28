
for s in 42 3470 215
do
   python main.py --log_name ablationB_xjtu_1_${s}.log --GPU cuda:0 --random_state $s --use_data xjtu
   python main.py --log_name ablationB_xjtu_2_${s}.log --blocker '' --GPU cuda:0 --random_state $s --use_data xjtu
   python main.py --log_name ablationB_xjtu_3_${s}.log --gcub none --GPU cuda:0 --random_state $s --att se --use_data xjtu
done

python main.py --log_name ablationB3_42.log --gcub none --GPU cuda:0 --random_state 42 --att se
python main.py --log_name ablationB3_3470.log --gcub none --GPU cuda:0 --random_state 3470 --att se
