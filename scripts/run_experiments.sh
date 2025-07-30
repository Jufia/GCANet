#!/bin/bash

# 实验运行脚本
# 使用方法: ./run_experiments.sh

# 设置PyTorch环境变量以解决CUDA DSA警告
export TORCH_USE_CUDA_DSA=1

echo "开始运行实验..."

# 设置基础参数
BASE_LR=0.001
BASE_EPOCHS=10
BASE_BATCH_SIZE=128
GPU="cuda:7"

# 创建日志目录
mkdir -p ./checkpoint/log/

# 1. 基础实验 - 不同数据集
echo "=== 运行基础实验 ==="

# HIT数据集
echo "运行HIT数据集实验..."
python -u main.py --algorithm GCA --use_data hit --log_name hit_GCA.log \
    --lr $BASE_LR --blocker T --gcug T --attb True --gcub True \
    --head 2 --epochs $BASE_EPOCHS --GPU $GPU --batch_size $BASE_BATCH_SIZE

# XJTU数据集
echo "运行XJTU数据集实验..."
python -u main.py --algorithm GCA --use_data xjtu --log_name xjtu_GCA.log \
    --lr $BASE_LR --blocker T --gcug T --attb True --gcub True \
    --head 2 --epochs $BASE_EPOCHS --GPU $GPU --batch_size $BASE_BATCH_SIZE

# MCC5数据集
echo "运行MCC5数据集实验..."
python -u main.py --algorithm GCA --use_data mcc5 --log_name mcc5_GCA.log \
    --lr $BASE_LR --blocker T --gcug T --attb True --gcub True \
    --head 2 --epochs $BASE_EPOCHS --GPU $GPU --batch_size $BASE_BATCH_SIZE

# 2. 消融实验
echo "=== 运行消融实验 ==="

# 消融实验1: 去掉全局卷积中的GCU
echo "消融实验1: 去掉全局卷积中的GCU..."
python -u main.py --algorithm GCA --use_data mcc5 --log_name ablation_no_gcug.log \
    --lr $BASE_LR --blocker T --gcug '' --attb True --gcub True \
    --head 2 --epochs $BASE_EPOCHS --GPU $GPU --batch_size $BASE_BATCH_SIZE

# 消融实验2: 不使用blocker
echo "消融实验2: 不使用blocker..."
python -u main.py --algorithm GCA --use_data mcc5 --log_name ablation_no_blocker.log \
    --lr $BASE_LR --blocker '' --gcug '' --attb True --gcub True \
    --head 2 --epochs $BASE_EPOCHS --GPU $GPU --batch_size $BASE_BATCH_SIZE

# 消融实验3: 不使用通道注意力机制中的GCU
echo "消融实验3: 不使用通道注意力机制中的GCU..."
python -u main.py --algorithm GCA --use_data mcc5 --log_name ablation_no_gcub.log \
    --lr $BASE_LR --blocker T --gcug '' --attb True --gcub '' \
    --head 2 --epochs $BASE_EPOCHS --GPU $GPU --batch_size $BASE_BATCH_SIZE

# 3. 不同学习率实验
echo "=== 运行不同学习率实验 ==="

for lr in 0.0001 0.0005 0.001 0.005 0.01; do
    echo "运行学习率 $lr 的实验..."
    python main.py --algorithm GCA --use_data mcc5 --log_name lr_${lr}.log \
        --lr $lr --blocker T --gcug T --attb True --gcub True \
        --head 2 --epochs $BASE_EPOCHS --GPU $GPU --batch_size $BASE_BATCH_SIZE
done

# 4. 不同head数量实验
echo "=== 运行不同head数量实验 ==="

for head in 1 2 4 8; do
    echo "运行head数量 $head 的实验..."
    python main.py --algorithm GCA --use_data mcc5 --log_name head_${head}.log \
        --lr $BASE_LR --blocker T --gcug T --attb True --gcub True \
        --head $head --epochs $BASE_EPOCHS --GPU $GPU --batch_size $BASE_BATCH_SIZE
done

# 5. 不同算法对比实验
echo "=== 运行不同算法对比实验 ==="

for algorithm in GCA softshape PatchTST UniTS DLinear TCN InceptionTime NAT MA1DCNN ConvTran; do
    echo "运行算法 $algorithm 的实验..."
    python main.py --algorithm $algorithm --use_data mcc5 --log_name ${algorithm}_mcc5.log \
        --lr $BASE_LR --epochs $BASE_EPOCHS --GPU $GPU --batch_size $BASE_BATCH_SIZE
done

echo "所有实验运行完成！"
echo "日志文件保存在 ./checkpoint/log/ 目录中" 