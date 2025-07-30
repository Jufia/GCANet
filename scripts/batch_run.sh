#!/bin/bash

# 批量实验运行脚本
# 使用方法: ./batch_run.sh

# 实验配置数组 - 可以添加多个实验配置
declare -a experiments=(
    # 格式: "数据集:算法:学习率:轮数:批次大小:GPU:头数:日志名"
    "mcc5:PatchTST:1e-3:20:"
    "mcc5:GCA:0.001:10:128:cuda:7:2:mcc5_GCA_baseline"
    "hit:GCA:0.001:10:128:cuda:7:2:hit_GCA_baseline"
    "xjtu:GCA:0.001:10:128:cuda:7:2:xjtu_GCA_baseline"
    "mcc5:GCA:0.0005:10:128:cuda:7:2:mcc5_GCA_lr0005"
    "mcc5:GCA:0.005:10:128:cuda:7:2:mcc5_GCA_lr005"
    "mcc5:softshape:0.001:10:128:cuda:7:2:mcc5_softshape"
    "mcc5:PatchTST:0.001:10:128:cuda:7:2:mcc5_PatchTST"
)

# 创建日志目录
mkdir -p ./checkpoint/log/

echo "开始批量运行实验..."
echo "总共 ${#experiments[@]} 个实验配置"

# 遍历所有实验配置
for i in "${!experiments[@]}"; do
    # 解析实验配置
    IFS=':' read -r dataset algorithm lr epochs batch_size gpu head log_name <<< "${experiments[$i]}"
    
    echo ""
    echo "=== 运行第 $((i+1)) 个实验 ==="
    echo "数据集: $dataset"
    echo "算法: $algorithm"
    echo "学习率: $lr"
    echo "轮数: $epochs"
    echo "批次大小: $batch_size"
    echo "GPU: $gpu"
    echo "头数: $head"
    echo "日志: $log_name"
    
    # 运行实验
    python main.py \
        --algorithm $algorithm \
        --use_data $dataset \
        --log_name $log_name \
        --lr $lr \
        --epochs $epochs \
        --batch_size $batch_size \
        --GPU $gpu \
        --head $head \
        --blocker T \
        --gcug T \
        --attb True \
        --gcub True
    
    echo "第 $((i+1)) 个实验完成！"
done

echo ""
echo "所有实验运行完成！"
echo "日志文件保存在 ./checkpoint/log/ 目录中"

# 显示所有日志文件
echo ""
echo "生成的日志文件:"
ls -la ./checkpoint/log/ 