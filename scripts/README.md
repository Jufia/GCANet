# 实验脚本使用说明

本目录包含了用于运行GCANet实验的各种shell脚本。

## 脚本列表

### 1. `run_experiments.sh` - 完整实验套件
运行完整的实验套件，包括：
- 基础实验（不同数据集）
- 消融实验
- 不同学习率实验
- 不同head数量实验
- 不同算法对比实验

**使用方法：**
```bash
chmod +x scripts/run_experiments.sh
./scripts/run_experiments.sh
```

### 2. `custom_experiment.sh` - 自定义单次实验
运行单个自定义实验，可以修改脚本开头的参数配置。

**使用方法：**
```bash
# 编辑脚本开头的参数
vim scripts/custom_experiment.sh

# 运行脚本
chmod +x scripts/custom_experiment.sh
./scripts/custom_experiment.sh
```

### 3. `batch_run.sh` - 批量实验运行
批量运行多个预定义的实验配置。

**使用方法：**
```bash
# 编辑脚本中的experiments数组来添加实验配置
vim scripts/batch_run.sh

# 运行脚本
chmod +x scripts/batch_run.sh
./scripts/batch_run.sh
```

### 4. `GCANet.sh` - 原始实验脚本
包含原有的实验配置。

## 参数说明

### 主要参数
- `--algorithm`: 算法类型 (GCA, softshape, PatchTST, UniTS, DLinear, TCN, InceptionTime, NAT, MA1DCNN, ConvTran)
- `--use_data`: 数据集 (hit, xjtu, mcc5, dirg)
- `--lr`: 学习率
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--GPU`: GPU设备
- `--head`: 注意力头数
- `--log_name`: 日志文件名

### 模型特定参数
- `--blocker`: 是否使用Gradient Blocking Layer (T/空)
- `--gcug`: 是否在Global Convolution中使用GCU (T/空)
- `--attb`: 是否在Block中使用Attention (True/False)
- `--gcub`: 是否在Block中使用GCU (True/False)

## 日志文件

所有实验的日志文件保存在 `./checkpoint/log/` 目录中，文件名格式为：
- `{数据集}_{算法}_{参数}.log`

## 示例

### 运行单个实验
```bash
python main.py --algorithm GCA --use_data mcc5 --log_name test.log \
    --lr 0.001 --epochs 10 --batch_size 128 --GPU cuda:7
```

### 运行消融实验
```bash
# 去掉GCU
python main.py --algorithm GCA --use_data mcc5 --log_name no_gcug.log \
    --lr 0.001 --gcug '' --attb True --gcub True

# 不使用blocker
python main.py --algorithm GCA --use_data mcc5 --log_name no_blocker.log \
    --lr 0.001 --blocker '' --gcug '' --attb True --gcub True
```

## 注意事项

1. 运行脚本前确保已安装所有依赖
2. 确保GPU设备可用
3. 根据需要调整GPU设备号
4. 大量实验可能需要较长时间，建议使用screen或tmux
5. 定期检查日志文件以监控实验进度 