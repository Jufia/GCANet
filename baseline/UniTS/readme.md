# Overall
paper: UniTS: A Unified Multi-Task Time Series Model (NIPS 2024)
code: https://github.com/mims-harvard/UniTS

# Paramenters setting
UniTS是一共通用模型，但本project只关注‘classification’任务，在确保不改变分类任务模型结构的前提下已删除与‘classification’任务无关的代码增强UniTS的可读性。

超参数按照论文中的描述设置  
1. 从头开始训练的模型 (UNITS-SUP)，d_model=64
2. batch_size=32
3. prompt tokens number=10
4. patch size=16
5. dropout=0.1
6. epoch=5
7. learning rate=3.2e-2
8. batch size=1024