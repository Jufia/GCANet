paper: LightTS: Lightweight Time Series Classification with Adaptive Ensemble Distillation

Link: https://dl.acm.org/doi/10.1145/3589316

Code: https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py

超参数: https://github.com/thuml/Time-Series-Library/blob/main/scripts/classification/LightTS.sh
```js
--train_epochs 100
--batch_size 16
--learning_rate 0.001
--optimizer 'radam'
```

cite
```bibtex
@article{10.1145/3589316,
author = {Campos, David and Zhang, Miao and Yang, Bin and Kieu, Tung and Guo, Chenjuan and Jensen, Christian S.},
title = {LightTS: Lightweight Time Series Classification with Adaptive Ensemble Distillation},
year = {2023},
issue_date = {June 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {1},
number = {2},
url = {https://doi.org/10.1145/3589316},
doi = {10.1145/3589316},
abstract = {Due to the sweeping digitalization of processes, increasingly vast amounts of time series data are being produced. Accurate classification of such time series facilitates decision making in multiple domains. State-of-the-art classification accuracy is often achieved by ensemble learning where results are synthesized from multiple base models. This characteristic implies that ensemble learning needs substantial computing resources, preventing their use in resource-limited environments, such as in edge devices. To extend the applicability of ensemble learning, we propose the LightTS framework that compresses large ensembles into lightweight models while ensuring competitive accuracy. First, we propose adaptive ensemble distillation that assigns adaptive weights to different base models such that their varying classification capabilities contribute purposefully to the training of the lightweight model. Second, we propose means of identifying Pareto optimal settings w.r.t. model accuracy and model size, thus enabling users with a space budget to select the most accurate lightweight model. We report on experiments using 128 real-world time series sets and different types of base models that justify key decisions in the design of LightTS and provide evidence that LightTS is able to outperform competitors.},
journal = {Proc. ACM Manag. Data},
month = jun,
articleno = {171},
numpages = {27},
keywords = {knowledge distillation, tinyML}
}
```