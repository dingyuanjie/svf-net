# SVF-Net

Structural Vital Force Network：在 Transformer 注意力中加入结构势场的实验性架构。

当前版本：`v0.1-alpha`。目标是用可复现实验检验结构项是否带来收益，不宣称结构生力理论已经得到物理验证。

## 核心公式

标准注意力：

```text
softmax(QKᵀ / sqrt(d)) V
```

SVF 注意力：

```text
softmax(QKᵀ / sqrt(d) + λS) V
```

其中 `S` 由可学习结构角和结构半径构成。

## Ablation 模式

`train.py` 支持三个模式：

- `baseline`：标准因果 Transformer，不使用结构场
- `svf`：可学习结构角/半径的结构势场
- `random`：固定随机结构场，用来排除“任意偏置都有效”的可能

三种模式使用相同模型规模和训练流程。当前默认是 Tiny 配置：6 层、256 hidden、8 heads、512 context。

## 运行

```powershell
python -m pip install -r requirements.txt
python train.py --model baseline --steps 1000
python train.py --model svf --steps 1000
python train.py --model random --steps 1000
```

NVIDIA CUDA 环境请使用 CUDA 12.8 依赖：

```powershell
python -m pip install -r requirements-cuda.txt
```

## 一条命令运行完整消融实验

推荐直接运行统一入口：

```powershell
python run_experiment.py
```

它会依次运行 baseline、svf、random 三组实验，并将结果写入 `outputs/ablation/`。默认使用长距离复制任务：

```text
随机前半段 A B C D ... → 复制后半段 A B C D ...
```

模型必须从较远位置检索前半段，才能预测后半段。每组实验还会定期计算固定验证集 loss。

```powershell
python run_experiment.py --task copy --steps 1000 --device cuda
```

如需复现之前的随机 token 冒烟测试：

```powershell
python run_experiment.py --task random --steps 1000 --device cuda
```

推荐先运行较小的可诊断复制基准：

```powershell
python run_experiment.py --task copy --small-copy --steps 5000 --eval-every 250 --device cuda
```

该模式会额外记录 `copy_accuracy`，只统计复制区间的预测准确率。

进行多随机种子稳定性验证：

```powershell
python run_multiseed.py --seeds 1,2,3,4,5 --steps 5000 --device cuda
```

结果位于 `outputs/multiseed/summary.json`，包含每个模型的均值和标准差。

扫描结构场强度：

```powershell
python run_lambda_sweep.py --lambdas 0,0.01,0.05,0.1,0.2,0.5 --seeds 1,2,3 --steps 5000 --device cuda
```

结果位于 `outputs/lambda-sweep/summary.json`。

结果文件包括：

- `baseline.csv`、`svf.csv`、`random.csv`：训练 loss 和验证 loss
- `summary.json`：任务、配置、参数量、初始 loss、最终 loss、验证 loss 和耗时

快速验证可使用：

```powershell
python run_experiment.py --steps 10 --device cpu
```

训练脚本目前使用合成 token 数据进行工程冒烟测试；要得到有意义的语言建模结论，需要接入相同 tokenizer 和数据切分的 TinyStories 或 WikiText。

每次启动会打印模型模式、参数量、设备和训练 loss。建议保存三组完整日志后比较最终 loss、收敛速度和不同 context 长度下的验证 loss。
