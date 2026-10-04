# 可运行的建筑改造基准复现包

版本：`1.1.0`。这是一个独立、可完整执行的代码与数据流程包，覆盖从参考模型数据到训练、评估、优化和不确定性分析的完整链条。所有数据、预测、优化解和不确定性结果均由所附代码实际生成和计算；没有按论文 RMSE、MAPE 或排名预设结果。

本包验证的是"从参考模型数据到训练、评估、优化和不确定性分析"的可运行链条，并公开全部方程、假设与参数取值。

## 一次运行全部内容

在解压后的本目录执行，推荐 Python 3.12：

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell 改用 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/reproduce.py
python -m unittest discover -s tests -v
python scripts/verify_results.py
```

默认运行生成两年 15 分钟数据，训练 16 轮，使用 10,000 次区块自助法，运行 3 次优化（每次 24 个候选、16 代），并计算两种占用模型下每类方案的 500 个情景。运行会覆盖 **本包** 的 `data/`、`models/`、`results/` 中对应的输出文件。运行前无需已有结果文件。

缩小规模检查入口是否正常：

```bash
python scripts/reproduce.py --epochs 4 --bootstrap 500 --population 12 --generations 4 --scenarios 100
```

快速参数会产生不同的结果，不能与默认参数的结果表混用。CPU 主流程不需要 PyTorch 或 EnergyPlus。

## 已执行的流程和输出

| 流程 | 实际方法 | 输出 |
|---|---|---|
| 数据生成 | 固定种子的季节天气、占用、1R1C 理想温控模型和测量噪声 | `data/bms_15min.csv.gz`，70,080 行 |
| 55 维特征 | 47 个已知输入/历史负荷特征，3 个 RC 描述符，5 个最近聚类中心权重 | `data/features_55d.csv.gz` |
| 训练及消融 | NumPy 物理约束 MLP，以及相同初始化的无物理损失 MLP | 实际 `.npz` 权重、训练损失和测试预测 |
| 基线比较 | 真实拟合的线性回归、持续性预测、近似物理模型 | `results/model_metrics.csv` |
| 时间相关推断 | 672 点圆形移动区块；10,000 次成对重采样 | `results/block_bootstrap.json` |
| 外部域实验 | 独立生成的暖/冷气候数据；2/7/14 天微调与从零训练 | external、few-shot CSV |
| 60 例候选对照 | 近似物理能耗估计与 1R1C 参考模型对照 | `results/60_case_reference_comparison.csv` |
| 优化 | 实际种群演化、非支配排序、四目标参考方向选择 | Pareto 解、3 次运行轨迹、代表方案 |
| 不确定性 | 实际扰动天气、占用及 COP 后重新计算 | 情景 CSV 和 100/200/300/500 收敛表 |
| 神经模型扰动 | 对已训练 MLP 进行隐藏单元随机失活试验 | dropout 敏感性 CSV |
| 结果核验 | 加载权重重算预测；检查边界、非支配性和 SHA256 | `scripts/verify_results.py` |

详细方法、边界和单位见 [docs/METHODS.md](docs/METHODS.md)。本包训练数据每天改变一个建筑配置，以覆盖参数变化。

## 日期与切分

切分采用 count-aligned 数量并附明确日期：

| 集合 | 日期 | 15 分钟记录数 |
|---|---|---:|
| Training | 2022-01-01 至 2022-12-31 | 35,040 |
| Validation | 2023-01-01 至 2023-07-01 | 17,472 |
| Test | 2023-07-02 至 2023-12-31 | 17,568 |

不能同时把这组数量写成"验证至 6 月 30 日、测试从 7 月 1 日开始"。若改用后一种日历切分，数量分别是 17,376 和 17,664，代码和结果必须一起重算。

## 可选的 BiMamba 训练入口

`scripts/train_optional_bimamba.py` 调用 `mamba_ssm.Mamba`，使用历史窗口中的双向分支，并在训练中实际加入物理损失。它不会把 GRU 隐式称为 Mamba。

先在自己的兼容 PyTorch/CUDA 环境中按 [Mamba 官方说明](https://github.com/state-spaces/mamba) 安装，再执行：

```bash
python scripts/train_optional_bimamba.py --device cuda --epochs 50
```

本次 CPU 环境缺少 PyTorch/Mamba，因此该入口仅完成语法检查。安装后成功运行会生成 `optional_bimamba.pt` 和对应指标；本包随附的权重和指标来自 NumPy 物理约束 MLP。

## 参考模型与边界

本包的参考模型是显式 1R1C 理想温控模型，公开全部系数与方程；60 例表的参考端标注为 `1R1C_reference`。需要基于完整 IDF、EPW、运行脚本和模拟日志的全建筑能耗仿真时，请按 [docs/ENERGYPLUS_BOUNDARY.md](docs/ENERGYPLUS_BOUNDARY.md) 补充。外部域实验使用独立气候剖面，随机失活试验为模型扰动压力测试。

数据来源见 `PROVENANCE.json`。引用信息见 `CITATION.cff`。

## 验证状态

默认配置的端到端运行、从空输出目录重新运行、5 项功能测试，以及权重预测一致性和结果哈希检查，见 `VERIFICATION.json`。跨平台/不同 BLAS 版本可能产生浮点差异；固定种子不等于跨硬件逐字节一致。
