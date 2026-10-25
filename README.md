# PSELDNets 复现、评测与初始化对照实验

[PSELDNets](https://github.com/Jinbo-Hu/PSELDNets)（mACCDOA + HTS-AT）的**独立复现、指标核对、误差分析**，以及一组**控制变量实验**：在相同数据、增强、学习率和训练轮数下，对比 AudioSet SEC 初始化与 PSELDNets 空间预训练初始化。

本仓库不包含模型代码、权重和数据，只提供复现脚本、评测工具、分析结果、实验曲线与一份上游 bug 的修复补丁。

## 结果（360 片段 / 2160 个 10 s chunk）

| 平均方式 | ER↓ | F↑ | LE↓ | LR↑ | SELD↓ |
| --- | --- | --- | --- | --- | --- |
| macro（论文采用） | 0.7793 | 22.07 % | 20.16° | 26.96 % | **0.6002** |
| micro | 0.7793 | 25.37 % | 19.06° | 29.92 % | 0.5831 |

- 误差分解：Nref = 46 132，Substitution = 16 584，Deletion = 15 763，Insertion = 3 602。
- 检测（20° 匹配）：TP = 9 640，FP = 16 016，FP_spatial = 4 170，FN = 32 347；recall@20° = 22.96 %，precision@20° = 32.32 %。
- 与模型自身 `trainer.validate`（0.601）及官方 `evaluate.py --consolidated_score`（0.600 / 22.1 / 20.2 / 27.0）三方一致。
- 两次独立推理的预测文件逐字节相同，可复现。

## 主要发现

### 1. 官方评测脚本的类别索引 bug

`src/evaluate.py` 的逐类打印按 `range(args.nb_classes)`（170）循环，但 `SELD_metrics` 只保留测试集中出现的类（158），运行到第 158 行必然：

```
IndexError: index 158 is out of bounds for axis 0 with size 158
```

修复补丁见 [`patches/fix-evaluate-classwise.patch`](patches/fix-evaluate-classwise.patch)：按实际类数循环，并打印真实类别 ID（原代码打印的是 0..157 的序号）。补丁已验证：输出 158 行、不再崩溃，macro/micro 结果与 `--consolidated_score` 一致。

### 2. 与 checkpoint 名称中 0.567 的差异

本地只有 `test360_ov3`（360 片段，最大重叠 3）；论文的 synthetic-test-set 是 **3 060 片段**（`test1800_ov1` + `test900_ov2` + `test360_ov3`）。0.567 应是完整测试集（或训练验证集）的分数，与本地 0.600 不可直接比较。

### 3. 阈值扫描（仅改 `sed_threshold`）

| 阈值 | ER | F(macro) | LR | SELD(macro) | Insertions |
| --- | --- | --- | --- | --- | --- |
| 0.2 | 1.0265 | 21.57 % | 34.44 % | 0.6477 | 18 891 |
| 0.3 | 0.8659 | 22.00 % | 31.15 % | 0.6137 | 9 902 |
| **0.5** | **0.7793** | **22.07 %** | 26.96 % | **0.6002** | 3 602 |
| 0.7 | 0.7795 | 21.78 % | 23.11 % | 0.6079 | 1 421 |

F 在所有阈值下都停在 21–22 %：瓶颈是类别判别与召回，而不是工作点。默认 0.5 最优。

## 初始化对照实验

唯一变量是初始化，其余（数据、增强、lr、batch、epoch、seed）完全一致：

- **SEC**：HTS-AT 的 AudioSet 权重（`audioset_pretrain=true`），SELD 头随机。
- **SELD**：PSELDNets 合成空间预训练权重（`audioset_pretrain=false`），SELD 头随机。

### STARSS23（正式对照）

90 个 fold3 片段训练，fold4 的 78 个片段测试（1198 个 10 s chunk），13 类，40 epoch，seed 2024。

固定阈值 0.5：

| 初始化 | macro SELD | micro SELD | ER | F | LE | LR | 出现类 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SELD（PSELDNets）** | **0.3261** | **0.2891** | 0.4587 | 58.13 % | 15.33° | 65.80 % | **13/13** |
| SEC（AudioSet） | 0.4149 | 0.3751 | 0.5705 | 43.85 % | 19.82° | 58.25 % | 7/13 |

STARSS23 per-class 阈值下 micro SELD 为 SELD **0.2918** / SEC 0.4215。SEC 在阈值 0.5 下只有 7/13 类产生匹配 track，macro 的分母不同，因此以 **micro 和类别覆盖率** 为准：空间预训练带来约 23 % 的 micro SELD 相对提升。

详见 [`report/starss23_experiment.md`](report/starss23_experiment.md) 与 [`report/starss23_curves.png`](report/starss23_curves.png)。

> 注意：音频来自 HuggingFace 镜像 `labhamlet/STARSS23`（官方 STARSS23 dev 的 4 声道 FOA），标注为官方 metadata；两者均只训 40 epoch（原配置 70）、单 seed，且仓库配置中 valid 与 test 同为 fold4。绝对分数不能直接与论文/DCASE 结果比较，比较的是**相对趋势**。

### 合成数据（初步）

`test360_ov3` 按房间切分：rooms 1–5 训练（1200 chunk）、rooms 6–7 验证、rooms 8–9 测试，25 epoch。

| 初始化 | 最佳 val SELD | 测试 macro SELD | 测试 micro SELD | F | 说明 |
| --- | --- | --- | --- | --- | --- |
| **SELD（PSELDNets）** | 0.403 | 0.3699 | 0.3783 | 54.39 % | 正常收敛 |
| SEC（AudioSet） | 1.000 | 1.000 | 1.000 | 0 % | 25 epoch 内未收敛 |

SEC 在 170 类设置下输出塌缩（活动度集中在 0.01–0.1），因此这组只作为初步观察，不作为“SEC 无效”的结论。详见 [`report/exp_init_comparison.md`](report/exp_init_comparison.md) 与 [`report/exp_init_curves.png`](report/exp_init_curves.png)。

## 仓库结构

```
report/     REPORT.pdf / REPORT.html / REPORT.md   合成测试集复现报告
            classwise.csv                          158 个出现类的 ER/F/LE/LR/SELD
            results.json                           逐类/逐房间/逐片段完整结果
            starss23_experiment.md                  STARSS23 初始化对照报告
            starss23_curves.png                     验证曲线
            exp_init_comparison.md                  合成数据初始化对照报告
            exp_init_curves.png                     验证曲线
            val_*.csv                               逐 epoch 验证指标
scripts/    eval_analysis.py                       评测与切片分析脚本
            run_inference.sh                       复现推理命令
            synth_eval_only.yaml                   仅测试集的 Hydra 数据配置
            synth_split.yaml                       按房间切分的合成数据配置
patches/    fix-evaluate-classwise.patch            上游 bug 修复
```

## 快速开始

```bash
# 1) 获取上游代码
git clone https://github.com/Jinbo-Hu/PSELDNets.git
cd PSELDNets

# 2) 准备权重与数据（见上游 README）
#    ckpts/mACCDOA-HTSAT-0.567.ckpt
#    datasets/test360_ov3/  （FOA + metadata）
#    并按上游 scripts/prepare_DataSynthSELD.bash 生成 _hdf5/

# 3) 加入仅测试集配置
cp /path/to/pseldnets-repro/scripts/synth_eval_only.yaml configs/data/

# 4) 推理（约 1 分钟 / RTX 4090）
bash /path/to/pseldnets-repro/scripts/run_inference.sh

# 5) 评测与切片分析
python /path/to/pseldnets-repro/scripts/eval_analysis.py \
  --pred_dir logs/multi_accdoa_HTSAT_infer/runs/<run>/submissions
```

`eval_analysis.py` 支持 `--gt_dir`、`--label_h5`、`--nb_classes`、`--out_dir`，输出总体 macro/micro、逐房间、逐片段、检测 TP/FP/FN 与逐类 CSV。

## 环境

Python 3.11，`torch==2.2.1+cu121`、`torchaudio==2.2.1`、`lightning==2.2.1`、`hydra-core==1.3.2`；GPU 为 RTX 4090 24 GB。完整版本见 [`requirements.txt`](requirements.txt)。

## 限制

- 合成测试集只评测了 `test360_ov3`（360/3060 片段），未覆盖 `test1800_ov1` / `test900_ov2`，因此 0.600 与 checkpoint 名称中的 0.567 尚未直接对齐。
- macro 指标只对出现类平均（合成 158/170，STARSS23 中 SEC 仅 7–9/13），与“全类平均”不可混用。
- STARSS23 对照为 40 epoch、单 seed；仓库配置中 valid 与 test 同为 fold4，测试集参与了模型选择。
- STARSS23 音频来自 HuggingFace 镜像，未逐文件与官方 Zenodo 包做波形比对。
- 本仓库为第三方复现，非官方实现；结论以官方论文与代码为准。

## 致谢与许可

模型、训练代码与数据来自 [Jinbo-Hu/PSELDNets](https://github.com/Jinbo-Hu/PSELDNets)（MIT License, Copyright (c) 2025 Jinbo Hu）。本仓库的脚本、报告与补丁同样以 MIT 协议发布，见 [`LICENSE`](LICENSE)。

## 引用

```bibtex
@article{hu2024pseldnets,
  title   = {PSELDNets: Pre-trained Neural Networks on Large-scale Synthetic Datasets for Sound Event Localization and Detection},
  author  = {Hu, Jinbo and Cao, Yin and Wu, Ming and Kang, Fang and Yang, Feiran and Wang, Wenwu and Plumbley, Mark D. and Yang, Jun},
  journal = {arXiv preprint arXiv:2411.06399},
  year    = {2024}
}
```
