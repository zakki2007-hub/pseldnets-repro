# STARSS23 正式对照实验：AudioSet SEC 初始化 vs PSELDNets 空间预训练

日期：2026-10-23 · 数据：STARSS23 dev（168 片段，78 个 fold4 测试片段）· 模型：mACCDOA + HTS-AT

## 1. 实验设计

| 项目 | 设置 |
| --- | --- |
| 数据 | STARSS23 dev：90 训练片段（fold3，9 房间）、39 验证 / 39 测试（fold4，7 房间） |
| 音频 | 4 声道 FOA，24 kHz，来自 HuggingFace 镜像 `labhamlet/STARSS23`；标注用官方 `metadata_dev`（168 个 csv） |
| 输入 / 输出 | 7 通道 log-mel+IV，64 mel；13 类，mACCDOA 3 track |
| 训练 | `experiment=starss23/maccdoa_augmix1`（SEC）与 `finetune_maccdoa_augmix1`（SELD），AdamW lr 1e-4，batch 32，AugMix，seed 2024 |
| 轮数 | 两者均 **40 epoch**（每 2 epoch 验证一次；原配置 70，受时间限制统一缩短） |
| 测试 | fold4 的 78 个片段 / 1198 个 10 s chunk |
| 阈值 | 默认 0.5，以及 `infer.yaml` 中 STARSS23 的 13 类 per-class 阈值 |

**唯一变量**：初始化。

- SEC：HTS-AT AudioSet 权重（`HTSAT-fullset-imagenet-768d-32000hz.ckpt`），SELD 头随机。
- SELD：PSELDNets 合成空间预训练权重（`mACCDOA-HTSAT-0.567.ckpt`），SELD 头随机（代码跳过 `tscam_conv`/`fc`）。

## 2. 结果

### 2.1 验证集最佳（40 epoch 内）

| 初始化 | 最佳 val macro SELD | 对应 epoch | 最佳 val micro SELD |
| --- | --- | --- | --- |
| **SELD（PSELDNets）** | **0.325** | 35 | **0.289** |
| SEC（AudioSet） | 0.413 | 22 | 0.350 |

### 2.2 测试集（fold4，78 片段）

固定阈值 0.5：

| 初始化 | macro SELD | micro SELD | ER | F | LE | LR | 出现类数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SELD** | **0.3261** | **0.2891** | 0.4587 | 58.13 % | 15.33° | 65.80 % | **13/13** |
| SEC | 0.4149 | 0.3751 | 0.5705 | 43.85 % | 19.82° | 58.25 % | 7/13 |

STARSS23 per-class 阈值：

| 初始化 | macro SELD | micro SELD | ER | F | LE | LR | 出现类数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SELD** | **0.3212** | **0.2918** | 0.4825 | 58.26 % | 15.94° | 70.39 % | **13/13** |
| SEC | 0.4640 | 0.4215 | 0.7700 | 37.88 % | 21.20° | 65.29 % | 9/13 |

> macro 的“出现类数”不同（SEC 只有 7–9 类有匹配 track），因此 **micro 和类别覆盖率** 是更公平的对比；两种阈值下 SELD 初始化的 micro SELD 都低 0.086–0.130。

### 2.3 收敛曲线

见 `starss23_curves.png`：

- SELD 初始化从第 2 个 epoch 起就明显更低，之后稳定在 macro 0.32–0.35、micro 0.29–0.31。
- SEC 初始化前 4 个 epoch 快速下降，但随后停在 macro 0.41–0.50、micro 0.35–0.43，且 F 在 30–45 % 之间波动，明显低于 SELD 的 50–59 %。
- SEC 的 F 在 epoch 26 附近有一次明显下跌（macro F 42→30 %），SELD 相对稳定。

## 3. 结论

1. **空间预训练确实有用**：在完全相同的 STARSS23 数据、增强、学习率、batch、epoch、seed 下，PSELDNets 初始化的 micro SELD 从 0.375 降到 0.289（固定阈值），相对提升约 **23 %**；per-class 阈值下从 0.422 降到 0.292。
2. **检测与定位同时受益**：F 从 43.9 % 升到 58.1 %，ER 从 0.571 降到 0.459，LE 从 19.8° 降到 15.3°，LR 从 58.3 % 升到 65.8 %。
3. **类别覆盖更完整**：SEC 在阈值 0.5 下只有 7/13 类产生匹配 track，SELD 覆盖全部 13 类；说明 SEC 初始化在稀有类上更容易“完全不报”。
4. 与合成数据上的初步实验方向一致：空间预训练带来的收益在真实录制数据上仍然存在，但差距从“完全塌缩”缩小为“稳定但较小”（micro 0.086）。

## 4. 局限（写论文前必须补）

- **轮数**：两者都只训了 40 epoch（原配置 70），SEC 可能尚未完全收敛；需补 70 epoch 确认差距是否缩小。
- **单 seed**：目前只有 seed 2024，需 3 seeds 报均值±方差。
- **验证/测试同分布**：仓库 `starss23/default.yaml` 的 valid 与 test 都是 fold4 七个房间；严格说测试集参与了模型选择，需改用独立划分或只报最后一轮。
- **macro 不可直接比**：SEC 出现类少，macro 分母不同；论文主表建议以 micro 为主，macro 附类别覆盖说明。
- **数据来源**：音频来自 HF 镜像（官方 STARSS23 dev 的 4 声道 FOA），标注为官方 metadata；建议用官方 Zenodo 包复核一次。
- **阈值**：per-class 阈值来自 `infer.yaml` 注释，未针对两个模型分别调参；固定 0.5 的结果同样报告。

## 5. 复现命令

```bash
# 预处理（已执行）
python src/preproc.py dataset=STARSS23 dataset_type=dev wav_format=.wav data=starss23/default

# SEC 初始化
python src/train.py experiment=starss23/maccdoa_augmix1 compile=false num_workers=8 \
  trainer.num_sanity_val_steps=0 trainer.check_val_every_n_epoch=2 trainer.max_epochs=40

# SELD 初始化
python src/train.py experiment=starss23/finetune_maccdoa_augmix1 compile=false num_workers=8 \
  trainer.num_sanity_val_steps=0 trainer.check_val_every_n_epoch=2 trainer.max_epochs=40

# 推理（固定阈值 0.5，per-class 阈值见 infer.yaml 注释）
python src/infer.py experiment=starss23/finetune_maccdoa_augmix1 data=starss23/default mode=test \
  ckpt_path=<best.ckpt> model.kwargs.pretrained_path=null model.kwargs.audioset_pretrain=false \
  model.batch_size=8 num_workers=4 compile=false sed_threshold=0.5
```

## 6. 文件

- `_analysis/starss23_sec_0.5/results.json`、`starss23_seld_0.5/results.json`：固定阈值结果
- `_analysis/starss23_sec_thr/results.json`、`starss23_seld_thr/results.json`：per-class 阈值结果
- `_analysis/val_starss23_sec.csv`、`val_starss23_seld.csv`：逐 epoch 验证曲线
- `_analysis/starss23_curves.png`：验证曲线图
- 训练日志：`logs/train_starss23_sec.log`、`logs/train_starss23_seld.log`
- 权重：`logs/multi_accdoa_HTSAT/runs/2026-10-23_10-18-25/checkpoints/epoch_023.ckpt`（SEC）、`.../2026-10-23_10-39-13/checkpoints/epoch_035.ckpt`（SELD）
