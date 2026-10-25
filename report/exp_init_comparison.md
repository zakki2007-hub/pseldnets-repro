# 初始化对照实验：AudioSet SEC vs PSELDNets 空间预训练

日期：2026-10-22 · 数据：`test360_ov3` 按房间切分（STARSS23 因下载受限，先用合成数据做同款对照）

## 1. 实验设计

| 项目 | 设置 |
| --- | --- |
| 训练集 | `fold1_room1`–`fold1_room5`，200 个片段 / 1200 个 10 s chunk |
| 验证集 | `fold1_room6`–`fold1_room7`，80 个片段 / 480 chunk |
| 测试集 | `fold1_room8`–`fold1_room9`，80 个片段 / 480 chunk |
| 输入 | FOA 7 通道 log-mel+IV，64 mel，24 kHz |
| 类别 / 输出 | 170 类，mACCDOA 3 track |
| 增强 | AugMix（`experiment=synth_maccdoa`） |
| 优化 | AdamW，lr 1e-4，batch 32，25 epoch，seed 2024，StepLR(20) |
| 评测 | DCASE SELD，DOA 阈值 20°，`sed_threshold=0.5` |

**唯一变量**是初始化：

- **SEC 初始化**：HTS-AT 的 AudioSet 权重（`ckpts/HTSAT-fullset-imagenet-768d-32000hz.ckpt`），`audioset_pretrain=true`，SELD 头随机初始化。
- **SELD 初始化**：PSELDNets 合成空间预训练权重（`ckpts/mACCDOA-HTSAT-0.567.ckpt`），`audioset_pretrain=false`，SELD 头同样随机初始化（代码跳过 `tscam_conv`/`fc`）。

```bash
# SEC
python src/train.py experiment=synth_maccdoa data=synth_split \
  model.kwargs.pretrained_path=ckpts/HTSAT-fullset-imagenet-768d-32000hz.ckpt \
  model.kwargs.audioset_pretrain=true compile=false num_workers=8

# SELD
python src/train.py experiment=synth_maccdoa data=synth_split \
  model.kwargs.pretrained_path=ckpts/mACCDOA-HTSAT-0.567.ckpt \
  model.kwargs.audioset_pretrain=false compile=false num_workers=8
```

## 2. 结果

| 初始化 | 最佳 val macro SELD | 测试 macro SELD | ER | F | LE | LR |
| --- | --- | --- | --- | --- | --- | --- |
| **SELD（PSELDNets）** | **0.403**（epoch 22） | **0.3699** | 0.523 | 54.39 % | 12.57° | 56.93 % |
| SEC（AudioSet HTSAT） | 1.000（无检测） | 1.000（无检测） | 1.000 | 0 % | 180° | 0 % |

SELD 初始化的 micro 指标：SELD 0.3783 / F 54.05 % / ER 0.523 / LE 12.78° / LR 54.05 %；检测 TP 4627、FP 1311、FP_spatial 920、FN 4715，recall@20° 49.5 %，precision@20° 67.5 %。

验证曲线见 `exp_init_curves.png`：

- SELD 初始化：第 2 个 epoch 即开始下降，第 6 个 epoch 到 0.46，之后稳定在 0.40–0.43。
- SEC 初始化：25 个 epoch 全部为 1.000（F 显示 -100 %，表示没有任何匹配 track）。

## 3. SEC 初始化失败诊断

这不是简单的“分数低”，而是**输出塌缩**：

- 训练 loss 始终停在 0.0036（约等于“全部预测为 0”的损失），25 个 epoch 没有下降。
- `tscam_conv` 头权重标准差从 0.008463 变为 0.008440，几乎没有更新。
- 阈值 0.5：测试集 0 个预测；阈值 0.1：仍然 0 个；阈值 0.01：突然出现 389 598 行预测。
- 说明模型输出的活动度集中在 0.01–0.1，没有学会把真实事件推到 1。

可能原因（尚需验证）：

1. AudioSet 权重是 32 kHz / 单通道 / 128 mel 配置，本实验是 24 kHz / 7 通道 / 64 mel；代码用“通道重复 + 除以通道数”适配 `patch_embed`，特征分布可能不匹配。
2. 170 类、3 track、随机头，25 epoch / 950 step 对 SEC 初始化可能远远不够。
3. 输入 `scalar` BatchNorm 从 AudioSet 的 `bn0` 拷贝，归一化尺度与本数据不一致，可能导致激活过小/死亡。

因此，**目前不能得出“SEC 预训练没用”的结论**；只能说“按仓库默认 SEC 初始化路径，在 25 epoch 的同预算下没有收敛”。需要按 STARSS23 原设置（13 类、70 epoch）复核。

## 4. 阶段性结论

- 在低资源（200 片段）、170 类的合成 SELD 任务上，**PSELDNets 空间预训练带来明显的收敛优势**：val SELD 0.403、测试 0.370，F 54 %。
- AudioSet 初始化在相同预算下完全未收敛，提示“普通声音识别预训练”与“空间 SELD 预训练”之间存在显著差距，但当前证据受实现细节和训练预算影响，暂不作为最终结论。
- 该结果支持继续在 STARSS23 上做正式对照（13 类、70 epoch、3 seeds），预期差距会缩小但方向一致。

## 5. 下一步

1. **STARSS23 正式实验**：数据到位后，用 `starss23/maccdoa_augmix1`（SEC）和 `starss23/finetune_maccdoa_augmix1`（SELD），70 epoch、batch 32、lr 1e-4，3 个 seed。
2. **修正 SEC 基线**：单独验证 `audioset_pretrain=true` 的权重加载（跳过 `scalar` 或调整 `patch_embed` 缩放），排除实现问题。
3. **补充随机初始化对照**：区分“AudioSet 初始化有害”和“随机头 + 170 类太难”。
4. **报告方式**：主表报 macro SELD / ER / F / LE / LR，附图报 val 曲线；同时给出每类结果与收敛所需 epoch。

## 6. 复现材料

- `val_seld_init.csv`、`val_sec_init.csv`：逐 epoch 验证指标
- `exp_init_curves.png`：验证曲线
- `exp_seld_0.5/results.json`、`classwise.csv`：SELD 初始化测试集完整结果
- 训练日志：`logs/train_seld_init.log`、`logs/train_seld_resume.log`、`logs/train_sec_init.log`
- 数据切分：`configs/data/synth_split.yaml`
