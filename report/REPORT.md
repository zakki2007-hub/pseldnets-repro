# PSELDNets 评测报告 — 合成测试集 `test360_ov3`

日期：2026-10-21 · 工作目录：`/data/PSELDNets`

## 1. 测试设置

| 项目 | 内容 |
| --- | --- |
| 环境 | conda `pseldnets`（Python 3.11，torch 2.2.1+cu121，lightning 2.2.1） |
| GPU | NVIDIA GeForce RTX 4090 24 GB |
| 模型 | `ckpts/mACCDOA-HTSAT-0.567.ckpt`（mACCDOA + HTS-AT，34.6 M 参数） |
| 数据 | `datasets/test360_ov3`，360 个 1 分钟片段，最大重叠数 3；切为 2160 个 10 s chunk |
| 真值 | `datasets/test360_ov3/metadata`（360 个 DCASE 格式 csv，46 132 个"类-段"track 标注） |
| 指标 | DCASE SELD：ER / F / LE / LR / SELD_scr（DOA 阈值 20°，1 s 段，macro 对 158 个出现类平均） |
| SED 阈值 | `sed_threshold=0.5`（`configs/infer.yaml` 默认值） |

推理命令（与 README Quick evaluate 一致）：

```bash
python src/infer.py experiment=synth_maccdoa data=synth_eval_only mode=test \
  ckpt_path=ckpts/mACCDOA-HTSAT-0.567.ckpt \
  model.kwargs.pretrained_path=null model.batch_size=8 num_workers=4 compile=false
```

- 2160 个 chunk 推理约 1 分钟，输出 360 个预测 csv。
- 两次独立运行（`21-03-18` 与 `21-48-59`）的预测文件逐字节一致 → 推理可复现。
- 模型自身 `trainer.validate`（同一数据、同一 pipeline）日志给出 macro SELD **0.601**，与下文外部评测 **0.600** 一致，说明评测链路没有对齐/版本问题。

## 2. 总体结果

| 平均方式 | ER↓ | F↑ | LE↓ | LR↑ | SELD↓ |
| --- | --- | --- | --- | --- | --- |
| macro（论文采用） | 0.7793 | 22.07 % | 20.16° | 26.96 % | **0.6002** |
| micro | 0.7793 | 25.37 % | 19.06° | 29.92 % | 0.5831 |

误差分解（类-段级别）：Nref = 46 132，Substitution = 16 584，Deletion = 15 763，Insertion = 3 602，因此

`ER = (16584 + 15763 + 3602) / 46132 = 0.779`

检测层面（20° 匹配）：TP = 9 640，FP = 16 016，FP(空间超限) = 4 170，FN = 32 347
→ **recall@20° = 22.96 %，precision@20° = 32.32 %**。

## 3. SED 阈值敏感性（同一模型，仅改 `sed_threshold`）

| sed_threshold | ER | F(macro) | LE | LR | SELD(macro) | SELD(micro) | Insertions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 1.0265 | 21.57 % | 22.38° | 34.44 % | 0.6477 | 0.6320 | 18 891 |
| 0.3 | 0.8659 | 22.00 % | 21.64° | 31.15 % | 0.6137 | 0.5961 | 9 902 |
| **0.5** | **0.7793** | **22.07 %** | **20.16°** | 26.96 % | **0.6002** | **0.5831** | 3 602 |
| 0.7 | 0.7795 | 21.78 % | 18.17° | 23.11 % | 0.6079 | 0.5943 | 1 421 |

- 阈值降到 0.2/0.3：召回上升（LR 34 %/31 %），但插入数暴涨，ER 反而恶化。
- 阈值升到 0.7：插入减少、定位误差略降，但漏检显著增加（LR 23 %）。
- **F 在所有阈值下都停在 21–22 %**：瓶颈是类别判别能力，而不是工作点选择；0.5 恰好是 ER/F/SELD 的平衡点。

## 4. 按房间（fold1_room1…9，每房间 40 个片段）

| 房间 | ER | F(macro) | LE | LR | SELD(macro) |
| --- | --- | --- | --- | --- | --- |
| room1 | 0.785 | 30.47 % | 19.45° | 38.98 % | 0.5496 |
| room2 | 0.733 | 34.29 % | 18.24° | 41.63 % | **0.5188**（最好） |
| room3 | 0.756 | 30.97 % | 18.44° | 39.03 % | 0.5396 |
| room4 | 0.742 | 32.62 % | 18.98° | 40.09 % | 0.5299 |
| room5 | 0.798 | 30.50 % | 17.44° | 37.39 % | 0.5540 |
| room6 | 0.793 | 31.18 % | 20.45° | 41.11 % | 0.5460 |
| room7 | 0.835 | 20.97 % | 21.03° | 35.22 % | **0.5974**（最差） |
| room8 | 0.777 | 31.07 % | 16.71° | 36.38 % | 0.5488 |
| room9 | 0.795 | 25.48 % | 24.11° | 40.86 % | 0.5664 |

房间间差异 0.519–0.597（约 0.08），room7 误差率最高（0.835），room9 定位误差最大（24.1°）。没有出现某个房间完全失效的情况。

## 5. 逐类表现（158/170 类在测试集中出现）

- 逐类 F 的中位数仅 **21.1 %**；F > 50 % 的类只有 **7** 个，F < 10 % 的类有 **41** 个。
- 高支持度（GT 段数 ≥ 500）类的中位 F 仅 28.6 %。

| 类别（高支持度中最好） | nref | F | LE | LR |
| --- | --- | --- | --- | --- |
| Drum | 583 | 56.6 % | 17.4° | 69.3 % |
| Writing | 462 | 50.8 % | 18.5° | 64.5 % |
| Wind | 358 | 50.3 % | 18.3° | 55.7 % |
| Packing tape, duct tape | 649 | 50.1 % | 15.7° | 55.9 % |
| Laughter | 715 | 45.9 % | 15.1° | 61.6 % |
| Snare drum | 634 | 42.6 % | 20.0° | 62.5 % |

| 类别（高支持度中最差） | nref | F | LE | LR |
| --- | --- | --- | --- | --- |
| Dishes, pots, and pans | 340 | 1.9 % | 20.5° | 2.6 % |
| Chicken, rooster | 338 | 2.8 % | 16.2° | 3.0 % |
| Subway, metro, underground | 704 | 3.9 % | 35.6° | 7.1 % |
| Electric guitar | 321 | 8.0 % | 35.6° | 16.5 % |
| Bus | 338 | 9.5 % | 18.7° | 10.9 % |
| Tambourine | 687 | 11.0 % | 30.4° | 29.3 % |

规律：打击乐、人声/笑声、部分室内瞬态声（书写、胶带、火焰、咳嗽）较好；交通/城市噪声（地铁、公交、警笛）、部分乐器（电吉他、铃鼓）和餐具类明显偏弱。

## 6. 逐片段分布（micro SELD，360 个片段）

均值 0.583，中位数 0.591，P10 0.494，P90 0.668，最好 0.786，最差 0.362。

最差 5 个片段：`fold1_room5_mix037`(0.362)、`fold1_room1_mix037`(0.384)、`fold1_room2_mix007`(0.387)、`fold1_room8_mix005`(0.396)、`fold1_room8_mix031`(0.398)。

## 7. 结论与分析

1. **召回是主要瓶颈。** FN 32 347 vs TP 9 640：约 77 % 的真值 track 在段内找不到帧对齐的预测 track（LR 27 %），导致 ER 偏高、F 偏低。插入仅 3 602，说明模型"少报"而非"乱报"。
2. **定位处于阈值边缘。** 已匹配 track 的平均角误差 20.16°，且 30 % 的匹配对超过 20°（FP_spatial 4 170）。即使类别报对，方向也常落在阈值之外。
3. **阈值不是解药。** 0.2–0.7 扫描中 F 几乎不变，降低阈值只把删除换成插入；0.5 为当前最优。
4. **类别不平衡/难类明显。** 41 个类 F<10 %，集中在交通、城市噪声和部分乐器；这些类即使有 600+ 段真值也几乎检测不到。
5. **房间泛化尚可**，但 room7 的 ER 偏高，room9 的定位误差偏大，可作为后续 fine-tune 的重点。

## 8. 与 checkpoint 名称中 0.567 的差异

本地 0.600 与文件名 `mACCDOA-HTSAT-0.567` 相差 0.033，原因不是评测代码错误：

- 本地只有 `test360_ov3`（360 片段，最大重叠 3）。论文的 synthetic-test-set 是 **3 060 个 1 分钟片段**（`test1800_ov1` + `test900_ov2` + `test360_ov3`），0.567 应是完整测试集（或训练时验证集）上的分数，二者不可直接比较。
- 同一模型的内部验证在同一子集上给出 0.601，与本文外部评测 0.600 一致。
- 论文 Table I 中同规模模型（CNN14-Conformer + mACCDOA）在完整合成测试集上为 ER 0.805 / F 25.4 % / LE 17.3° / LR 32.1 % / SELD 0.582，量级与本文结果一致。

如需严格对齐 0.567，需要补下 `test1800_ov1`、`test900_ov2` 并在合并后的 3 060 片段上评测。

## 9. 发现的代码问题

`src/evaluate.py` 的逐类打印循环用 `range(args.nb_classes)`（170），但 `classwise_test_scr` 只保留 158 个"出现类"（`SELD_metrics.py` 中按 `LE==180` 删除），因此运行到第 158 行必然 `IndexError`：

```
IndexError: index 158 is out of bounds for axis 0 with size 158
```

建议改为遍历 `len(classwise_test_scr[0])`，或让 `SELD_metrics` 返回类 ID 映射后再打印。本次评测用 `_analysis/eval_analysis.py` 复刻同一套公式并保留类 ID 映射，结果与 `evaluate.py --consolidated_score` 完全一致（0.600 / 22.1 / 20.2 / 27.0）。

## 10. 产出文件

```bash
# 复现本报告（在 /data/PSELDNets 下，pseldnets 环境）
python _analysis/eval_analysis.py \
  --pred_dir logs/multi_accdoa_HTSAT_infer/runs/2026-10-21_21-48-59/submissions
```

- `_analysis/eval_analysis.py` — 评测脚本（修复逐类打印、增加逐房间/逐片段/检测统计）
- `_analysis/results.json` — 本次完整结果（含逐类、逐房间、逐片段）
- `_analysis/classwise.csv` — 158 个出现类的 ER/F/LE/LR/SELD/nref
- `_analysis/thr_0.2/`、`thr_0.3/`、`thr_0.7/` — 阈值扫描结果
- 预测：`logs/multi_accdoa_HTSAT_infer/runs/2026-10-21_21-48-59/submissions`（阈值 0.5）
