# Rope Diagnostic 可靠性验证

日期：2026-07-05

## 问题

在正式把 rope signal 加入训练前，先验证它在现有 baseline prediction 上是否是一个有意义的诊断信号。

这里的 rope diagnostic 比较的是每个样本五根手指的归一化 wrist-to-tip 距离：

- `gt_rope_norm[5]`：由 GT joints 计算得到。
- `pred_rope_norm[5]`：由 baseline 导出的 `pred.json` joints 再计算一遍得到。
- `abs(pred_rope_norm - gt_rope_norm)`：作为 rope error。

归一化时，prediction 使用的是 GT finger-chain length。这样做的目的是避免模型预测错手部尺度后，把 rope error 自己抵消掉。

## Rope label 可视化

下面两张图展示了 label 生成阶段的可视化结果。左侧是原图和 GT hand chain overlay，右侧是五根手指的 `rope_norm` 柱状图。柱子越接近 1，表示 wrist-to-tip 距离越接近该手指链长，也就是手指越伸展；越接近 0，表示手指越收拢。

![FreiHAND rope label overlay](assets/rope_diagnostics/label_freihand_00000000.png)

这张 FreiHAND 样本中，大部分手指都接近伸展状态，所以五根手指的 `rope_norm` 都接近 1。

![HO3D v2 rope label overlay](assets/rope_diagnostics/label_ho3d_v2_SM1_0000.png)

这张 HO3D v2 样本中，不同手指的伸展程度差异更明显，例如 middle/ring 的 `rope_norm` 明显低于 index，说明 rope label 能表达不同手指的 extension state，而不是只记录一个整体手形态。

## 输出位置

远程输出根目录：

`/data/wentao/ropetrack/runs/rope_phase12_20260705_031056`

远程诊断结果：

`/data/wentao/ropetrack/runs/rope_phase12_20260705_031056/diagnostics`

本地已拷贝结果：

`.local_checks/rope_phase12_20260705_031056/diagnostics`

主要生成文件：

- `run_summary.tsv`
- `hard_clean_delta.tsv`
- `per_finger.tsv`
- `gt_bin_summary.tsv`
- `worst_cases.tsv`
- `figures/hard_clean_delta.png`
- `figures/scatter_*.png`
- `figures/worst_*.png`

## 主要结果

rope diagnostic 对 hard occlusion 是敏感的。最明显的退化出现在 FreiHAND hard splits：

| Run | 相比 clean 的 rope_norm_mae 增量 |
|---|---:|
| FreiHAND finger_end80 HaMeR | +0.068127 |
| FreiHAND finger_end80 WiLoR | +0.056781 |
| FreiHAND mask70 HaMeR | +0.053044 |
| FreiHAND mask70 WiLoR | +0.051303 |
| HO3D v2 mask70 WiLoR | +0.023835 |
| FreiHAND tip_square80 HaMeR | +0.023636 |
| HO3D v2 mask70 HaMeR | +0.014994 |
| FreiHAND tip_square80 WiLoR | +0.014747 |

HO3D v2 的 `tip_square80` 基本没有明显变化：

- HaMeR：`+0.002432`
- WiLoR：`-0.001981`

![Hard-clean rope diagnostic delta](assets/rope_diagnostics/hard_clean_delta.png)

这张图把每个 hard split 相对 clean baseline 的 `rope_norm_mae` 增量画出来。可以看到 FreiHAND 的 `finger_end80` 和 `mask70` 排在最前面，说明这两类遮挡对 finger extension-state 的破坏最明显。

这和前面对 hard split 的观察一致：大面积 mask 和 finger-end 遮挡会造成更明显的手部姿态失败，而小范围 fingertip square 的影响更弱。

## 失败模式

GT-bin 分析说明，rope error 不是纯随机噪声，而是在暴露具体失败模式。

在 FreiHAND 上，closed fingers 在 hard split 下明显更难：

| Run/bin | Count | MAE | Bias |
|---|---:|---:|---:|
| FreiHAND mask70 HaMeR, closed | 2728 | 0.316276 | +0.293416 |
| FreiHAND finger_end80 HaMeR, closed | 2728 | 0.275939 | +0.250788 |
| FreiHAND finger_end80 WiLoR, mid | 2792 | 0.218905 | +0.047460 |

这里的 positive bias 表示：模型预测出来的手指比 GT 更加 open，也就是手指实际更弯曲或更收拢时，baseline 倾向于预测成更伸展的状态。

这点很关键，因为它说明 rope diagnostic 不只是重复 MPJPE 的结论，而是在指出一种更具体的错误：遮挡后模型对手指伸展/收缩状态的判断变差，尤其容易把 closed finger 预测得过于 open。

HO3D v2 上 closed-bin 的误差也很高，但 closed-bin 只有 55 个 finger instances，样本数太少。因此这部分只能作为 caveat，不能作为主结论。

![FreiHAND hard finger_end80 HaMeR scatter](assets/rope_diagnostics/scatter_freihand_finger_end80_hamer.png)

这张 scatter 的横轴是 GT `rope_norm`，纵轴是 prediction `rope_norm`，黑色斜线是理想情况。点离黑线越远，说明该手指的伸展程度预测越错。FreiHAND `finger_end80` 下散点明显变散，尤其在 GT 接近 closed/mid 的区域，prediction 经常被推到更 open 的位置。

![HO3D v2 hard mask70 WiLoR scatter](assets/rope_diagnostics/scatter_ho3d_v2_mask70_wilor.png)

HO3D v2 `mask70` 也能看到偏离黑线的情况，但样本主要集中在 GT `rope_norm` 较高的位置。也就是说，HO3D v2 当前 eval split 里更多是 open 或 near-open 的手指状态，closed 状态证据不足。

## 分手指结果

FreiHAND hard splits 中，ring 和 pinky 的 rope error 最大：

| Run | Finger | MAE | Bias |
|---|---|---:|---:|
| FreiHAND finger_end80 HaMeR | pinky | 0.198757 | +0.031495 |
| FreiHAND mask70 HaMeR | pinky | 0.192398 | +0.064278 |
| FreiHAND finger_end80 HaMeR | ring | 0.187795 | +0.027111 |
| FreiHAND finger_end80 WiLoR | pinky | 0.177806 | +0.011619 |
| FreiHAND mask70 WiLoR | pinky | 0.175908 | -0.002406 |
| HO3D v2 mask70 WiLoR | ring | 0.167085 | -0.102019 |

这说明 rope signal 对不同手指的有效性并不完全一致。后续如果进入训练，可以先保留五根手指的 `rope_norm[5]`，但评估时应该单独监控 per-finger loss，而不是只看一个整体平均值。

## 解释

这个诊断结果支持继续研究 rope data，但它应该被理解为一个低维约束：

- 它能捕捉手指伸展/收缩状态的错误。
- 它不能完整约束 3D pose、手掌朝向、tip 横向位置、关节角合理性。
- 它应该作为 MPJPE/PCK 和可视化之外的补充信号，而不是替代标准 pose metric。

目前最稳妥的下一步是：

1. 继续把 `rope_norm[5]` 作为结构化 JSONL label 保存。
2. 用 rope diagnostic 筛选 hard cases，并比较 clean 和 hard 的行为差异。
3. 如果后续进入训练，优先考虑 auxiliary loss 或小模块 conditioning，同时报告 per-finger 和 per-bin 结果，不只报告一个整体 MAE。

## 可以汇报的结论

在没有训练的情况下，我们验证了 proposed rope signal 不是任意构造的额外标签。它能够检测 baseline prediction 从 clean 到 hard occlusion 的退化，尤其是在 FreiHAND 的 mask 和 finger-end 遮挡上最明显。

更重要的是，rope diagnostic 暴露了一类具体失败模式：遮挡后模型容易把 closed/mid finger 预测得过于 open。这说明 rope signal 可以提供一种普通 pose metric 之外的 finger extension-state constraint，值得作为后续训练设计的候选信号继续研究。
