# 2026-07-06 Rope Refinement 方案评审与下一步计划

## 目的

这份文档不是最终报告，而是当前阶段的工作判断：我们已经证明 rope signal 有用，但增益还小；下一步不能直接堆训练模块，必须先弄清楚增益小的原因、动作空间的上限，以及当前指标是否稀释了 rope 的真实贡献。

本文把当前证据、主要假设、P0/P1/P2/P3 路线和决策门槛写清楚，方便后续实验按结论推进，而不是反复凭感觉改模型。

## 当前证据

### Clean / Hard Baseline

目前 clean baseline 和 hard split 的评测链路已经基本可信：

- clean baseline 已覆盖 FreiHAND、HO3D v2/v3，以及 HaMeR/WiLoR/AnyHand variants；
- hard split 已覆盖 mask、tip square、finger end 等遮挡方式；
- hard split 会让 baseline 下降，说明构造的遮挡不是无效扰动；
- rope diagnostic 已经观察到一个关键失败模式：hard occlusion 下，闭合手指经常被预测得过于张开。

这些结果记录在 `experience/0020` 到 `experience/0023`，以及 `docs/2026-07-05-rope-diagnostic-reliability.zh.md`。

### 45-dim MLP Refiner 是负结果

实验记录：

- `experience/0025_rope_refiner_full_training_runs.md`
- `experience/0026_rope_refiner_hard_eval_probe.md`

训练 cache 上 pose L1 能下降：

| Backend | Base Pose L1 | Refined Pose L1 |
|---|---:|---:|
| WiLoR | 0.2991 | 0.2313 |
| HaMeR | 0.2923 | 0.2131 |

但 held-out FreiHAND `mask70` hard eval 变差：

| Backend | Metric | Base | Refined | Delta |
|---|---|---:|---:|---:|
| WiLoR | `xyz_procrustes_al_mean3d` | 1.0068 | 1.0975 | +0.0908 |
| WiLoR | `mesh_al_mean3d` | 1.0051 | 1.1043 | +0.0993 |
| HaMeR | `xyz_procrustes_al_mean3d` | 1.0824 | 1.2014 | +0.1191 |
| HaMeR | `mesh_al_mean3d` | 1.0852 | 1.2156 | +0.1304 |

解释：这不是“学习一定不行”，而是当前 MLP 的输出空间太大。rope 每根手指只有一个长度约束，直接输出 45-dim MANO pose delta，等于让网络修改大量 rope 根本观测不到的自由度。训练集上 loss 降低可能来自姿态统计记忆，held-out eval 反而变差。

因此后续训练模块不应该继续输出 45-dim pose delta，而应该先限制到 rope 可解释的低维动作空间，例如 5/15 维 curl 或 flexion alpha。

### Test-Time Rope Optimization 是正结果

实验记录：

- `experience/0027_rope_optimization_probe.md`
- `experience/0028_rope_optimization_cross_split_and_ho3d.md`

当前有效配置：

- `steps=120`
- `lr=2.0`
- `alpha_l2=0.001`
- `max_alpha=0.5`
- action space: `mult5`，每根手指 1 个乘法 curl scalar，共 5 维

它不训练网络，也不用 eval GT joints/verts 做优化，只用 rope label 和 MANO cache 对每个样本做 test-time correction。

结果方向稳定：

| Dataset / Split | Backend | PA Joint Delta | PA Mesh Delta | F@5 Delta |
|---|---|---:|---:|---:|
| FreiHAND `mask70` | WiLoR | -0.0162 | -0.0158 | +0.0044 |
| FreiHAND `mask70` | HaMeR | -0.0143 | -0.0140 | +0.0041 |
| FreiHAND `tip_square80` | WiLoR | -0.0100 | -0.0096 | +0.0047 |
| FreiHAND `tip_square80` | HaMeR | -0.0112 | -0.0108 | +0.0047 |
| FreiHAND `finger_end80` | WiLoR | -0.0156 | -0.0152 | +0.0044 |
| FreiHAND `finger_end80` | HaMeR | -0.0167 | -0.0162 | +0.0043 |
| HO3D v2 `mask70` | WiLoR | -0.0049 | -0.0048 | +0.0013 |
| HO3D v2 `mask70` | HaMeR | -0.0062 | -0.0053 | +0.0019 |

注意：HO3D v2 `mask70` 两行来自 `experience/0028` 的旧优化代码。后续
`experience/0029` 审查发现 WiLoR MANO wrapper 输出的是
OpenPose/FreiHAND 顺序 joints，但旧优化器按 HO3D eval joint 顺序索引
rope chain，导致 4/5 根手指的 rope residual 错配。因此 HO3D 旧数字只能作为
“审查前结果”，报告和决策前必须用修复后的代码重跑；FreiHAND 行不受影响。

结论：rope 是真实有用的几何约束，但当前全手平均指标上的增益仍然小。接下来的重点不是马上训练，而是判断“增益小”到底来自测量口径、动作空间、rope 歧义，还是优化目标没用好。

## 需要区分的两层问题

现在至少有两层问题，不能混在一起：

### 第一层：测量问题

**H1: 全手平均指标稀释了 rope 的真实贡献。**

rope 主要影响被遮挡手指和 fingertip/finger chain，但当前 PA-MPJPE 是全部样本、全部 21 个关节平均。如果 rope 只修了 1-3 根被遮挡手指，全手平均会把真实局部改善稀释很多。

H1 决定的是“怎么讲故事”：如果局部切片增益远大于全手增益，报告里应把主要结果放在被遮挡手指和 high residual 样本上，而不是只讲全手平均。

### 第二层：方法问题

这些假设决定下一步应该怎么改方法：

**H2: 乘法 curl 参数化限制了修正能力。**

当前 `mult5` 形式是：

```text
refined_pose = base_pose + alpha[finger] * base_finger_pose
```

如果某根手指已经被预测得很直，相关 pose 分量较小，乘法缩放很难产生新的弯曲修正。这个解释和“closed fingers 被预测得过于 open”的诊断吻合，但目前仍然是假设，不是结论。

**H3: rope 约束本身存在歧义。**

同一个 wrist-to-tip distance 可能对应不同关节弯曲分布。rope 只能约束长度，不能告诉模型到底是 MCP、PIP 还是 DIP 需要弯，也不能观测侧摆和扭转。如果 rope residual 已经能闭合，但 joint error 仍不改善，就说明 rope-only 信息不够，需要图像证据。

**H4: 当前 rope objective / regularization 没有用好。**

conservative 配置几乎不动，aggressive 配置才有增益。说明目标函数和正则强度很敏感。还需要看 oracle 上限和 residual closure，判断是否只是优化配置没调好。

**H5: 当前 rope input 是 GT 派生的理想信号。**

已有 optimization 数字默认 rope label 完美可靠。真实传感器会有噪声、dropout 和标定误差。这不是当前增益小的原因，但它是结论有效性的威胁。报告前至少需要一个简短的 noise/dropout 消融计划，后续 P1 应补实验。

H1 与 H2-H5 不互斥。H1 解决汇报口径，H2-H5 决定技术路线。

## P0：不训练，先做诊断实验

P0 的目标是把关键假设量化。P0 不应该训练网络，也不应该碰 backbone。

### P0-1 Oracle 天花板实验

目的：判断某个 action space 理论上能修多少。

核心原则：action space 不变，只换 objective。

```text
rope objective:
  minimize rope_loss(MANO(base_pose + delta(alpha)), rope_label)

oracle objective:
  minimize joint_loss(MANO(base_pose + delta(alpha)), gt_joints)
```

这样可以分开两个问题：

- action space 本身有没有能力修正；
- rope objective 有没有把这个能力用出来。

#### Oracle Objective 分层

不要只用 full 21 joints。P0 至少记录三种目标，但主判据放在前两种：

| Objective | 说明 | 用途 |
|---|---|---|
| `oracle_tip` | 只看 5 个 fingertip | 最直接对应 rope 约束，作为主判据 |
| `oracle_chain` | 看每根手指 finger chain joints | 判断整根手指是否能改善 |
| `oracle_full` | 全 21 joints | 只记录，不作为核心结论 |

`oracle_full` 对低维 finger action space 的误导有限，因为 palm/wrist 误差基本是常数；但如果后续引入 `free45`，full objective 会更容易产生解释问题，所以 P0 不把 `free45` 作为主实验。

#### Action Space 候选

| Action Space | Dim | 说明 | P0 是否跑 |
|---|---:|---|---|
| `mult5` | 5 | 当前方法，每根手指一个乘法 scalar | 必跑 |
| `mult15` | 15 | 每根手指 3 个关节分别乘法缩放 | 建议跑 |
| `flex15` | 15 | rope-gradient 定义的加法 flexion 方向 | 必跑 |
| `free45` | 45 | 自由 axis-angle delta + 强 L2 | 暂缓 |

`mult15` 很重要，因为它和 `flex15` 同为 15 维。若 `flex15` 明显优于 `mult15`，才有证据支持“乘法死区/参数化是瓶颈”。如果两者接近，瓶颈可能不是乘法，而是 rope 歧义或优化目标。

#### `flex15` 的精确定义

`flex15` 不能含糊地说“屈曲方向”。第一版定义为 rope-gradient flexion：

```text
for each sample, finger, joint:
  g_j = d(rope_distance_finger) / d(axis_angle_j) at base pose
  direction_j = normalize(g_j)
  delta_pose_j = alpha_j * direction_j
```

注意：

- 这是 rope-gradient 定义的 flexion，不是严格解剖学 flexion；
- 方向在 base pose 处计算并冻结，优化过程中只更新 alpha；
- 这样 action space 是清楚的线性子空间，便于和 `mult15` 公平比较；
- 可以额外记录该方向与 MANO 固定局部轴的夹角，作为 sanity check，不作为 P0 主指标。

### P0-2 Rope Residual Closure

目的：判断 optimization 是否真的满足 rope 约束。

每次优化后都应输出：

- base rope residual；
- optimized rope residual；
- residual reduction ratio；
- per-finger residual；
- residual 与 joint improvement 的相关性。

典型解读：

- residual 降不下去：action space 连 rope 长度都满足不了，H2/H4 更可疑；
- residual 闭合但 joint 不改善：rope-only 约束有歧义，H3 更可疑；
- residual 和 joint improvement 正相关：rope 信号确实定位了错误区域。

### P0-3 被遮挡手指切片指标

目的：判断全手平均是否稀释了 rope 的真实效果。

应统计：

- all joints PA-MPJPE delta；
- fingertip-only error delta；
- occluded-finger chain error delta；
- high rope residual bucket 的 delta；
- per-finger delta；
- base rope residual 与 improvement 的相关性。

hard split 生成时有 `hard_manifest.jsonl`，其中应优先使用遮挡记录来判断哪些 finger 被影响；如果某些 split 记录不完整，则用 rope residual bucket 作为替代切片。

## 决策门槛

这些门槛不是最终论文标准，而是下一步工程决策用的初值。跑完 P0 后可以调整。

| 判断 | 门槛 | 结论 |
|---|---|---|
| H1: 指标稀释成立 | slice gain >= 5x all-joint gain | 汇报重点转向 occluded finger / high residual |
| H2: 参数化瓶颈成立 | oracle(`flex15`) >= 2x oracle(`mult15`) | 优先实现 `flex15 + gating` |
| H3: rope 歧义成立 | residual closure > 80%，但 joint gain < oracle gain / 3 | 后续需要图像特征 head |
| H4: objective 没用好 | oracle gain >= 3x rope gain | 优先改 rope objective / gating / regularization |
| H5: 传感器鲁棒性风险 | noise/dropout 下增益明显消失 | 后续训练必须加 noise/dropout augmentation |

这些规则可以同时触发。例如 H1 和 H2 同时成立时，含义是：汇报上讲局部改善，技术上继续改参数化。

## P1：改进 Test-Time Optimization

P1 只有在 P0 说明 action space 或 objective 还有空间时才做。

优先事项：

1. 实现 `flex15` rope-gradient additive correction；
2. 加 residual gating，只优化 residual 明显偏大的 finger；
3. 把 rope residual summary 作为所有 optimization 输出的标准字段；
4. 补 rope noise / dropout 消融；
5. 先跑 FreiHAND `mask70` 和 `finger_end80` 的 WiLoR，再扩到 HaMeR 和 HO3D v2 `mask70`。

P1 的目标不是最终系统，而是得到更强、更可信的 test-time teacher。

## P2：蒸馏成低维可训练模块

如果 P1 的 teacher 稳定有效，再训练低维模块。不要回到 45-dim MLP。

第一版建议：

```text
input:
  base_hand_pose        45
  base_rope_norm         5
  input_rope_norm        5
  rope_residual          5
  rope_valid             5

output:
  flexion alpha         15
```

训练目标：

- imitation loss：模仿 P1 optimization 得到的 alpha；
- rope loss：refined pose 的 rope 长度接近 input rope；
- delta regularization：限制修正幅度；
- validation / early stopping；
- rope noise augmentation；
- shuffled-rope control：打乱 rope label 后增益应消失。

shuffled-rope control 很关键。它能证明模块学到的是 rope-conditioned correction，而不是数据集姿态先验。

## P3：加入图像证据的 Rope-Conditioned Head

更长期方向是冻结 backbone，只训练一个 rope-conditioned refinement head。它解决的是 H3：rope 长度本身有歧义，需要图像特征帮助判断弯在哪一节、是否可见、是否有侧摆/扭转。

最小版本先不做 transformer：

```text
pooled image feature + base pose + rope residual + rope valid
  -> MLP
  -> 15-dim flexion alpha
```

如果有效，再考虑每根手指一个 query 的 cross-attention head，或者参考 WiLoR RefineNet，从当前预测投影位置采样局部图像特征。

这一步应该放在 P0/P1/P2 之后。现在直接改 HaMeR/WiLoR 内部训练风险太高，失败后无法判断是 rope 不行、训练不行，还是 backbone 被破坏。

## 近期执行顺序

1. 扩展 `apply_rope_refinement.py`，支持 `--objective rope|oracle_tip|oracle_chain` 和 `--action-space mult5|mult15|flex15`。
2. 给每次 optimization 输出 rope residual summary。
3. 写 slice scoring 脚本，输出 occluded finger / fingertip / residual bucket 指标。
4. 先跑 WiLoR × FreiHAND `mask70` 和 `finger_end80`。
5. 根据 P0 决策门槛判断是否进入 P1 的 `flex15 + gating`。
6. P1 稳定后再考虑 P2 蒸馏；P3 只作为报告后计划。

## 当前注意事项

- `apply_rope_refinement.py` 的默认超参偏保守；复现实验必须显式传入有效配置，或把有效配置写进 tracked config。
- `build_freihand_refiner_cache.py` 的 `--base-hand-pose-source target` 有 GT 泄漏风险；后续训练默认应改成必须显式选择。
- HO3D v2 当前没有 train split，已跑的 HO3D rope optimization 是 eval/test-time correction，不是训练。
- 报告里跨表比较时优先使用同一 scoring protocol，或使用 mesh/F 指标，避免早期协议变动带来的小偏差。
- 当前 rope input 来自 GT joints，是理想传感器近似；报告中必须明确这一点，并把 noise/dropout 消融列为后续必要验证。
