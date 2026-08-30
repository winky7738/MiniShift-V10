# MiniShift-Simple3D 二阶段精度优化规范

> 用途：直接交给 AI 代码编辑器实施。本文是上次优化方案的修正版，目标是在保留无损加速的同时，恢复并提高 P-ROC，重点改善 Screw O-ROC 和 Toothbrush P-ROC。

```yaml
spec_version: 2.0
project: MiniShift-Simple3D
hardware_target:
  os_recommended: Ubuntu_or_WSL2
  gpu: NVIDIA_RTX_4070_Laptop
  assumed_vram_gb: 8
primary_objective:
  metric: "(mean_O_ROC + mean_P_ROC) / 2"
  setting: ALL
baseline_all_12:
  mean_o_roc: 0.673
  mean_p_roc: 0.660
  final_score: 0.6665
resource_constraints:
  competition_max_vram_gb: 48
  competition_max_total_hours: 24
  laptop_soft_vram_limit_gb: 7.2
  preferred_runtime_multiplier_vs_current: [1.5, 1.8]
  hard_total_runtime_target_hours: 22
prohibited:
  - modify_metric_formula
  - modify_ground_truth
  - use_test_label_or_filename_in_model_logic
  - select_parameters_per_test_sample
```

## 1. 本轮日志结论

### 1.1 原始基线与上次优化的前四类对比

| 类别 | 原O-ROC | 新O-ROC | ΔO | 原P-ROC | 新P-ROC | ΔP |
|---|---:|---:|---:|---:|---:|---:|
| Capsule | 0.912 | 0.934 | +0.022 | 0.812 | 0.722 | **-0.090** |
| Cube | 0.695 | 0.685 | -0.010 | 0.653 | 0.654 | +0.001 |
| Spring_Pad | 0.729 | 0.757 | +0.028 | 0.629 | 0.613 | -0.016 |
| Screw | 0.224 | 0.233 | +0.009 | 0.651 | 0.649 | -0.002 |
| 四类均值 | 0.640 | 0.652 | +0.012 | 0.686 | 0.660 | **-0.026** |

四类两指标平均由约 `0.6631` 降至 `0.6560`，下降约 `0.0071`。上次方案没有达到保留条件。

### 1.2 不能从当前日志确定单项因果

上次同时启用了：

```text
修复重复尺度
geom4d(weight=0.2)
prototype density normalization
Top-ratio目标聚合
完全关闭后平滑
类别配置
```

因此不能断言某一项单独有效。新版必须逐项消融，禁止再次一次性全开。

### 1.3 日志缺陷

虽然 `use_category_profiles=True`，日志只打印了全局参数，没有打印每个类别解析后的有效参数。必须新增：

```text
[EffectiveConfig][capsule] ...
[EffectiveConfig][cube] ...
```

否则无法判断类别配置是否真正生效。

## 2. 新点云的几何诊断

所有文件均为500,000个XYZ点。由于上传时丢失原类别目录，以下按几何形态描述；AI编辑器不得仅凭上传后的重名后缀把它们硬编码为某个类别。

| 文件 | 几何形态 | PCA有效尺度（约） | 40-NN半径中位数 | 曲率特征与风险 |
|---|---|---|---:|---|
| `54_1_scratch.txt` | 光滑长圆柱/胶囊体 | 21.51×7.60×7.58 | 0.113 | 曲率中位数约4.2e-5，表面很平滑；完全不平滑会把采样噪声当缺陷 |
| `80_1_scratch(1).txt` | 极薄环形垫片 | 14.87×14.64×1.00 | 0.079 | 内外边界占比高；几何特征和密度归一化容易放大正常边缘 |
| `80_1_scratch(2).txt` | 圆角厚盘/纽扣形 | 11.52×11.42×5.26 | 0.094 | 平面、侧壁、圆角三种正常区域，需保留适度平滑 |
| `100_1_scratch(1).txt` | 大头部+窄柄多区域零件 | 34.12×23.33×23.41 | 0.255 | 头柄交界和内部开口是天然高响应区，单一全局校准不稳 |
| `108_1_scratch.txt` | 六角螺母/带孔零件 | 14.27×13.04×6.31 | 0.117 | 外角、内孔和端面同时存在，局部曲率差异大 |
| 既有 `80_1_scratch.txt` | 螺钉 | 16.10×13.22×13.13 | 0.130 | 少量正常凹槽/头杆交界会产生极高分，干扰O-ROC |
| 既有 `100_1_scratch.txt` | 牙刷 | 175.70×15.94×11.26 | 0.339 | 平滑手柄与高曲率刷毛并存，细划痕易被平滑抹除 |

共同结论：MiniShift类别之间、同一物体不同区域之间的几何差异非常大。`geom4d=0.2 + 全局密度归一化 + 完全无平滑`不适合作为全类别统一默认值。

## 3. 本轮总设计：彻底分离O分支和P分支

当前最重要的架构修改：目标级分数和点级分数必须从同一原始中心距离出发，但之后独立处理。

```text
center_raw_distance
├── P branch: raw + light smoothing -> interpolate -> point_score_map
└── O branch: normal-tail calibration + spatial coherence -> object_score
```

禁止：把原型密度归一化后的分数图直接同时用于O-ROC和P-ROC。

理由：

- O-ROC需要降低正常边缘极值对整物体分数的干扰；
- P-ROC需要保留每个点之间的真实排序；
- 位置相关或原型相关的除法会改变点之间的排序，可能解释Capsule P-ROC下降0.09。

## 4. 第一阶段：恢复一个“快速但数值等价”的基线

实验ID：`R0_SAFE_BASELINE`

只保留不会改变理论结果的速度修改：

1. 插值全排序改最近3点 `torch.topk`；
2. 插值与 `cdist` 分块；
3. 特征缓存；
4. 关闭可视化文件写入；
5. 修复类别循环和日志；
6. 不启用 geom4d、密度归一化、新目标聚合和无平滑。

### 4.1 中尺度重复的“精确等价压缩”

旧代码特征是：

```text
[F40, F80, F80, F120]
```

直接删除第二个F80会改变欧氏距离中的尺度权重。要既减少维数又复现旧距离，应改为：

```text
[F40, sqrt(2)*F80, F120]
```

数学上：

```text
||ΔF80||² + ||ΔF80||² = ||sqrt(2)*ΔF80||²
```

实现：

```python
scale_weights = [1.0, math.sqrt(2.0), 1.0]
fpfh = torch.cat([
    scale_weights[0] * f40,
    scale_weights[1] * f80,
    scale_weights[2] * f120,
], dim=-1)
```

要求：

- 不额外进行旧基线没有的逐尺度L2归一化；
- 单元测试比较旧132维距离和新99维距离，相对误差 `<1e-5`；
- 参数命名：`--scale_fusion legacy_equivalent`；
- 另保留 `legacy_duplicate` 和 `equal_3scale` 供消融，但默认先用 `legacy_equivalent`。

### 4.2 R0验收

R0应尽量恢复原始前四类表现：

```yaml
R0_gate:
  capsule_p_roc: ">= 0.79"
  first4_mean_p_roc: ">= 0.675"
  score_difference_vs_original_baseline: "absolute <= 0.005 preferred"
```

如果R0无法恢复，不得继续叠加新方法，先检查数据版本、随机种子和数值等价性。

## 5. 第二阶段：P-ROC双分支细节保持

实验ID：`P1_DUAL_MAP`

### 5.1 不再使用 `post_smooth_mode=none` 作为全局默认值

在4096/6144个中心上构造KNN图，只对中心分数做一次轻量平滑，然后再插值到500k点。不要再次FPS降到1024。

```python
raw_center = center_raw_distance
smooth_center = knn_weighted_mean(raw_center, center_xyz, k=smooth_k)

# 使用训练正常分数统计得到的同一类别尺度做归一化，不能每个测试样本min-max
raw_norm = raw_center / (train_raw_q99 + eps)
smooth_norm = smooth_center / (train_smooth_q99 + eps)
fused_center = raw_weight * raw_norm + smooth_weight * smooth_norm
point_score_map = interpolate(fused_center)
```

推荐：

```yaml
global_start:
  p_map_mode: dual_center
  smooth_k: 8
  raw_weight: 0.65
  smooth_weight: 0.35
```

候选只测试三组：

```yaml
p_fusion_candidates:
  - [0.80, 0.20]
  - [0.65, 0.35]
  - [0.50, 0.50]
```

不允许大网格搜索。优先用已有缓存的中心分数完成融合消融，无需重新提取FPFH。

### 5.2 几何分组的初始配置

```yaml
geometry_profiles:
  smooth_body:       # capsule, plastic_cylinder
    raw_weight: 0.50
    smooth_weight: 0.50
    smooth_k: 12
  planar_or_ring:    # spring_pad, flat_pad, button_cell
    raw_weight: 0.65
    smooth_weight: 0.35
    smooth_k: 8
  sharp_multi_part:  # screw, nut, light
    raw_weight: 0.70
    smooth_weight: 0.30
    smooth_k: 8
  elongated_mixed:   # toothbrush
    raw_weight: 0.85
    smooth_weight: 0.15
    smooth_k: 6
  freeform:          # piggy等
    raw_weight: 0.65
    smooth_weight: 0.35
    smooth_k: 8
```

所有类别映射集中写在一个YAML文件中；算法函数内禁止出现类别名字符串。

### 5.3 geom4d降级为可选辅助分支

全局默认：

```yaml
use_geom4d_for_p: false
geom_weight: 0.0
```

只有 Toothbrush 或经消融证明确有收益的类别才测试：

```yaml
use_geom4d_for_p: true
geom_weight_candidates: [0.03, 0.05, 0.10]
```

不得再直接使用0.20作为全局默认值。更推荐先生成独立 `geom_score_map`，用小权重与FPFH分数后融合，而不是把几何向量直接拼入所有类别的特征距离。

## 6. 第三阶段：Screw O-ROC正常尾部分布校准

实验ID：`O1_NORMAL_TAIL`

上次Top-500与密度归一化只把Screw O-ROC从0.224提高到0.233，说明单纯增大top-k不是主要解法。

### 6.1 正常训练样本留出评分

为训练正常特征保留 `sample_id`。采用3折正常样本校准：

1. 将训练正常样本分为3折；
2. 每次用2折构建临时原型库；
3. 对剩余1折计算中心异常分数；
4. 合并得到“正常情况下的真实中心分数分布”；
5. FPFH特征必须缓存，3折过程不能重新读取500k点并提取FPFH。

记录：

```text
normal_q95, normal_q99, normal_q995, normal_q999
normal_tail_median, normal_tail_MAD
```

### 6.2 尾部超额分数

```python
threshold = normal_q995
excess = torch.relu(test_center_score - threshold)
excess = excess / (normal_tail_MAD + 1e-6)
tail_mass = excess.mean()
tail_top = topk_mean(excess, ratio=0.01)
```

这里在中心分数上计算，4096中心的1%约41个中心，不使用500k点中固定80点。

### 6.3 空间一致性分数

正常边缘误报往往零散；划痕/缺陷通常形成连续区域。复用中心KNN图：

```python
local_excess = knn_mean(excess, center_graph, k=12)
coherence = topk_mean(local_excess, ratio=0.01)
```

最终O分数：

```python
object_score = (
    0.35 * tail_mass +
    0.35 * tail_top +
    0.30 * coherence
)
```

三个权重先固定，不使用测试标签搜索。可使用少量训练正常点云合成局部划痕做一次小规模验证，但不能读取测试GT选择权重。

### 6.4 原型密度归一化只作为O分支候选

```yaml
prototype_density_norm_for_p: false
prototype_density_norm_for_o: [false, true]
```

只做一次A/B比较。如果Screw O-ROC增益小于0.02或其他类别O均值下降，则关闭。

## 7. 第四阶段：允许增加50%-80%耗时的输入精度

实验ID：`I1_RESOLUTION`

不要在R0/P1/O1未通过前增加采样点。先证明评分方法有效，再投入计算资源。

### 7.1 RTX 4070 Laptop精度平衡配置

假设显存8GB：

```yaml
precision_balanced_4070_laptop:
  num_group: 6144
  group_size: 128
  max_nn: 40
  normal_max_nn: 20
  num_MSND: 2
  use_MSND: true
  use_LFSA: true
  scale_fusion: legacy_equivalent
  interp_chunk_size: 4096
  query_chunk: 512
  bank_chunk: 4096
  p_map_mode: dual_center
  smooth_k: 8
  use_geom4d_for_p: false
  prototype_density_norm_for_p: false
  normal_calibration_folds: 3
  vis_save: true
  cache_features: true
```

理由：

- 4096→6144中心使表面采样分辨率提高50%，与允许的耗时增长匹配；
- `normal_max_nn`由原代码10提高到20，降低FPFH法向噪声，尤其有利于光滑胶囊和圆柱；
- 保留`group_size=128`作为论文精度设置，不再同时改变过多参数；
- 6144中心时将插值块降到4096，避免8GB显存溢出；
- `query_chunk=512, bank_chunk=4096`用时间换显存稳定性；
- 不要随意关闭`vis_save`！！！！默认就是开启的

### 7.2 类别精度配置

```yaml
categories:
  capsule:
    num_group: 6144
    group_size: 128
    max_nn: 40
    normal_max_nn: 24
  cube:
    num_group: 6144
    group_size: 128
    max_nn: 40
  spring_pad:
    num_group: 6144
    group_size: 96
    max_nn: 40
  screw:
    num_group: 6144
    group_size: 128
    max_nn: 40
    object_score_mode: normal_tail_coherence
  toothbrush:
    num_group: 8192
    group_size: 64
    max_nn: 30
    normal_max_nn: 20
    interp_chunk_size: 2048
    raw_weight: 0.85
    smooth_weight: 0.15
```

说明：Screw O-ROC不是采样分辨率主导，不建议单独把Screw提高到8192；Toothbrush细划痕的P-ROC更可能受采样分辨率影响，因此只给Toothbrush 8192中心。

### 7.3 精度上限配置

只有平衡配置确实提升最终平均分时才启用：

```yaml
precision_max_4070_laptop:
  default_num_group: 8192
  default_group_size: 96
  default_max_nn: 40
  interp_chunk_size: 2048
  query_chunk: 384
  bank_chunk: 4096
  normal_max_nn: 24
```

如果峰值显存超过7.2GB，按顺序调整：

```text
interp_chunk 2048 -> 1024
query_chunk 384 -> 256
bank_chunk 4096 -> 2048
最后才降低num_group
```

减小chunk只影响速度，不应改变精确计算结果。

## 8. 时间目标的解释与控制
用户自己把握时间，不要做任何控制时间的代码级改动


### 8.1 时间预算控制器

用户自己把握，不要做任何控制时间的代码级改动

## 9. 推荐运行命令

AI编辑器应增加YAML配置加载，推荐命令：

```bash
python -u main.py \
  --dataset minishift \
  --level ALL \
  --config configs/minishift_precision_balanced_4070.yaml \
  --expname MiniShift_precision_v2 \
  2>&1 | tee logs/MiniShift_precision_v2_console.log
```

如果仍保留CLI参数，等价起始命令：

```bash
python -u main.py \
  --dataset minishift \
  --level ALL \
  --num_group 6144 \
  --group_size 128 \
  --max_nn 40 \
  --normal_max_nn 20 \
  --use_MSND \
  --use_LFSA \
  --num_MSND 2 \
  --scale_fusion legacy_equivalent \
  --interp_chunk_size 4096 \
  --query_chunk 512 \
  --bank_chunk 4096 \
  --p_map_mode dual_center \
  --normal_calibration_folds 3 \
  --expname MiniShift_precision_v2
```

不得写：

```bash
--vis_save False
```

旧版`type=bool`会把非空字符串解析为True。应改成 `action='store_true'`，精度测试时不传 `--vis_save`。

## 10. 必须实施的缓存

缓存分层：

```text
L1: 每个样本的F40/F80/F120全点特征或中心聚合特征
L2: centers、center_idx、center_features、center_normals
L3: 原型库、原型局部统计、正常留出分数分位数
L4: raw_center_scores，用于离线测试P融合和O聚合
```

缓存键至少包含：

```text
dataset_version, category, split, sample_path_hash,
max_nn, normal_max_nn, num_group, group_size,
scale_fusion, code_version, random_seed
```

修改P融合权重、O聚合或平滑参数时，不得重新提取FPFH。

## 11. 消融执行顺序

| 顺序 | 实验ID | 相对R0新增内容 | 是否重提FPFH | 保留条件 |
|---:|---|---|---|---|
| 0 | R0 | 数值等价快速基线 | 是 | 恢复原基线分数 |
| 1 | P1a | dual P图 0.65/0.35 | 否 | 前四类P均值提高 |
| 2 | P1b | 三个P融合候选 | 否 | Capsule P不再大跌 |
| 3 | O1 | 3折正常尾部+空间一致性 | 否，使用缓存 | Screw O至少提高0.03 |
| 4 | O1d | O分支密度归一化A/B | 否 | 全类O均值增加才保留 |
| 5 | I1 | 4096→6144、法向10→20 | 是 | 最终分提高且时间≤1.8x |
| 6 | T1 | Toothbrush 8192/64/30 | 是，仅牙刷 | Toothbrush P提高≥0.03 |
| 7 | G1 | Toothbrush geom 0.03/0.05 | 可复用部分缓存 | P提高才保留 |

禁止跨过R0直接跑I1，也禁止第一次就同时启用O1、I1和G1。

## 12. 验收标准

### 12.1 第一阶段前四类闸门

```yaml
first4_acceptance:
  combined_score_previous_optimized: 0.6560
  combined_score_original_baseline: 0.6631
  required_combined_score: "> 0.6631"
  capsule_p_roc: ">= 0.79 preferred"
  screw_o_roc: ">= 0.26 first_gate; >=0.30 target"
  mean_p_roc_drop_vs_original: "<= 0.005"
```

### 12.2 完整12类闸门

```yaml
all12_acceptance:
  baseline_final_score: 0.6665
  minimum_keep_score: "> 0.6665"
  preferred_score: ">= 0.675"
  stretch_score: ">= 0.680"
  runtime_multiplier: "<= 1.8"
  laptop_peak_vram_gb: "<= 7.2 preferred"
  official_peak_vram_gb: "< 48"
  total_runtime_hours: "< 24; target <=22"
  any_single_metric_drop_guard: "no unexplained drop >0.03"
```

## 13. 必须新增的诊断输出

每个测试样本输出到CSV/JSONL：

```text
category, sample_id, label,
raw_top80, raw_top500,
normal_q995_excess_mass,
normal_q995_excess_top,
coherence_score,
final_object_score,
raw_score_q99, raw_score_q999,
fit_seconds, test_seconds, peak_vram_gb
```

注意：这些字段可以在评价后写日志，但模型计算过程中不能根据label选择分支。

每个类别自动生成正常/异常对象分数分布图，用于判断 Screw O-ROC低的真实原因：

- 如果异常分数整体低于正常分数：检查特征符号、校准方向和正常边缘干扰；
- 如果两者重叠但异常尾部更宽：使用尾部超额质量；
- 如果异常只有局部连续高分：提高空间一致性权重；
- 如果所有统计均无区分：再考虑类别专用几何描述子，而不是继续调top-k。

## 14. 自动测试

AI编辑器必须实现：

1. `test_legacy_equivalent_scale_distance.py`：132维旧距离与99维加权距离等价；
2. `test_topk_interpolation_equivalence.py`：全排序与topk最近3点插值误差 `<1e-6`；
3. `test_o_p_branch_separation.py`：启用O校准不改变P分数图；
4. `test_no_label_leakage.py`：label/mask不得进入参数选择或分数计算；
5. `test_effective_category_config.py`：逐类有效配置正确覆盖并写日志；
6. `test_cache_key_integrity.py`：输入参数变化时不会错误复用缓存；
7. `test_runtime_budget.py`：预测总时长超过22小时时触发降级策略；
8. `compare_experiments.py`：输出12类ΔO、ΔP、最终分差、耗时比和显存比。

## 15. 不再推荐的做法

- 不再全类别默认 `post_smooth_mode=none`；
- 不再把 `prototype_density_norm=True` 直接用于P分数图；
- 不再全类别默认 `geom_weight=0.2`；
- 不再认为“删除重复F80”必然提高分数，应使用数学等价的`sqrt(2)`权重；
- 不再仅靠Top-80改Top-500解决Screw O-ROC；
- 不在没有消融证据时全类别提升到8192；
- 不打开`vis_save`进行全量数值实验；
- 不使用测试GT、文件名或测试标签选择参数。

## 16. 给AI代码编辑器的最终执行指令

```text
先建立R0数值等价快速基线，确保恢复原始P-ROC；
然后分离O/P评分支路；
P支路实现中心级raw+smooth双分支，不再二次降采样到1024；
O支路实现3折正常留出尾部分布校准和中心图空间一致性；
原型密度归一化只允许进入O支路；
geom4d默认关闭，只允许小权重类别消融；
通过R0/P1/O1闸门后，才把默认num_group提升到6144、normal_max_nn提升到20；
Toothbrush可单独使用8192/64/30；
所有实验输出逐类有效参数、逐样本评分统计、时间和显存；
最终只保留完整12类综合平均分提高且总时长小于24小时的改动。
```

