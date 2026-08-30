# MiniShift-Simple3D 优化改造规范（供 AI 代码编辑器执行）

```yaml
project: MiniShift-Simple3D
task: 在不修改官方评价逻辑的前提下，提高 ALL 设置下的 O-ROC 与 P-ROC
baseline:
  mean_o_roc: 0.673
  mean_p_roc: 0.660
  final_score: 0.6665
priority_weaknesses:
  screw_o_roc: 0.224
  toothbrush_p_roc: 0.459
hard_constraints:
  dataset_structure_unchanged: true
  evaluation_code_unchanged: true
  random_seed_fixed: true
  max_vram_gb: 48
  max_total_runtime_hours: 24
  setting: ALL
optimization_budget:
  preferred_runtime_ratio: "<= 1.15x baseline"
  preferred_vram_ratio: "<= 1.20x baseline and always < 48GB"
```

## 1. 必须遵守的边界

1. 不修改 `roc_auc_score`、测试标签、GT 掩码、类别平均方式和最终评分公式。
2. 不根据测试文件名、测试标签或 GT 掩码决定模型输出；测试标签只用于最终评价。
3. 官方最终分数为：

   ```text
   final_score = (mean_O_ROC_over_12_classes + mean_P_ROC_over_12_classes) / 2
   ```

4. 所有实验必须使用 `--level ALL`，固定随机种子并记录运行时长、峰值显存和逐类别结果。
5. 每项修改必须能通过配置开关关闭，保留 `baseline` 回退路径。

## 2. 已观察到的样本特征

### 2.1 `80_1_scratch.txt`：螺钉类

- 点数：500,000；只有 XYZ 三列。
- PCA 主轴有效长度约 16.10，两个横向尺度约 13.22、13.13，属于紧凑型旋转体。
- 几何包含杆体、头部、头杆交界和头部凹槽；这些结构本身会产生强 FPFH/曲率响应。
- 40-NN 物理半径中位数约 0.130，99% 分位约 0.141，点密度整体均匀。
- 局部曲率中位数约 `8.4e-5`，99% 分位约 `3.59e-3`，但 99.9% 分位跃升到约 `9.12e-2`，说明少量正常边缘/凹槽极易形成异常高分。
- 结论：O-ROC=0.224 更像是“少量结构边缘高分支配目标分数”，而不是整体特征提取完全失效。优先修改目标级聚合与正常区域尺度校准，不应先增加大模型。

### 2.2 `100_1_scratch.txt`：牙刷类

- 点数：500,000；只有 XYZ 三列。
- PCA 主轴有效长度约 175.70，横向尺度约 15.94、11.26，长宽比约 11:1。
- 同一物体同时包含低曲率平滑手柄和高曲率、高拓扑复杂度刷毛区。
- 40-NN 物理半径中位数约 0.339，99% 分位约 0.461，局部尺度比螺钉更不均匀。
- 局部曲率中位数约 `3.74e-4`，99% 分位约 `1.04e-1`，复杂刷毛区会形成大量正常高响应。
- 侧视诊断中可见沿手柄延伸的细长高曲率带；现有二次平滑会显著削弱这类窄划痕。
- 结论：P-ROC=0.459 的主要风险是复杂刷毛背景偏置、细划痕采样不足和二次插值过度平滑。

> 限制：当前只提供了两个异常点云，没有相应 GT 掩码、同编号正常点云和程序输出的原始 score map。因此以上是几何诊断，不能把观察到的高曲率点直接视为 GT，也不能承诺具体分数增幅。

## 3. 当前源码中的关键问题

### P0-1：修复 MSND 尺度重复（高优先级）

文件：`feature_extractors/FPFH.py`

当前逻辑先执行：

```python
fpfh = torch.cat([fpfh, fpfh2], dim=-1)
```

随后又执行：

```python
fpfh = torch.cat([fpfh, fpfh2, fpfh3], dim=-1)
```

结果实际为 `FPFH40 + FPFH80 + FPFH80 + FPFH120`。改成只拼接一次：

```python
scale_features = [fpfh1, fpfh2, fpfh3]
scale_features = [torch.nn.functional.normalize(f, p=2, dim=-1)
                  for f in scale_features]
fpfh = torch.cat(scale_features, dim=-1)
```

要求：

- `--max_nn 40 --num_MSND 2`严格对应 40、80、120 三尺度。
- 增加单元测试，断言三尺度 FPFH 最终维数为 `33 * 3 = 99`，不能是132。
- 保留 `--legacy_duplicate_scale` 开关，仅用于对照。
- 预期资源影响：特征维度减少25%，`cdist`时间和显存下降，FPFH计算次数不变。

### P0-2：关闭或可配置二次平滑（高优先级）

文件：`feature_extractors/features.py`

当前 MiniShift 流程为：

```text
4096个中心分数 -> 插值到500k点
-> 再FPS到1024点 -> 12邻域均值
-> 再插值回500k点
```

第二段会模糊面积小于1%的细划痕。新增参数：

```text
--post_smooth_mode {none,legacy,knn_mean}
--post_smooth_k 12
--post_smooth_centers 1024
```

实现要求：

- `none`：第一次4096中心到全分辨率插值后直接输出。
- `legacy`：完全保留原逻辑，供基线复现。
- 首轮对照优先测试 `none` 与 `legacy`。
- 牙刷推荐起始值：`none`。
- 如果关闭后噪声过多，再尝试轻量 `knn_mean`，但不能再次降到1024中心；只在原4096中心分数上做很小邻域平滑。
- 预期资源影响：`none`比基线更快、更省显存。

### P0-3：插值只求最近3点，不做全排序（无损加速）

文件：`feature_extractors/pointnet2_utils.py`

把：

```python
dists, idx = dists.sort(dim=-1)
dists, idx = dists[:, :, :3], idx[:, :, :3]
```

替换为：

```python
dists, idx = torch.topk(
    dists, k=3, dim=-1, largest=False, sorted=False
)
```

要求：

- 保持原逆距离权重公式不变。
- 用固定输入验证新旧输出最大绝对误差 `< 1e-6`。
- `chunk_size`改为CLI参数，24GB显存推荐起始值10000；不得一次构造 `50000 x 4096` 后完整排序。

## 4. 螺钉 O-ROC 专项优化

### P1-1：把固定 Top-80 改为可配置的稳健目标分数

文件：`feature_extractors/features.py::compute_anomay_scores`

当前 MiniShift 使用500,000点中最高80点的均值，比例只有0.016%，仍接近最大值，极易被螺钉头部边缘噪声控制。

新增：

```python
def robust_object_score(point_scores, top_ratio=0.001,
                        min_topk=80, max_topk=2048):
    flat = point_scores.flatten()
    k = int(round(flat.numel() * top_ratio))
    k = max(min_topk, min(k, max_topk, flat.numel()))
    return torch.topk(flat, k, largest=True, sorted=False).values.mean()
```

候选值：

```yaml
top_ratio_candidates: [0.00016, 0.0005, 0.001, 0.002]
# 500k点时约对应80、250、500、1000点
```

选择规则：

- 不允许通过测试标签为每个测试样本选参数。
- 优先用训练正常点云的留一法分数，加少量训练期伪异常（局部法向位移/细线划痕）选择一个类别级参数。
- 如果没有可靠的训练期验证器，首选螺钉 `top_ratio=0.001`，并在完整ALL结果中与80点基线比较。
- 此修改只改变目标级聚合，不改变点分数，因此不会直接降低 P-ROC。

### P1-2：正常原型局部密度校准

目标：正常特征空间中稀疏区域（螺钉凹槽、头杆交界、牙刷刷毛）不应天然获得更高异常分数。

在 coreset 完成后，对每个原型计算其到其他原型的局部尺度：

```python
# 伪代码，必须分块实现，禁止一次生成过大的 bank x bank 矩阵
rho_j = median(distance(proto_j, k nearest other prototypes), k=5)
```

测试时：

```python
raw_dist, proto_idx = nearest_prototype_distance(test_features, patch_lib)
calibrated_dist = raw_dist / (rho[proto_idx] + 1e-6)
```

要求：

- 新增 `--prototype_density_norm` 开关。
- `rho`只由训练正常特征计算。
- 对 `rho`做1%和99%训练分位裁剪，防止极端除数。
- 原型库通常远小于原始点数，额外计算只发生一次；用分块距离避免显存峰值。
- 优先观察 Screw O-ROC，同时检查其他类别是否回退。

### P1-3：分层 coreset，保持总库大小不变

现有5%全局coreset可能丢掉少量但正常的凹槽、边缘和刷毛特征。新增可选 `stratified_coreset`：

1. 根据归一化PCA主轴位置分成8个重叠区间。
2. 再按局部表面变化率分成低/中/高三档。
3. 每个分层内做coreset选择。
4. 各层配额至少包含一个最小配额，并按层样本量分配；总原型数仍等于原来的5%。

资源要求：总特征库不得变大，搜索时间不增加。此项放在密度校准之后测试，不与其同时首次启用。

## 5. 牙刷 P-ROC 专项优化

### P2-1：增加4维轻量局部几何特征，复用现有KNN

文件：`feature_extractors/FPFH.py::get_fpfh_features`

现有代码已经得到每个中心的 `group_size` 个邻点索引。复用邻域坐标计算3x3协方差特征，不新增一次KNN：

```text
surface_variation = lambda0 / (lambda0 + lambda1 + lambda2)
linearity         = (lambda2 - lambda1) / lambda2
planarity         = (lambda1 - lambda0) / lambda2
scattering        = lambda0 / lambda2
```

其中 `lambda0 <= lambda1 <= lambda2`。

要求：

- 使用 `torch.linalg.eigvalsh` 批量处理4096个3x3矩阵。
- 对4维几何特征使用训练正常特征的 median/MAD 做标准化。
- 与L2归一化后的FPFH拼接：`concat(fpfh, geom_weight * geom)`。
- `geom_weight`候选 `[0.10, 0.20, 0.30]`，起始推荐0.20。
- 新增 `--use_geom4d` 与 `--geom_weight`。
- 预期额外耗时很小，能增强平滑手柄上窄划痕的响应；正常刷毛由训练原型与密度校准吸收。

### P2-2：牙刷使用更局部的邻域，优先降低而非增加计算量

候选配置：

```yaml
toothbrush_sweep:
  max_nn: [20, 30, 40]
  group_size: [64, 96, 128]
  post_smooth_mode: [none, legacy]
```

建议第一组：

```yaml
toothbrush:
  max_nn: 30       # 三尺度30/60/90
  group_size: 64
  post_smooth_mode: none
```

原因：减小FPFH与LFSA邻域可减少窄划痕被周围正常手柄平均掉，同时减少计算量。只有当P-ROC明显下降时才恢复40/128。

### P2-3：仅在必要时将牙刷中心数提升到6144

如果关闭二次平滑和减小邻域后，牙刷 P-ROC 仍低于0.55，再测试：

```yaml
toothbrush:
  num_group: 6144
```

不要直接全类别使用8192。只把牙刷从4096提高到6144，理论上该类别中心相关计算约增加50%，摊到12类总运行时间约增加4%-6%。必须记录实测值。

### P2-4：可选的法向感知插值

仅当无二次平滑导致跨表面泄漏时实现。复用Open3D已计算的法向，给最近3个中心增加法向一致性权重：

```text
w_i = 1/(d_i+eps) * exp(-(1-abs(dot(n_point,n_center)))/tau)
```

候选 `tau=[0.05, 0.10, 0.20]`。此项复杂度较高，排在前述方案之后，不作为第一轮修改。

## 6. 计算资源与复现实验优化

### P3-1：特征缓存

缓存键必须至少包含：

```text
dataset, category, split, level, feature,
max_nn, num_MSND, num_group, group_size,
use_MSND, use_LFSA, use_geom4d, geom_weight,
source_code_version
```

分两层缓存：

1. `train_feature_bank_before_coreset`；
2. `test_center_features + centers + optional_normals`。

只改变目标分数、密度校准、平滑或插值时，不重新提取FPFH。缓存不得包含GT和测试标签。

### P3-2：分块 `cdist`

将：

```python
dist = torch.cdist(patch, self.patch_lib)
```

改成精确的分块最近邻计算，逐块维护最小值和原型索引。不得使用近似搜索作为第一版，以免改变分数。新增参数：

```text
--query_chunk 1024
--bank_chunk 8192
```

### P3-3：完整日志

每个类别记录：

```json
{
  "category": "screw",
  "fit_seconds": 0.0,
  "test_seconds": 0.0,
  "peak_vram_gb": 0.0,
  "num_train_samples": 0,
  "num_test_samples": 0,
  "num_points": 500000,
  "o_roc": 0.0,
  "p_roc": 0.0,
  "config": {}
}
```

运行结束必须输出12类表、两个均值、最终分数、总时间和峰值显存。

## 7. 推荐实施顺序与消融矩阵

不要一次叠加全部改动。按以下顺序，每步完整跑ALL并与前一步比较。

| 实验ID | 修改 | 主要目标 | 预期资源影响 |
|---|---|---|---|
| B0 | 原基线复现 | 对照 | 1.00x |
| E1 | 修复MSND重复 + 插值topk | 正确性/加速 | 时间下降 |
| E2 | E1 + `post_smooth=none` | 提升P-ROC | 时间下降 |
| E3 | E2 + 原型密度校准 | Screw O-ROC | 训练建库小幅增加 |
| E4 | E3 + `top_ratio=0.001` | Screw O-ROC | 几乎不变 |
| E5 | E4 + geom4d(0.20) | Toothbrush P-ROC | 小于10% |
| E6 | E5 + Toothbrush 30/64 | Toothbrush P-ROC/提速 | 可能下降 |
| E7 | E6 + Toothbrush 6144中心 | 最后补充分辨率 | 总时间约+4%-6% |

每一步保留条件：

```yaml
acceptance:
  final_score_delta: "> 0"
  preferred_final_score_delta: ">= 0.005"
  runtime_ratio: "<= 1.15"
  vram_gb: "< 48"
  catastrophic_drop_guard:
    any_category_o_roc_drop: "< 0.03"
    any_category_p_roc_drop: "< 0.03"
```

若某修改显著提高短板但使少数类别下降，可通过单一 `configs/minishift.yaml` 配置类别参数，不要在算法函数中散布类别名判断。

建议配置结构：

```yaml
default:
  max_nn: 40
  num_group: 4096
  group_size: 128
  post_smooth_mode: none
  object_top_ratio: 0.001
  prototype_density_norm: true
  use_geom4d: true
  geom_weight: 0.20

categories:
  screw:
    object_top_ratio: 0.001
    prototype_density_norm: true
  toothbrush:
    max_nn: 30
    group_size: 64
    num_group: 4096
    post_smooth_mode: none
```

## 8. 自动化验收测试

AI代码编辑器必须增加以下测试或检查脚本：

1. `test_msnd_dimension.py`：三尺度维数为99，且不存在重复拼接。
2. `test_interpolation_equivalence.py`：旧全排序和新topk插值误差 `<1e-6`。
3. `test_metric_integrity.py`：确认 `calculate_metrics()`、标签和GT读取逻辑未被修改。
4. `test_no_test_label_leakage.py`：模型输出路径中不能读取label或mask来选择参数、阈值、分支或融合权重。
5. `test_object_aggregator.py`：不同点数下top-ratio边界正确，输出有限值。
6. `smoke_one_category.py`：单类别完成fit/evaluate并输出计时、显存和两项指标。
7. `compare_all.py`：读取两份12类日志，输出每类ΔO、ΔP、均值差、最终分数差、耗时比和显存比。

## 9. 明确不建议的方向

- 不要第一阶段引入PointTransformer、大型神经网络或长epoch训练。
- 不要全类别把4096点直接提高到8192或更高。
- 不要扩大coreset总比例后再无控制地使用全矩阵 `cdist`。
- 不要对每个测试样本做min-max归一化后计算O-ROC，这会破坏样本间目标分数排序。
- 不要根据测试GT、异常文件名或测试标签调阈值/权重。
- 不要修改官方类别平均、AUROC计算或ALL数据选择逻辑。

## 10. 预期收益优先级

当前最终分数：

```text
(0.673 + 0.660) / 2 = 0.6665
```

如果仅把 Screw O-ROC 从0.224提升到0.500，其他项目不变，则最终分数约提升：

```text
(0.500 - 0.224) / 12 / 2 = 0.0115
```

如果再把 Toothbrush P-ROC 从0.459提升到0.600，则约再提升：

```text
(0.600 - 0.459) / 12 / 2 = 0.0059
```

因此最合理的资源顺序是：

```text
稳健目标聚合/正常密度校准
> 关闭二次平滑/修复尺度重复
> 轻量几何特征与较小局部邻域
> 仅牙刷增加中心数量
> 大型学习模型（当前不建议）
```

