# MiniShift Simple3D V10

这是基于 Simple3D / MiniShiftAD V2 主干整理的类别路由优化与 AI 代码交接仓库。

## 当前内容

- `current/`：V10 实验代码与便携启动脚本。
- `v2_runtime/`：配套 V2 主干运行时代码。
- `evidence/`：12 类结果 JSON 与两批汇总 JSON。
- `V10_AI代码修改说明.txt`：供 ChatGPT、Codex、Cursor 等 AI 代码编辑器读取的完整改造规范。
- `交接说明.txt`：目录、环境和待验证事项说明。

## 已测结果

在本地 MiniShift 12 类评估中：

- Simple3D 基线平均 O-ROC：约 `0.710`
- V10 平均 O-ROC：约 `0.740`
- Simple3D 基线平均 P-ROC：约 `0.666`
- V10 平均 P-ROC：约 `0.689`
- 类别综合均值：约 `0.688 -> 0.715`

12 类中 10 类综合值提高。`screw` 的对象级聚合和 `plastic_cylinder` 的原始路径回退仍是下一阶段待验证项，详见 AI 修改规范。

## 运行

1. 准备 Ubuntu 22.04 WSL、CUDA、PyTorch 与项目依赖。
2. 将 12 类数据放在 `D:\BaiduNetdiskDownload`，或修改 `current/train_single_category.py` 中的 `DATA_ROOT`。
3. 如环境脚本路径不是 `/home/xu/simple3d-env.sh`，修改 `current/start_training.bat`。
4. 双击 `current/start_training.bat`，或在 WSL 中运行：

```bash
cd current
python -u train_single_category.py
```

代码支持类别专属参数、逐类结果、诊断 JSONL、批次汇总和断点续跑。

## 注意

- 仓库不包含数据集、CUDA 环境、模型缓存和可视化大文件。
- `evidence/` 只用于核对历史实验，不得被训练代码读取或用于在线选择测试参数。
- 从零运行 12 类的本地估计约为 23 小时，硬件和磁盘速度不同会造成变化。
