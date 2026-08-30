# readme徐补充

这份文档用于补充原项目 `README.md`，面向第一次在自己电脑上部署并运行本项目的同学。

本文档基于两部分信息整理：

1. 原作者 `README.md` 给出的训练命令与核心版本要求
2. 本机实际部署、调试、跑通 MiniShift 数据集时的经验总结

---

## 1. 先说结论

本项目推荐使用下面这套环境：

- Windows 10/11
- WSL2
- Ubuntu 22.04 LTS
- Miniconda
- Python 3.8
- PyTorch 1.12.1
- torchvision 0.13.1
- torchaudio 0.12.1
- cudatoolkit 11.3
- gcc/g++ 10

不推荐直接在 Windows 原生 Python 环境运行。  
原因是项目依赖 `pointnet2_ops`、`KNN_CUDA` 等 CUDA 扩展，在 Windows 下更容易出编译和路径问题。

## 2. WSL 和 Ubuntu 安装

### 2.1 在 Windows 终端中执行

请在 **管理员身份** 的 `Anaconda Prompt` 或 `PowerShell` 中输入：

```bash
wsl --install -d Ubuntu-22.04
```

如果之前没有启用 WSL，系统会自动安装。

安装完成后，启动 Ubuntu：

```bash
wsl -d Ubuntu-22.04
```

第一次启动时会要求：

1. 创建 Linux 用户名
2. 设置 Linux 密码

例如用户名可以设为：

```text
xu
```

设置密码时，终端里通常会出现类似提示：

```text
New password:
Retype new password:
```

注意：

- 输入密码时，屏幕上**不会显示任何字符**
- 输入数字、字母后看起来是空白，这属于**正常现象**
- 输入完后直接按 `Enter`
- 然后再次输入**相同密码**，再按 `Enter`
- 如果两次密码一致，就会完成创建

### 2.2 在 Ubuntu 中确认版本

```bash
lsb_release -a
```

正确输出应类似：

```bash
Description: Ubuntu 22.04.x LTS
Codename: jammy
```

### 2.3 在 Ubuntu 中确认 GPU

```bash
nvidia-smi
```

如果能看到显卡信息，说明 WSL GPU 可用。

---

## 3. 基础系统依赖安装

以下命令都在 **Ubuntu 终端** 中输入。

```bash
sudo apt update
sudo apt install -y build-essential git wget curl unzip ninja-build patchelf libgl1 libglib2.0-0 gcc-10 g++-10
```

检查 `g++-10` 是否装好：

```bash
g++-10 --version
```

---

## 4. 安装 Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中：

1. 连续按 `Enter`
2. 出现协议时输入 `yes`
3. 安装路径默认即可
4. 最后初始化时输入 `yes`

安装完成后：

```bash
source ~/.bashrc
conda --version
```

## 5. 本项目推荐环境版本

下面这组版本是结合：

- 原作者 README
- 本机实际部署经验
- Python 3.8 兼容性

整理出来的稳定版本。

### 5.1 创建虚拟环境

```bash
conda create -n Simple3D_env python=3.8 -y
conda activate Simple3D_env
```

如果新开了终端，先执行：

```bash
source ~/.bashrc
conda activate Simple3D_env
```

### 5.2 安装 PyTorch

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
```

检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

应类似输出：

```bash
1.12.1
True
```

### 5.3 常规 Python 依赖

```bash
pip install tifffile==2023.7.10 open3d-cpu==0.18.0 pandas==2.0.3 scikit-learn==1.3.2 scipy==1.10.1 tqdm opencv-python==4.9.0.80 pillow==10.4.0 timm==0.6.13 moviepy==1.0.3 matplotlib==3.7.5 seaborn==0.12.2 tabulate
```

### 5.4 安装 PointNet2

```bash
cd ~
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
```

如果 `git clone` 失败，可改用 zip 下载。

设置 CUDA 编译环境：

```bash
conda install -c conda-forge cudatoolkit-dev=11.3 -y
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_DONT_CHECK_COMPILER_ABI=1
export TORCH_NVCC_FLAGS="-allow-unsupported-compiler"
```

安装：

```bash
cd ~/Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .
```

检查：

```bash
python -c "from pointnet2_ops import pointnet2_utils; print('pointnet2_ops ok')"
```

### 5.5 安装 KNN_CUDA

原作者 wheel 下载链接已经失效，建议用源码安装。

```bash
cd ~
wget https://github.com/kangya998/KNN_CUDA/archive/refs/heads/master.zip -O KNN_CUDA.zip
unzip KNN_CUDA.zip
cd ~/KNN_CUDA-master
```

设置环境变量：

```bash
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_DONT_CHECK_COMPILER_ABI=1
export TORCH_NVCC_FLAGS="-allow-unsupported-compiler"
```

安装：

```bash
python setup.py install
```

检查：

```bash
python -c "from knn_cuda import KNN; print('knn_cuda ok')"
```

---

## 6. 项目代码拷贝到 WSL 本地目录

假设原项目在 Windows 盘：

```bash
cp -r "/mnt/e/study/MAS_project/智能制造2026/MiniShift-Simple3D-main" ~/MiniShift-Simple3D-main
cd ~/MiniShift-Simple3D-main
```

后续建议都在这个目录里运行：

```bash
cd ~/MiniShift-Simple3D-main
```

---

## 7. 数据集位置是写死的，必须提前检查

### 7.1 MiniShift

当前代码中，MiniShift 数据集路径写死在：

- `data/MiniShiftAD.py`

当前值是：

```python
DATASETS_PATH = '/mnt/g/MiniShiftAD'
```

这意味着：

- Windows 中数据位置：`G:\MiniShiftAD`
- WSL 中访问路径：`/mnt/g/MiniShiftAD`

每次新开终端后，建议先挂载：

```bash
sudo mkdir -p /mnt/g
sudo mount -t drvfs G: /mnt/g
```

然后检查：

```bash
ls /mnt/g/MiniShiftAD
```

应能看到 12 个类别目录，例如：

```bash
capsule
cube
spring_pad
screw
screen
piggy
nut
flat_pad
plastic_cylinder
button_cell
toothbrush
light
```

### 7.2 其他数据集路径也是写死的

如果以后跑别的数据集，需要去对应文件里改 `DATASETS_PATH`：

- `data/mvtec3d.py`
- `data/real3d.py`
- `data/MulSen.py`
- `data/anomalyshape.py`

也就是说，本项目不是通过命令行传数据路径，而是直接在代码里写死。

---

## 8. 核心依赖检查命令

进入项目目录后，执行：

```bash
cd ~/MiniShift-Simple3D-main
python - <<'PY'
mods = [
    ("torch", "import torch"),
    ("torchvision", "import torchvision"),
    ("pandas", "import pandas"),
    ("open3d", "import open3d"),
    ("cv2", "import cv2"),
    ("timm", "import timm"),
    ("scipy", "import scipy"),
    ("sklearn", "import sklearn"),
    ("tifffile", "import tifffile"),
    ("pointnet2_ops", "from pointnet2_ops import pointnet2_utils"),
    ("knn_cuda", "from knn_cuda import KNN"),
]
ok = True
for name, code in mods:
    try:
        exec(code, {})
        print(f"[OK] {name}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {e}")
print("ALL CORE DEPS OK" if ok else "SOME DEPS FAILED")
PY
```

---

## 9. 每次训练前推荐执行的环境准备命令

新开 Ubuntu 终端后，建议先执行这一整段：

```bash
source ~/.bashrc && \
conda activate Simple3D_env && \
sudo mkdir -p /mnt/g && \
sudo mount -t drvfs G: /mnt/g && \
export CC=/usr/bin/gcc-10 && \
export CXX=/usr/bin/g++-10 && \
export CUDAHOSTCXX=/usr/bin/g++-10 && \
export CUDA_HOME=$CONDA_PREFIX && \
export PATH=$CUDA_HOME/bin:$PATH && \
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
export TORCH_DONT_CHECK_COMPILER_ABI=1 && \
export TORCH_NVCC_FLAGS="-allow-unsupported-compiler" && \
cd ~/MiniShift-Simple3D-main
```

---

## 10. 按原作者 README 复现 MiniShift 全 12 类

原作者 README 对应的核心命令是：

```bash
python main.py --dataset minishift --num_group 4096 --group_size 128 --max_nn 40 --use_LFSA True --use_MSND True --num_MSND 2 --expname MiniShift_ALL --level ALL --vis_save True
```

如果本机磁盘空间足够，并且希望完全按原作者命令保存可视化结果，就直接使用上面这条命令。

如果本机空间不够，或者不需要保存每个测试样本的点云可视化结果，可以把 `vis_save` 改成 `False`，这样会减少大量 `.txt` 文件写入。

### 10.1 不保存可视化的训练命令

```bash
python main.py --dataset minishift --num_group 4096 --group_size 128 --max_nn 40 --use_LFSA True --use_MSND True --num_MSND 2 --expname MiniShift_ALL --level ALL --vis_save False
```

### 10.2 保存可视化的训练命令

```bash
python main.py --dataset minishift --num_group 4096 --group_size 128 --max_nn 40 --use_LFSA True --use_MSND True --num_MSND 2 --expname MiniShift_ALL --level ALL --vis_save True
```

---

## 11. 只训练 2-4 个类别，而不是全部 12 类

默认 `main.py` 会自动遍历整个数据集里的所有类别。  
也就是说：

```bash
python main.py --dataset minishift ...
```

会跑全部 12 类。

如果只想跑其中 2-4 类，最简单的方法是用内嵌 Python 调用，并临时覆盖类别列表。

### 11.1 只跑 2 类：`capsule` 和 `cube`

```bash
python - <<'PY'
from types import SimpleNamespace
import main

main.minishiftAD_classes = lambda: ["capsule", "cube"]

args = SimpleNamespace(
    expname="MiniShift_2cls",
    device="cuda:0",
    dataset="minishift",
    max_nn=40,
    num_group=4096,
    group_size=128,
    use_MSND=True,
    use_LFSA=True,
    vis_save=False,
    num_MSND=2,
    feature="FPFH",
    level="ALL",
)

main.run_3d_ads(args)
PY
```

### 11.2 跑 3 类：`capsule`、`cube`、`spring_pad`

```bash
python - <<'PY'
from types import SimpleNamespace
import main

main.minishiftAD_classes = lambda: ["capsule", "cube", "spring_pad"]

args = SimpleNamespace(
    expname="MiniShift_3cls",
    device="cuda:0",
    dataset="minishift",
    max_nn=40,
    num_group=4096,
    group_size=128,
    use_MSND=True,
    use_LFSA=True,
    vis_save=False,
    num_MSND=2,
    feature="FPFH",
    level="ALL",
)

main.run_3d_ads(args)
PY
```

### 11.3 跑 4 类：`capsule`、`cube`、`spring_pad`、`screw`

```bash
python - <<'PY'
from types import SimpleNamespace
import main

main.minishiftAD_classes = lambda: ["capsule", "cube", "spring_pad", "screw"]

args = SimpleNamespace(
    expname="MiniShift_4cls",
    device="cuda:0",
    dataset="minishift",
    max_nn=40,
    num_group=4096,
    group_size=128,
    use_MSND=True,
    use_LFSA=True,
    vis_save=False,
    num_MSND=2,
    feature="FPFH",
    level="ALL",
)

main.run_3d_ads(args)
PY
```

### 11.4 MiniShift 的 12 个类别名

```text
capsule
cube
spring_pad
screw//徐亦捷
screen
piggy
nut
flat_pad//倪鹏
plastic_cylinder
button_cell
toothbrush
light//宋平原
```

---

## 12. 训练结果怎么看

每跑完一个类别，终端会打印类似下面这一行：

```text
Class: capsule, Simple3D Image ROCAUC: 0.912, Simple3D Pixel ROCAUC: 0.812, Simple3D AU-PRO: 0.000
```

其中：

- `Image ROCAUC`：可以理解为图像级 O-ROC
- `Pixel ROCAUC`：可以理解为点/像素级 P-ROC
- `AU-PRO`：当前代码对 `minishift` 默认是 `0`

### 12.1 日志位置

日志会写到：

```text
./logs/<expname>.txt
```

例如：

```text
./logs/MiniShift_ALL.txt
```

### 12.2 可视化输出位置

如果开启了 `--vis_save True`，会生成两类结果：

- `./vis-results/`
- `./vis-results-GT/`

注意：  
这些不是图片，而是**带颜色信息的点云 txt 文件**。  
文件中每一行通常是：

```text
x y z r g b
```

所以打开后看到一堆数字是正常的。

### 12.3 如何把 txt 结果渲染成视频

示例命令：

```bash
python render_video.py --input_paths "./vis-results/capsule/test/good/0.txt" "./vis-results-GT/capsule/test/good/0.txt" --output_path "capsule_good_0.mp4"
```

生成后即可直接看：

```text
capsule_good_0.mp4
```

## 13. 最推荐的一套实际执行顺序

### 13.1 第一次部署

1. 安装 WSL2 + Ubuntu 22.04
2. 安装 Miniconda
3. 创建 `Simple3D_env`
4. 安装 PyTorch、PointNet2、KNN_CUDA
5. 把项目复制到 `~/MiniShift-Simple3D-main`
6. 挂载 `G:\MiniShiftAD`
7. 跑核心依赖检查命令

### 13.2 正式训练

每次新开终端：

1. 执行第 9 节的环境准备命令
2. 先跑 2 类或 4 类小规模验证
3. 确认没问题后再跑全 12 类

### 13.3 全量训练命令

```bash
python main.py --dataset minishift --num_group 4096 --group_size 128 --max_nn 40 --use_LFSA True --use_MSND True --num_MSND 2 --expname MiniShift_ALL --level ALL --vis_save False
```
