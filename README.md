# PVD_hand — 手部点云补全（Diffusion）

基于 **Point-Voxel CNN（PVCNN2）+ 扩散模型** 的人手点云补全工作（PVD_hand）。
给定被抓握物体的局部点云条件，模型用扩散逆向过程补全出**人手点云**，并按手指部件输出类别。

本仓库只保留与 **manohand 补全的训练 / 推理 / 可视化** 相关的代码。

---

## 特性

- PVCNN2 主干 + 高斯扩散（`mse / eps / fixedsmall`，linear 噪声表）。
- 条件为被抓握**物体上与手最接近的点**（已按接触度降序重排，见“数据”）。
- 人手每个点 5 通道：`xyz + 手部件标签(1..5) + 接触度(0..1)`。
- 自带可旋转交互可视化（Qt 弹窗）与非交互出图两种。
- 自定义 CUDA 算子 `_pvcnn_backend` 首次 import 时自动编译，编译一次后复用。

---

## 环境与依赖

- Python ≥ 3.10，建议 **3.12**；Linux + NVIDIA GPU。
- **PyTorch 需为能支持你 GPU 计算能力的 CUDA 版**。例如 RTX 50 系(Blackwell, sm_120)：
  ```bash
  micromamba create -n test-py312 -c conda-forge python=3.12
  micromamba run -n test-py312 pip install torch torchvision \
      --index-url https://download.pytorch.org/whl/cu130
  micromamba run -n test-py312 pip install numpy scipy matplotlib trimesh tqdm ninja
  ```
  或用 `requirements.txt`：
  ```bash
  micromamba run -n test-py312 pip install -r requirements.txt
  ```

## CUDA 算子编译（首次运行自动完成）

`modules/functional` 的核心算子通过 `torch.utils.cpp_extension.load` **首次 import 时 JIT 编译**，产物缓存到项目内 `.torch_extensions/`（已 gitignore），之后直接复用、不再重编。需要：

- 一个支持你 GPU 架构的 **nvcc**（如 `/usr/local/cuda/bin/nvcc`）；
- **gcc/g++** 与 **ninja**。

`modules/functional/backend.py` 里按 **compute_120/sm_120** 编译（适配 RTX 50 系）。换用其他架构请改该文件的 `extra_cuda_cflags` 的 `-gencode`。

首次编译较慢，请勿并发过多 nvcc（会吃内存）。可设 `MAX_JOBS=1` 限制。

---

## 数据准备与格式

**数据不入库**（体积大），请自行放到 `datasets/`：

```
datasets/manohand.pt
```

`manohand.pt` 是用 `torch.save` 存的一个 dict：

```
{
  "hands": Tensor[31292, 300, 5],   # 人手点云：300 点 × (xyz + 部件标签 + 接触度)
  "objs" : Tensor[31292, 2048, 5],  # 被抓握物体点云（同样 5 通道）
}
```

- 通道：`0..2 = xyz`，`3 = 手部件标签(1..5)`，`4 = 接触度(0..1，越大离手越近)`。
- 测试/训练划分：`datasets/Rdataset.py` 把每个文件**前 `testnum`(默认 500) 行当 test，其余当 train**。
- 语义（`mode='cmap'`）：每个样本的**条件点 = `objs` 的前 `svpoints`(默认 50) 个点**。仓库内已把 `objs` 按接触度**降序重排**，使前 50 个点即“物体上离手最近的点”。补全目标是手点云。
- 数据为 float32、坐标已归一化放缩。

> 说明：旧的生成 / 机械手 / 原始处理代码与数据都已整理到 `_deprecated_*` 目录，不入库、不参与补全链路。

---

## 训练

```bash
python train_completion.py --distribution_type single --gpu 0 --bs 32 --niter 1000
```

常用参数（默认已指向 `datasets/manohand.pt`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataroot` | `datasets` | 数据根目录 |
| `--category` | `['manohand.pt']` | 参与训练的类别文件 |
| `--nc` | `5` | 点云通道数 |
| `--npoints` | `300` | 完整点数 |
| `--svpoints` | `50` | 条件(已知)点数 |
| `--embed_dim` | `64` | 通道数 |
| `--bs` | `128` | batch size（小显存建议 32） |
| `--niter` | `10000` | 总 epoch |
| `--saveIter` | `100` | 每 N epoch 存 `epoch_*.pth` |
| `--model` | `''` | 续训：填某 `epoch_*.pth` |

- 支持 `--distribution_type single|multi`，`multi` 用 NCCL 多卡。
- 从已训权重续训/微调：`--model <epoch_X.pth>`（会自动剥掉 `DataParallel` 的 `module.` 前缀）。

## 推理（定量）

```bash
python test_completion.py --model <epoch_X.pth> --samples 0 5 20 --outdir output/completion
```

对每个测试样本做扩散采样得到补全手，保存 `sampleXXXX.npz`（含 GT 手与补全手点云）并打印 **Chamfer 距离**。

## 可视化（交互 & 非交互）

```bash
# 交互：弹出 Qt 窗口，可旋转/缩放（需图形环境，如 WSLg/X11）
python viz_interactive.py --model <epoch_X.pth> --sample 5

# 多个样本、依次看
python viz_interactive.py --model <epoch_X.pth> --samples 0 5 20

# 无窗口，直接存 PNG
python viz_interactive.py --model <epoch_X.pth> --sample 5 --save-only output/interactive_viz
```

每个样本出两个图：
1. **物体点云单独图**：按接触度红(高)→绿(低)渐变；
2. **物体 + 补全手叠加图**：物体接触着色，手按部件 link 分色（手点已放大便于观察）。

> 若弹窗不可用：确保 matplotlib 后端为 Qt5Agg/TkAgg（本项目脚本会自动选 Qt5Agg），并在有 `DISPLAY` 的会话里运行。

---

## 目录结构

```
PVD_hand_new/
├── datasets/
│   ├── Rdataset.py        # 数据加载（唯一入口）
│   └── manohand.pt        # 数据（gitignore，需自备）
├── model/pvcnn_completion.py  # PVCNN2Base 补全骨干
├── modules/               # PVCNN 算子(python 封装)
│   └── functional/        # 需编译的 CUDA 核心(首次 import 自动编)
├── utils/                 # file_utils / visualize / metrics
├── train_completion.py    # 训练
├── test_completion.py     # 推理 + Chamfer
├── viz_interactive.py     # 交互/出图可视化
├── requirements.txt
└── _deprecated_*          # 旧代码/数据/输出（不入库）
```

---

## 注意事项

- 首次跑任意入口会触发 `_pvcnn_backend` CUDA 编译，耗时几分钟，请耐心并控制并行。
- `output/`、`.torch_extensions/`、`datasets/*.pt`、`_deprecated_*` 均已被 `.gitignore` 忽略，不会入库。
- 本项目仓库仅用于 manohand 补全；需要其它手型 / 生成任务请参考 `_deprecated_*`。
