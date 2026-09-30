# PVD_hand — 人手点云补全（Diffusion）

[简体中文](README.md) | [English](README.en.md)

基于 **Point-Voxel CNN（PVCNN2）+ 扩散模型** 的**人手点云补全**（骨干与扩散框架基于 [PVD](https://github.com/alexzhou907/PVD)）。给定被抓握物体的局部点云条件，模型通过扩散逆向过程补全出**人手点云**，并为每个手点预测手指部件类别。

![object + completed hand](assets/Figure_0_2.png)

---

## 数据来源（重要）

本工作使用 **S2HGD** 数据集（*Single-view Scene-to-Human Grasp Dataset*）：

> Yan-Kang Wang, Chengyi Xing, Yi-Lin Wei, Xiao-Ming Wu, Wei-Shi Zheng.
> **Single-View Scene Point Cloud Human Grasp Generation.** arXiv:2404.15815, 2024.

S2HGD 包含约 **99,000** 个**单物体、单视角场景点云**，覆盖 **1,668 个不同物体**，每个点云配有**一个人手抓取标注**（真实采集并由人手拟合得到）。本仓库在此数据基础上，把人手抓取拟合为 **MANO** 参数并转成点云，构建 `manohand.pt` 用于人手点云补全。**数据版权归原作者所有**，请遵循其许可。

> **注意：只有“数据”来自 S2HGD；本仓库的“代码”与那篇论文无关**——代码是基于 PVCNN2 的扩散式人手点云补全的独立实现。

## 效果展示

`assets/` 下的图片命名规则：`Figure_<id>_1` = **物体点云**，`Figure_<id>_2` = **物体 + 补全的人手**（`<id>` 为样本/物体序号）。

| 物体 #0 | 物体 #0 + 补全手 |
|---|---|
| ![obj0](assets/Figure_0_1.png) | ![obj0+hand](assets/Figure_0_2.png) |

| 物体 #5 | 物体 #5 + 补全手 |
|---|---|
| ![obj5](assets/Figure_5_1.png) | ![obj5+hand](assets/Figure_5_2.png) |

| 物体 #20 | 物体 #20 + 补全手 |
|---|---|
| ![obj20](assets/Figure_20_1.png) | ![obj20+hand](assets/Figure_20_2.png) |

---

## 环境与依赖

- Python ≥ 3.10（建议 3.12）；Linux + NVIDIA GPU。
- **PyTorch 需为支持你 GPU 计算能力的 CUDA 版**。例如 RTX 50 系(Blackwell, sm_120)：

```bash
micromamba create -n pvd_hand -c conda-forge python=3.12
micromamba run -n pvd_hand pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu130
micromamba run -n pvd_hand pip install -r requirements.txt
```

**CUDA 算子编译**：`modules/functional` 的核心算子会在**首次 import 时自动 JIT 编译**（缓存到项目内 `.torch_extensions/`，之后复用）。需要 `nvcc` + `gcc/g++` + `ninja`。`modules/functional/backend.py` 默认按 `sm_120` 构建，换卡请改其中的 `-gencode`。

## 数据准备

数据不入库，请将 `manohand.pt` 放到 `datasets/`：

```
datasets/manohand.pt
```

格式（`torch.save` 的 dict）：`{"hands": Tensor[N,300,5], "objs": Tensor[N,2048,5]}`。
通道：`0..2 = xyz`，`3 = 手部件标签(1..5)`，`4 = 接触度(0..1)`。
详见 **[DATASET.md](DATASET.md)** / **[DATASET.en.md](DATASET.en.md)**（数据来源、生成流程、各中间文件说明）。

## 预训练模型

发布的权重位于：

```
checkpoints/epoch_599.pth
```

> 权重体积较大（~330MB），**不在 git 仓库中**（见 `.gitignore`）。请从发布页 / 网盘获取后放到 `checkpoints/`。
> 该权重为多类联合训练（manohand+shadowhand+allegro+ezgripper）的 epoch_599，可加载到本仓库的人手补全模型中直接推理。

## 训练

```bash
python train_completion.py --distribution_type single --gpu 0 --bs 32 --niter 1000
```

默认已指向 `datasets/manohand.pt`。常用参数：`--svpoints 50`、`--npoints 300`、`--nc 5`、`--bs`（小显存建议 32）、`--saveIter`（每 N epoch 存一次）、`--model <epoch_X.pth>`（续训/微调）。

## 推理（定量）

```bash
python test_completion.py --model checkpoints/epoch_599.pth --samples 0 5 20 --outdir output/completion
```

对测试样本做扩散采样得到补全手，保存 `sampleXXXX.npz`（含 GT 手与补全手点云）并打印 **Chamfer 距离**。

## 可视化（交互 & 出图）

```bash
# 交互弹窗（需图形环境，如 WSLg/X11），可旋转/缩放
python viz_interactive.py --model checkpoints/epoch_599.pth --sample 5

# 无窗口，直接存 PNG（每个样本出两张：物体单独图 + 物体+手叠加图）
python viz_interactive.py --model checkpoints/epoch_599.pth --samples 0 5 20 --save-only output/interactive_viz
```

---

## 目录结构

```
PVD_hand_new/
├── assets/                    # README 展示图 (Figure_<id>_1/_2)
├── checkpoints/               # 预训练权重 (gitignore)
├── datasets/
│   ├── Rdataset.py            # 数据加载
│   └── manohand.pt            # 数据 (gitignore, 需自备)
├── model/pvcnn_completion.py  # PVCNN2 补全骨干
├── modules/                   # PVCNN 算子(python 封装)
│   └── functional/            # 需编译的 CUDA 核心 (首次 import 自动编)
├── utils/                     # file_utils / visualize / metrics
├── train_completion.py        # 训练
├── test_completion.py         # 推理 + Chamfer
├── viz_interactive.py         # 交互/出图可视化
├── DATASET.md                 # 数据集说明（中文）
├── DATASET.en.md              # dataset doc (English)
└── requirements.txt
```

## 引用 / 致谢

### 代码基础（PVD）

本仓库的 PVCNN2 骨干与扩散框架基于 **PVD**（Point-Voxel Diffusion）：

- 代码：<https://github.com/alexzhou907/PVD>

```bibtex
@inproceedings{Zhou_2021_ICCV,
  author    = {Zhou, Linqi and Du, Yilun and Wu, Jiajun},
  title     = {3D Shape Generation and Completion Through Point-Voxel Diffusion},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2021},
  pages     = {5826-5835}
}
```

### 数据（S2HGD）

**数据来自 S2HGD；本仓库代码与该论文无关。** 若你使用了本数据，请引用 S2HGD 原论文：

```bibtex
@article{wang2024single,
  title={Single-View Scene Point Cloud Human Grasp Generation},
  author={Wang, Yan-Kang and Xing, Chengyi and Wei, Yi-Lin and Wu, Xiao-Ming and Zheng, Wei-Shi},
  journal={arXiv preprint arXiv:2404.15815},
  year={2024}
}
```

## 说明

- `output/`、`.torch_extensions/`、`datasets/*.pt`、`checkpoints/*.pth` 已被 `.gitignore` 忽略。
- 本仓库聚焦人手点云补全的训练 / 推理 / 可视化。
