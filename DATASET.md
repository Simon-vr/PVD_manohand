# manohand 数据集说明

[简体中文](DATASET.md) | [English](DATASET.en.md)

面向 **人手点云补全** 的人-物抓握点云数据集：每个样本 = 一只 MANO 人手点云 + 一个被抓握的物体点云。数据用于训练/评估扩散模型，在给定物体局部点云条件下补全出人手点云。

## 1. 数据来源

数据集建立在 **S2HGD**（*Single-view Scene-to-Human Grasp Dataset*）之上：

> Yan-Kang Wang, Chengyi Xing, Yi-Lin Wei, Xiao-Ming Wu, Wei-Shi Zheng.
> **Single-View Scene Point Cloud Human Grasp Generation.** arXiv:2404.15815, 2024.

S2HGD 提供约 **99,000** 个**单物体、单视角场景点云**，覆盖 **1,668 个不同物体**，每个点云配有一个真实采集的**人手抓取标注**。本数据集将这些抓取拟合为 **MANO** 参数并转换为点云。

**数据版权归 S2HGD 原作者所有，使用请遵循其许可。**

## 2. 文件与规模

`datasets/manohand.pt`：一个 `torch.save` 的 dict（float32）：

```
{
  "hands": Tensor[31292, 300, 5],   # 人手点云
  "objs" : Tensor[31292, 2048, 5],  # 被抓握的物体点云
}
```

- **样本数**：31,292。
- **划分**：`Rdataset.py` 将每个文件**前 500 行作为测试集，其余为训练集** → train **30,792** / test **500**。
- **坐标**：已归一化放缩（`xyz` 约在 `[-20, 20]` 量级）。

## 3. 通道定义

`hands` 与 `objs` 的每点为 5 通道：

| 通道     | 含义                                                                              |
| -------- | --------------------------------------------------------------------------------- |
| `0..2` | `xyz` 坐标                                                                      |
| `3`    | 手部件标签`{1,2,3,4,5}`                                                         |
| `4`    | 接触度`closeness ∈ (0,1)`，`= soft_distance(到另一物体的最短距离)`，越大越近 |

- 人手点：通道 4 = 该手点到被抓物体的贴近程度。
- 物体点：通道 3 = 该物体点最近的人手部件标签；通道 4 = 该物体点到手的贴近度。
- 训练时 `Rdataset.py` 会把通道 4 改写为类别标签（单类别为 0）作为模型输入约定；真实接触度只存在于 `.pt` 中。

## 4. 物体

- 物体来自 **S2HGD**，共 **1,668 个不同物体**。
- 每个样本的 `objs` 为该被抓握物体的点云（源网格采样 3000 点，数据集内取 **2048 点**）。

## 5. 人手点云的构建

人手点云由 **MANO** 生成并按手部件组织：

1. 由 MANO 参数（`pose 48 + shape 10 + translation 3 = 61` 维）经 `ManoLayer` 前向得到 **778 顶点**人手网格。
2. 按 MANO 顶点区间为每个顶点标注**手部件标签**。
3. **剔除手掌顶点**后，对各手指区域做**确定性抽点**，得到每样本 **300** 个手点，且**每类手点数固定**：

| 部件标签       | 每样本点数    |
| -------------- | ------------- |
| 1              | 69            |
| 2              | 60            |
| 3              | 56            |
| 4              | 56            |
| 5              | 59            |
| **合计** | **300** |

因此人手点云只包含 5 个手指部件类（标签 1..5），不含手掌。

## 6. 补全任务的条件点

在补全任务中，每样本的**已知（条件）点 = `objs` 的前 `svpoints`（默认 50）个点**，即**被抓握物体上离手最近的点**（`objs` 已按通道 4 接触度降序排列）。

模型在此条件下通过扩散逆向过程补全出 300 点人手云；`hands` 为对应的人手 ground-truth，用于对比与评估（如 Chamfer 距离）。

## 7. 生成流程

```
MANO 参数 (61)
   └─ ManoLayer 前向 ──► 778 顶点人手网格
                           └─ 手部件标注 ──► 去手掌 + 确定性抽点 ──► 300 点人手 (5 通道)
物体网格
   └─ 采样 3000 点 ──► 物体点云 (5 通道)
人手点云 + 物体点云
   └─ 接触图(closeness / 最近手部件) ──► 组装 ──► 去重 ──► 归一化
   └─ 物体抽点 3000 → 2048，并按接触度降序排列
   └─ 存为 manohand.pt
```

## 8. 中间产物（生成 manohand 的前身数据）

生成 `manohand.pt` 之前的中间数组位于 `data_hand/train/`（N=**26,744**）与 `data_hand/test/`（N=**4,548**），两目录结构一致，合计 **31,292**（= `manohand.pt` 行数）：

| 文件                     | 形状         | dtype | 含义                                                           |
| ------------------------ | ------------ | ----- | -------------------------------------------------------------- |
| `hand_config.npy`      | (N, 61)      | f32   | MANO 手参数：pose(48) + shape/beta(10) + translation(3)        |
| `hand_cor.npy`         | (N, 778, 3)  | f32   | 由`hand_config` 经 ManoLayer 前向得到的 778 顶点人手网格坐标 |
| `hand_tag.npy`         | (N, 778, 4)  | f32   | 778 顶点 + 每顶点手部件标签                                    |
| `hand_cpt.npy`         | (N, 778, 5)  | f32   | 778 顶点 + 部件标签 + 是否贴近物体（接触）                     |
| `hand_tag_part.npy`    | (N, 300, 4)  | f32   | 去手掌并确定性抽点得到的 300 个手指部件点                      |
| `hand_tag_part_tf.npy` | (N, 300, 4)  | f32   | 上者经姿态规范化（tf = transformed）后的 300 点                |
| `hand_near_tf.npy`     | (N, 50)      | f32   | 每样本 50 个"距物体最近"的手部支撑点                           |
| `obj.npy`              | (N, 3000, 3) | f64   | 被抓握物体点云（3000 点），与手一一配对                        |

其中 `hand_config → hand_cor → hand_tag → hand_tag_part` 为部件点云主线，`hand_cpt / hand_near_tf` 为接触/近物信息。

## 9. 引用

若使用本数据，请引用 S2HGD 原论文：

```bibtex
@article{wang2024single,
  title={Single-View Scene Point Cloud Human Grasp Generation},
  author={Wang, Yan-Kang and Xing, Chengyi and Wei, Yi-Lin and Wu, Xiao-Ming and Zheng, Wei-Shi},
  journal={arXiv preprint arXiv:2404.15815},
  year={2024}
}
```
