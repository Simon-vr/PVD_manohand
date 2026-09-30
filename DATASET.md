# manohand 数据集说明（Human-Hand Grasp Point-Cloud Dataset）

> 本文档描述 `datasets/manohand.pt` 的**来源、生成流程与元数据**，用于数据溯源与公开。
> 生成管线主要记录在原始处理脚本 `data_hand/dataprocess.py`（已归档至 `_deprecated_data/`）以及 `datasets/Rdataset.py`。

## 1. 任务与用途

面向 **人手点云补全**：给定被抓握物体的局部点云条件，用扩散模型补全出**人手点云**。
数据集本身是人-物抓握（hand–object grasp）的成对点云：每个样本 = 一只人手（以 MANO 网格表示，源自真实采集的注册） + 一个被抓握的物体。

## 2. 数据来源 / 溯源链

原始数据来自 **S2HGD 数据集**（*Single-view Scene-to-Human Grasp Dataset*）：

> Yan-Kang Wang, Chengyi Xing, Yi-Lin Wei, Xiao-Ming Wu, Wei-Shi Zheng.
> **Single-View Scene Point Cloud Human Grasp Generation.** arXiv:2404.15815, 2024.

S2HGD 包含约 **99,000** 个**单物体、单视角场景点云**，覆盖 **1,668 个不同物体**，每个点云配 **一个人手抓取标注**（真实采集的拟合）。本仓库把这些人手抓取**拟合为 MANO 参数**并转为点云，构建 `manohand.pt`。**数据版权归原作者所有，请遵循其许可。**

> ⚠️ **只有“数据”来自 S2HGD；本仓库/本数据集的处理代码与那篇论文的方法无关**。

> 说明：MANO 手参数是对**真实手型/姿态的注册拟合**（手侧由 MANO 合成，物体侧为真实物体网格采样）。

后处理链路（把手/物体处理成 manohand.pt）：

```
原始真实手物数据(他人采集) ——本仓库未含——
  └─ MANO 注册/拟合 → 人手参数 (61维: pose 48 + shape/beta 10 + translation 3)
  └─ 物体点云/网格
人手参数 → ManoLayer 前向 → 778 顶点人手 mesh + 关节点
两者 → 计算接触/部件标注 → 抽点 → 归一化 → manohand.pt
```

涉及的关键后处理文件（均在 `_deprecated_data/data_hand/`）：

| 文件 | 内容 |
|---|---|
| `hand_config.npy` / `hand_all.npy` / `hand_uniq.npy` | MANO 手参数（61 维），全量/唯一手 |
| `hand_cor.npy` | 人手 778 顶点 mesh 坐标 |
| `hand_tag.npy` / `hand_tag_part.npy` / `hand_tag_part_tf.npy` | 人手点 + 手部件标签 |
| `hand_cpt.npy` | 手点接触(距物体最近)标签 |
| `hand_near_tf.npy` | 手部"近物"支撑点索引 |
| `obj.npy` / `obj_uniq.npy` / `obj_all.npy` | 物体点云（3000 点/物体） |
| `assets/mano_v1_2` | MANO 模型资产 |
| `assets/hand_palm_full.txt` | 手掌顶点索引集（407 个）——**未被生成代码直接读取**（dataprocess 里的手掌/部件顶点区间是硬编码），作用待确认 |
| `assets/yodaobject_cat.json` | 一份 33 类物体名表——**未被生成代码引用**，与本数据的对应关系未证实，仅作参考保留 |

> ⚠️ 追溯口径：仓库内**不含**外部抓握数据集或物体的原始版权网格，物体以处理后点云给出；物体→类别的关联**没有在数据或代码中标注**，下列物体描述仅为已核实到的事实，非版权来源背书。

## 3. 数量与划分（manohand.pt）

`manohand.pt` 是一个 `dict`：

```
{
  "hands": Tensor[31292, 300, 5],   # 人手点云
  "objs" : Tensor[31292, 2048, 5],  # 被抓握物体点云
}
```

- **样本数 31292**，由 31292 个**唯一人手**（`hand_uniq.npy`，61 维去重后）配对各自物体而来。
- 更大的超集：`hand_all.npy` / `obj_all.npy` 为 **94735**（未去重/更多抓握），manohand 取自去重后的 31292。
- **训练/测试划分**（`Rdataset.py`）：每类**前 `testnum=500` 行为测试，其余为训练** → train **30792** / test **500**（单类别下按文件行序切，无随机 shuffle 的文件级划分）。
- 数值为 **float32**，坐标已归一化放缩（`xyz` 约在 `[-20, 20]` 量级，已减物体中心并放大）。
- 与机器人手(shadowhand/allegro/ezgripper…)不同，本文件**只含人 hand（manohand）**。

## 4. 物体（object）

- 每个样本的 `objs` 为该被抓握物体的**点云**（源网格 3000 点，数据集内取 **2048 点**）。
- 去重统计（按网格逐点哈希）：
  - **物体来源：S2HGD 数据集，共 1,668 个不同物体**（论文口径；见 §2）。
  - 按网格逐点哈希精确去重得到的行内唯一网格数 ≈2651，**多于 1,668**，原因：同一物体在不同抓握中的姿态/对齐不同，逐点严格相等会过计。**以 1,668 为准**；哈希计数仅说明配对物体网格数在同一量级。
  - `obj_all.npy`（94735 超集）：对应的物体集合相同。

## 5. 人手点云的筛选与分类（重点）

人手每个样本 300 点、5 通道，且**按 5 个手指部件固定配额**选取——每只手点数恒定：

| 部件标签 ch3 | 手点数/每样本 | 说明 |
|---|---|---|
| 1 | 69 | 手指/部件 1 |
| 2 | 60 | 手指/部件 2 |
| 3 | 56 | 手指/部件 3 |
| 4 | 56 | 手指/部件 4 |
| 5 | 59 | 手指/部件 5 |
| **合计** | **300** | |

即手点云由 **MANO 778 顶点人手**按手部件顶点区间的**确定性抽点**得到（每部件抽到固定配额）。从 `manohand.pt` 实测仅含标签 1..5、无标签 0，故推断**手掌(0)类被剔除**、只保留 5 个手指部件——这一“去掌”结论与 `dataprocess.py` 里手掌顶点被硬编码剔除的逻辑一致。

部件→具体手指的对应关系由处理代码中 MANO **顶点索引区间**标定（`dataprocess.py` 里 middle/ring/pinky 各为一段顶点区间；index 一段；余下/部件 1 应为 thumb）。**因 MANO 顶点编号与手型版本相关，确切的“标签号→手指名”映射请以所用 MANO 版本核对，此处不臆断**。`assets/hand_palm_full.txt` 未被代码直接读取，仅作参考备份。

## 6. 每点 5 通道含义

对所有 `hands` 与 `objs`：

```
ch0..2 : xyz
ch3    : 手部件标签 {1,2,3,4,5}
ch4    : 接触度 closeness(0..1)，= soft_distance(到另一物体的最短距离)，越大越近
```

- 人手点：ch4 = 该手点到被抓物体的贴近度。
- 物体点：ch3 = 该物体点**最近的手部件标签**，ch4 = 该物体点到手的贴近度。

> 注意：`Rdataset.py` 载入时会把 `ch4`（末通道）**改写为类别标签 0** 作为模型输入约定；真正的接触度只在原始 `.pt` 里。物体着色可视化时用的是 `.pt` 中的真实 ch4。

## 7. 补全的"条件点"语义

`mode='cmap'` 下，每样本的条件（已知）点 = `objs` 的**前 `svpoints`(=50) 个点**，表示"被抓握物体上最贴近手的点"。
实现上已把每个样本的 `objs` 按 ch4(接触度)**降序重排**，使前 50 点即离手最近的点；其余物体点随机抽样到 2048 后**不再打乱**（排序脚本 `sort_manohand.py` 归档在 `_deprecated_code/`）。

- 任务：给定这 50 个物体点（固定不更新），扩散模型补全剩余 250 点（人手部件点），输出 300 点手云。
- 手点 `hands` 为 ground-truth 人手云，用于对比/评估（如 Chamfer）。

## 8. 生成流程小结（dataprocess 逻辑）

1. MANO 参数 → 人手 mesh(778) + 关节；做姿态规范/归一化（`pose_transfer` 等）。
2. 部件标注：按顶点区间给每手点标手部件类；去手掌。
3. 确定性抽点到 300（每部件固定配额，见 §5）。
4. 物体 mesh 取 3000 点。
5. 计算接触图：物体↔手逐点最短距离 → closeness + 最近手部件标签。
6. 组装 5 通道 `hand_pc`(300) / `obj_pc`(3000)，并按接触排序。
7. 去重（唯一手，31292）；train/test 划分（testnum=500）。
8. xyz 归一化放缩。
9. obj 抽点 3000→2048；obj 按接触度**降序重排**（保证前 50 为最近点）。

## 8. 中间产物 train/ 与 test/（生成 manohand 的前身数据）

`_deprecated_data/data_hand/train/`(N=**26744**) 与 `test/`(N=**4548**) 是建成 `manohand.pt` 之前**按物体分组的中间产物**（train+test 合计 31292 = manohand 行数）。两目录文件一致，仅行数不同：

| 文件 | 形状 | dtype | 含义 / 产生流程 |
|---|---|---|---|
| `hand_config.npy` | (N, 61) | f32 | **输入手参数**：MANO 参数 = pose(48)+shape/beta(10)+translation(3)。源自真实人手抓握拟合的 MANO 参数 |
| `hand_cor.npy` | (N, 778, 3) | f32 | 由 `hand_config` 经 **ManoLayer 前向**得到的 **778 顶点人手网格**坐标 |
| `hand_tag.npy` | (N, 778, 4) | f32 | 778 顶点 + **每顶点手部件标签**(ch3)：(xyz + 部件类)，部件按 MANO 顶点区间划分 |
| `hand_cpt.npy` | (N, 778, 5) | f32 | 手 778 顶点**接触**：xyz + 部件标签 + 是否贴近物体(距物体 < 阈值) |
| `hand_tag_part.npy` | (N, 300, 4) | f32 | 由 `hand_tag` **去手掌并确定性抽点**得 **300 个手指部件点**(xyz + 部件类)，即每部件固定配额(69/60/56/56/59)的来源 |
| `hand_tag_part_tf.npy` | (N, 300, 4) | f32 | `hand_tag_part` 经**姿态规范化(pose_transfer, tf=transformed)**后的 300 点变体；结构相同 |
| `hand_near_tf.npy` | (N, 50) | f32 | 每样本 **50 个“距物体最近”的手部支撑点**（规范化 tf 系），用于条件/支撑点 |
| `obj.npy` | (N, 3000, 3) | f64 | 被抓握**物体点云**(3000 点)，与手一一配对 |

处理关系：`hand_config → ManoLayer → hand_cor(778) → 部件标注 hand_tag → 去掌抽点 hand_tag_part →(tf 规范化)→ hand_tag_part_tf`；`hand_cpt/hand_near_tf` 为接触/近物信息。`dataprocess.py` 再把这批 (N=31292) 手/物点云加工成 5 通道 `manohand.pt`。

> ⚠️ 其中 `hand_tag_part` 与 `hand_tag_part_tf` 哪一版实际进入当时某次 manohand 构建、以及 hand_near_tf 的精确索引语义，取决于当时 dataprocess main 的分支；此处按脚本用途给出一般含义，精确口径若需保证请以原始运行脚本为准。

## 9. 已知口径 / 注意

**可信度分级（避免误导）**：
- ✅ **从数据/代码直接证实**：样本数 31292；train30792/test500；手点每部件固定配额(69/60/56/56/59，和300)；5 通道定义；条件点=obj 前 50 最近点；手为 MANO 参数生成。
- ✅ **数据来源（由数据集所有者提供）**：原始数据为 **S2HGD**（Wang et al., arXiv:2404.15815），~99k 单视角单物体点云 / 1,668 物体，每点云一个人手抓取标注。
- ✅ **代码关系**：本仓库代码与该论文**无关**，是独立的 PVCNN2+扩散 人手点云补全实现。
- ⚠️ **推断（与代码一致但未在数据中标注）**：手点只含 5 手指部件、剔除手掌(0)。
- ❓ **未确认 / 待补充**：`yodaobject_cat.json` 与本数据物体的对应关系（该文件未被代码引用）；部件标签→具体手指名的精确保留。物体版权请遵循 S2HGD 原始许可。

- 坐标系与 MANO 对齐方式以生成脚本为准；如需不同归一化请自行处理。
- 部件类到手指名的精确保留以 MANO 顶点编号为准。
- `hand_all/obj_all`(94735) 为更大超集；`obj_all.npy` 只含 `xyz`(3 通道)，未带标签/接触。
- 物体原始版权网格不入库；如需标注/网格请另行获取。

## 10. 复现相关脚本

- 加载：`datasets/Rdataset.py`
- 生成参考：`_deprecated_data/data_hand/dataprocess.py`
- obj 排序：`_deprecated_code/sort_manohand.py`
- 训练/推理/可视化入口见 `README.md`。
