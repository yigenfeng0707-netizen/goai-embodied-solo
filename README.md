# GOAI Embodied Solo - 通用双臂协同操作参赛方案（赛事规则合规版）

> 基于 RoboDojo / X-Eval 平台的个人参赛方案，目标完成 12 项仿真双臂操作任务并晋级决赛。

- **团队**：GOAI Solo Builder（个人参赛）
- **赛道**：具身未来 Embodied Future / 赛题一：通用双臂协同操作能力测试
- **平台**：RoboDojo（仿真）+ XPolicyLab（policy 框架，30+ baseline）+ X-Eval（评测）
- **训练资源**：DSW GPU（魔搭社区）
- **开源协议**：Apache-2.0

---

## 合规声明（重要）

针对赛事方通知中明确的规则要求：

> "审核和正式评测期间，请保持同一服务在线，不要更换模型、动作类型或协议行为。"

本仓库已全面整改为 **单一 ckpt + 单一模型 + 单一动作类型 + 单一协议**
的合规架构，确保整个评测周期内：

| 维度 | 值 | 不可变性 |
|---|---|---|
| 模型 | **唯一一个** ACT checkpoint（cotrain recipe 产物） | 评测期 `nn.Parameter` 集合恒定 |
| 动作类型 | `joint`（14-dim 双臂关节） | 构造后不可变更 |
| 协议 | `ws`（XPolicyLab WebSocket） | 构造后不可变更 |
| 服务实例 | 单进程、单模型 | 评测期不重启、不替换 |

合规实现详见 [`policy/UnifiedACT/`](policy/UnifiedACT/)：

- 构造时只加载 **1 个** ACT checkpoint，不存在任务→ckpt 映射表
- `prepare_case(case_meta)` 仅记录日志用于诊断，**不切换/重载模型**
- 不存在 `task_models` / `fallback_map` / `active_model` 等多模型字段
- 提供 `compliance_self_check()` 方法，便于评测方动态核查合规不变量
- 完整测试覆盖：`tests/test_model_interface.py` 验证 24 个 case 切换不引发模型替换

---

## 目录

- [核心技术栈](#核心技术栈)
- [12 项评测任务](#12-项评测任务)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [提交方式](#提交方式)
- [时间节点](#时间节点)
- [开源协议](#开源协议)

---

## 核心技术栈

| 组件 | 说明 |
| --- | --- |
| **RoboDojo** | 仿真环境，提供双臂场景与 12 项任务的评测脚本 `scripts/robodojo.sh` |
| **XPolicyLab** | Policy 框架，内置 30+ baseline（ACT、Diffusion Policy、RDT 等） |
| **UnifiedACT** | 本仓库自研的 **合规单 ckpt 包装器**（替换旧 MultiACT） |
| **X-Eval** | 评测平台，仅评测 Generalization 维度，覆盖 24 配置 |
| **DSW GPU** | 魔搭社区 GPU 实例，用于训练 cotrain ckpt 与本地评测 |
| **HuggingFace Hub** | 通过 `hf download` 拉取 RoboDojo 资产与 GOAI-2026 数据集 |

---

## 12 项评测任务

X-Eval 仅评测 **Generalization 维度**。每个任务包含标准配置与 `_random` 随机化配置各 1 个，共 **24 个评测配置**。
**所有 24 配置由同一个 UnifiedACT ckpt 推理，无任务切换。**

| # | 任务名 | 标准配置 | 随机化配置 |
| --- | --- | --- | --- |
| 1 | 叠放碗 | `stack_bowls` | `stack_bowls_random` |
| 2 | 推 T 形块 | `push_T` | `push_T_random` |
| 3 | 装箱 | `pack_objects_into_box` | `pack_objects_into_box_random` |
| 4 | 折叠衣物 | `fold_clothes` | `fold_clothes_random` |
| 5 | 挂马克杯 | `hang_mugs` | `hang_mugs_random` |
| 6 | 扫块 | `sweep_blocks` | `sweep_blocks_random` |
| 7 | 倒液体入杯 | `pour_liquid_into_cup` | `pour_liquid_into_cup_random` |
| 8 | 做吐司 | `make_toast` | `make_toast_random` |
| 9 | 摆最大数 | `arrange_largest_number` | `arrange_largest_number_random` |
| 10 | 套娃按大小排序 | `sort_nesting_dolls_by_size` | `sort_nesting_dolls_by_size_random` |
| 11 | 收纳笔记本与耳机 | `store_laptop_and_headphones` | `store_laptop_and_headphones_random` |
| 12 | 叠方块 | `stack_blocks` | `stack_blocks_random` |

---

## 快速开始

> 以下流程严格遵循 X-Eval 平台官方提交指南（参考 https://xsparkai.com/goai-2026/ ）。

### 1. 环境搭建

```bash
cd scripts
bash setup_env.sh
```

脚本将依次完成：克隆 RoboDojo（含子模块）→ 安装 RoboDojo conda 环境 → 下载 GOAI 专用 Assets → 下载 GOAI-2026 数据集。

### 2. 准备 XPolicyLab（含 30+ baseline）

```bash
bash scripts/setup_xpolicylab.sh
```

### 3. 训练单一多任务 ckpt（合规关键步骤）

调用 XPolicyLab ACT 官方 cotrain recipe，在合并的 12 任务数据上联合训练，
产出 **唯一一个** 多任务 ckpt：

```bash
bash scripts/train_unified_act.sh 0      # GPU 0
```

或等价地：

```bash
cd XPolicyLab/policy/ACT
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

> 此步骤是合规关键：必须训练单一 ckpt（而非每任务一个 ckpt），
> 才能满足"评测期不更换模型"的硬性规则。

### 4. 接入 UnifiedACT 包装器

将 `policy/UnifiedACT/` 复制到 `XPolicyLab/policy/` 下：

```bash
cp -r policy/UnifiedACT XPolicyLab/policy/
```

并修改 `policy/UnifiedACT/deploy.yml::ckpt_dir` 指向步骤 3 的产物路径。

### 5. 部署 Policy Server

```bash
python XPolicyLab/setup_policy_server.py \
    --config_path policy/UnifiedACT/deploy.yml \
    --overrides host=0.0.0.0 port=19002
```

### 6. 冒烟测试

```bash
POLICY_DIR=XPolicyLab/policy/UnifiedACT \
POLICY_ENV=RoboDojo \
RUN_LABEL=cotrain \
ACTION_TYPE=joint \
bash scripts/smoke_test.sh
```

对 24 个配置各跑 1 episode，仅验证链路是否打通，不代表正式成绩。

### 7. 本地评测

```bash
POLICY_DIR=XPolicyLab/policy/UnifiedACT \
POLICY_ENV=RoboDojo \
RUN_LABEL=cotrain \
ACTION_TYPE=joint \
bash scripts/eval_local.sh
```

### 8. 正式提交

在 X-Eval 平台填写 Policy Server 的公网 WebSocket 地址（`wss://host:port`），平台将远程调用评测。

---

## 目录结构

```
goai-embodied-solo/
├── README.md                          # 项目主文档（本文件）
├── SUBMISSION_README.md               # 评测方先读
├── LICENSE                            # Apache-2.0
├── .gitignore
│
├── policy/                            # 【核心贡献】合规单 ckpt policy
│   └── UnifiedACT/                    # 单一多任务 ACT 包装器（合规版）
│       ├── __init__.py
│       ├── model.py                   # 单 ckpt 加载；prepare_case 不切换模型
│       ├── deploy.yml                 # 单一 ckpt 路径配置
│       └── README.md
│
├── configs/                           # 训练 / 评测配置（合规版）
│   ├── tasks.yaml                     # 12 任务 × 24 配置清单
│   ├── train_act.yaml                 # cotrain recipe 超参
│   └── eval_generalization.yaml       # Generalization 评测配置
│
├── scripts/                           # 环境搭建 / 部署 / 评测脚本
│   ├── setup_env.sh                   # RoboDojo 环境
│   ├── setup_xpolicylab.sh            # XPolicyLab policy 框架克隆
│   ├── train_unified_act.sh           # ★ 训练合规单一 ckpt（cotrain）
│   ├── deploy_policy_server.py        # Policy Server 部署入口（wss）
│   ├── smoke_test.sh                  # 24 配置各 1 episode 冒烟
│   ├── eval_local.sh                  # 24 配置完整评测
│   ├── check_and_recover.py           # 服务健康检查与恢复（可选）
│   ├── common_config.py               # 路径 / URL 占位配置
│   └── RESUME.md                      # 实例重启后的恢复指南
│
├── docs/                              # 设计文档
│   ├── architecture.md                # 整体架构（合规版）
│   ├── reproduction-guide.md          # 复现指南（合规版）
│   └── policy-server-deploy.md        # Policy Server 部署文档
│
└── tests/                             # 测试
    └── test_model_interface.py        # UnifiedACT 合规性测试（10 项）
```

> `RoboDojo/`、`XPolicyLab/` 为第三方子仓库，由脚本克隆，已在 `.gitignore` 中忽略，不进入本仓库。

---

## 提交方式

> **2026-08-12 整改更新**：方案已重构为单一 ckpt 架构（替换被官方拒绝的 MultiACT 多 ckpt 切换方案）。

- **提交形式**：源代码压缩包（本仓库打包）
- **评测维度**：仅 Generalization（24 配置）
- **次数**：最多 3 次评测，取最高分
- **晋级线**：>10 分晋级决赛
- **代码入口**：见 [SUBMISSION_README.md](./SUBMISSION_README.md)

### 评测流程（代码评测模式）

1. 评测方按 `SUBMISSION_README.md` 搭建 RoboDojo + XPolicyLab 环境
2. 将 `policy/UnifiedACT/` 接入 XPolicyLab 框架
3. 由 `policy/UnifiedACT/deploy.yml::ckpt_dir` 加载 **唯一一个** 多任务 ckpt
4. 在 24 配置上跑评测，**全程使用同一模型**，汇总成功率得出总分

### 备选部署（Policy Server 模式）

如评测方仍需 wss 远程评测，可参考 `docs/policy-server-deploy.md`：

```bash
python XPolicyLab/setup_policy_server.py \
    --config_path policy/UnifiedACT/deploy.yml \
    --overrides host=0.0.0.0 port=19002
```

---

## 时间节点

| 阶段 | 时间 | 说明 |
| --- | --- | --- |
| 初赛 | 2026-07-16 ~ 2026-08-20 | 提交源代码，评测方运行 |
| 评审 | 2026-08-21 ~ 2026-08-23 | 初赛结果评审 |
| 调试 | 2026-08-25 ~ 2026-09-20 | 决赛选手调试 |
| 决赛 | 2026-09-22 ~ 2026-09-23 | 现场决赛 |

---

## 实现状态

> 截至 2026-08-12 的实现进度（合规整改完成）。

- [x] **Policy 适配**：`policy/UnifiedACT/` 已实现单 ckpt 合规包装器
- [x] **合规测试**：`tests/test_model_interface.py` 10/10 PASS（验证无任务切换）
- [x] **配置文件**：`configs/` 全部对齐 cotrain 单 ckpt recipe
- [x] **部署模板**：`scripts/deploy_policy_server.py` 与 `policy/UnifiedACT/deploy.yml` 已就绪
- [x] **环境脚本**：`setup_env.sh` / `setup_xpolicylab.sh` / `smoke_test.sh` / `eval_local.sh`
- [x] **训练脚本**：`scripts/train_unified_act.sh` 一键产出合规单 ckpt
- [x] **旧方案清理**：`policy/MultiACT/` 已删除（多 ckpt 切换方案违反规则）
- [x] **统一 ckpt 训练完成**：RoboDojo-cotrain-arx_x5-joint epoch 3100（min_val_loss 0.216），已上传 ModelScope（gsym236998/goai-embodied-solo-ckpt）
- [ ] **完整评测得分**：依赖评测方运行确认

---

## 开源协议

本项目基于 [Apache License 2.0](./LICENSE) 开源，版权所有 © 2026 GOAI Solo Builder。

第三方组件（RoboDojo、XPolicyLab）遵循各自的开源协议。
