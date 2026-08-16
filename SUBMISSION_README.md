# 提交说明 — GOAI Embodied Solo（赛事规则合规版）

> 团队：GOAI Solo Builder（个人参赛）｜ 赛道：具身未来 / 赛题一
> 提交日期：2026-08-15（合规整改版） ｜ 提交形式：源代码压缩包

本文件面向大赛评测方，说明本压缩包的内容、依赖关系与运行方式。

---

## 0. 合规整改摘要（评测方先读）

### 0.1 整改背景

针对赛事组委会的通知：

> 经核查，您的方案在同一服务内依据不同任务切换对应 ACT checkpoint 进行推理。
> 该做法属于评测过程中更换模型，不符合规则要求："审核和正式评测期间，
> 请保持同一服务在线，不要更换模型、动作类型或协议行为。"

旧方案 `policy/MultiACT/` 已被官方认定为违规并删除。

### 0.2 整改结果（合规架构）

| 维度 | 旧 MultiACT（违规） | 新 UnifiedACT（合规） |
|------|--------------------|-----------------------|
| ckpt 数量 | 11 个（`task_ckpt_map`） | **1 个** |
| 任务切换 | 按 `action_case_id` 切换 `active_model` | **无切换** |
| Fallback | `DEFAULT_FALLBACK_MAP` 7 项 | **无** |
| 合规风险 | 评测期更换模型 | **单模型恒定** |
| 测试覆盖 | 接口测试 | **10 项合规性测试**（验证 24 case 不引发模型替换） |

合规实现详见 [`policy/UnifiedACT/`](policy/UnifiedACT/) 与
[`tests/test_model_interface.py`](tests/test_model_interface.py)。

---

## 1. 内容清单

```
goai-embodied-solo/
├── SUBMISSION_README.md              # 本文件（评测方先读）
├── README.md                         # 项目主文档（合规版）
├── LICENSE                           # Apache-2.0
├── .gitignore
│
├── policy/                           # 【核心贡献】合规单 ckpt policy
│   └── UnifiedACT/                   # 单一多任务 ACT 包装器（合规版）
│       ├── __init__.py
│       ├── model.py                  # 单 ckpt 加载；prepare_case 不切换模型
│       ├── deploy.yml                # 单一 ckpt 路径配置
│       └── README.md
│
├── configs/                          # 训练 / 评测配置（合规版）
│   ├── tasks.yaml                    # 12 任务 × 24 配置清单
│   ├── train_act.yaml                # cotrain recipe 超参
│   └── eval_generalization.yaml      # Generalization 评测配置
│
├── scripts/                          # 环境搭建 / 部署 / 评测脚本
│   ├── setup_env.sh                  # RoboDojo 环境（含子模块 + 资源 + 数据）
│   ├── setup_xpolicylab.sh           # XPolicyLab policy 框架克隆
│   ├── train_unified_act.sh          # ★ 训练合规单一 ckpt（cotrain）
│   ├── deploy_policy_server.py       # Policy Server 部署入口（wss）
│   ├── smoke_test.sh                 # 24 配置各 1 episode 冒烟
│   ├── eval_local.sh                 # 24 配置完整评测
│   ├── check_and_recover.py          # 服务健康检查与恢复（可选）
│   ├── common_config.py              # 路径 / URL 占位配置
│   └── RESUME.md                     # 实例重启后的恢复指南
│
├── docs/                             # 设计文档
│   ├── architecture.md               # 整体架构（合规版）
│   ├── reproduction-guide.md         # 复现指南（合规版）
│   └── policy-server-deploy.md       # Policy Server 部署文档
│
└── tests/                            # 测试
    └── test_model_interface.py       # UnifiedACT 合规性测试（10 项 PASS）
```

---

## 2. 核心贡献

本参赛方案的核心自研代码为 **`policy/UnifiedACT/`**，目标是严格满足赛事规则
"审核和正式评测期间不更换模型/动作类型/协议行为"。

### 2.1 单一 ckpt 包装器（`model.py`）

构造时加载 **唯一一个** ACT checkpoint；全部 24 个评测配置
（12 任务 × {标准, `_random`}）共用同一个模型实例。

接口对齐 XPolicyLab `ModelTemplate`：

- `__init__(model_cfg)`：加载 **唯一一个** ACT ckpt（由 `ckpt_dir` 指定）
- `prepare_case(case_meta)`：**仅记录** `case_meta` 用于日志
  - **不切换** 模型
  - **不重载** 模型
  - **不替换** 模型引用
- `update_obs(obs)` → `get_action()` → `reset()`：均委托给同一模型实例
- `compliance_self_check()`：返回合规自检字典（评测方可动态核查）

### 2.2 合规不变量（静态/动态核查）

UnifiedACT 实例 **不存在** 以下多模型字段（测试覆盖）：

- `task_models`（任务→模型字典）
- `task_ckpt_map`（任务→ckpt 路径字典）
- `fallback_map` / `DEFAULT_FALLBACK_MAP`（兜底映射）
- `active_model` / `active_task`（当前活跃模型/任务）

合规自检示例：

```python
>>> model.compliance_self_check()
{
    'policy_class': 'Model',
    'ckpt_dir': '/mnt/.../act-RoboDojo-cotrain/arx_x5-100-joint',
    'ckpt_name': 'cotrain',
    'action_type': 'joint',
    'model_id': 140234567890,
    'case_count': 24,
    'forbidden_fields_present': [],
    'rule_compliant': True
}
```

---

## 3. 依赖关系

### 3.1 第三方框架（评测方应已具备）

本方案运行在官方平台之上，依赖：

| 框架 | 来源 | 说明 |
|---|---|---|
| **RoboDojo** | https://github.com/RoboDojo-Benchmark/RoboDojo | 仿真环境，提供 12 项任务与评测脚本 |
| **XPolicyLab** | https://github.com/XPolicyLab/XPolicyLab | Policy 框架（含 `ModelTemplate`、`PolicyServer`、`utils.process_data`、ACT 完整实现含 `detr/`） |

> 完整运行必须克隆官方仓库：
>
> ```bash
> git clone https://github.com/XPolicyLab/XPolicyLab.git
> pip install -e XPolicyLab
> ```

### 3.2 UnifiedACT 接入方式

将 `policy/UnifiedACT/` 目录复制到官方 `XPolicyLab/policy/` 下：

```bash
cp -r policy/UnifiedACT XPolicyLab/policy/
```

`UnifiedACT/model.py` 通过延迟导入引用 `XPolicyLab.policy.ACT.model.Model`，
因此只需 XPolicyLab 可正常 import 即可。

---

## 4. 运行步骤

### 4.1 环境搭建

```bash
# 1. RoboDojo（仿真环境与数据）
bash scripts/setup_env.sh

# 2. XPolicyLab（policy 框架 + 30+ baseline）
bash scripts/setup_xpolicylab.sh
```

### 4.2 训练单一多任务 ckpt（合规关键步骤）

```bash
# 调用 XPolicyLab ACT 官方 cotrain recipe，产出 **唯一一个** 多任务 ckpt
bash scripts/train_unified_act.sh 0     # GPU 0
```

产物路径示例：

```
XPolicyLab/policy/ACT/checkpoints/RoboDojo-cotrain-arx_x5-joint-0/
```

> **合规关键**：必须使用单一 ckpt（cotrain recipe），不得退回到
> 按任务独立训练 + 按任务切换 ckpt 的多模型方案。

### 4.3 部署 Policy Server

```bash
python XPolicyLab/setup_policy_server.py \
    --config_path policy/UnifiedACT/deploy.yml \
    --overrides host=0.0.0.0 port=19002
```

`deploy.yml` 中 `ckpt_dir` 默认指向 DSW 上的 cotrain recipe 产物路径：

```
/mnt/workspace/RoboDojo/ckpt/ACT/act-RoboDojo-cotrain/arx_x5-100-joint
```

评测方环境若不同，请按需修改 `ckpt_dir`，或通过环境变量覆盖。

### 4.4 评测

```bash
# 冒烟（24 配置各 1 episode）
POLICY_DIR=XPolicyLab/policy/UnifiedACT \
POLICY_ENV=RoboDojo \
RUN_LABEL=cotrain \
ACTION_TYPE=joint \
bash scripts/smoke_test.sh

# 完整评测（按 EVAL_NUM 个 episode）
POLICY_DIR=XPolicyLab/policy/UnifiedACT \
POLICY_ENV=RoboDojo \
RUN_LABEL=cotrain \
ACTION_TYPE=joint \
bash scripts/eval_local.sh
```

---

## 5. Checkpoint 说明

本方案只使用 **一个** checkpoint（cotrain recipe 产物）：

| 字段 | 值 |
|------|---|
| ckpt 目录 | `policy/UnifiedACT/deploy.yml::ckpt_dir` |
| 训练 recipe | `bash train.sh RoboDojo cotrain arx_x5 joint 0 0` |
| 任务覆盖 | 全部 12 任务（合并数据联合训练） |
| 评测使用 | 全部 24 配置（同一 ckpt，不切换） |

> Checkpoint 文件本身不随源码打包（体积大，单文件约 320MB）。
> 已提供公开下载（无需评测方自行训练）：
> - ModelScope 仓库：https://modelscope.cn/models/gsym236998/goai-embodied-solo-ckpt
> - 单文件直链：https://modelscope.cn/models/gsym236998/goai-embodied-solo-ckpt/resolve/master/policy_epoch_3100_seed_0.ckpt
> - 命令行：`modelscope download gsym236998/goai-embodied-solo-ckpt policy_epoch_3100_seed_0.ckpt`
>
> 加载方式：将 ckpt 放入某目录后，在 `policy/UnifiedACT/deploy.yml` 设置
> `ckpt_dir: <该目录>`、`ckpt_name: epoch_3100_seed_0`（即加载 `policy_epoch_3100_seed_0.ckpt`）。
> 此 ckpt 为 **唯一一个**，24 配置共用，评测期不切换。
>
> 同一仓库内已一并提供 **`dataset_stats.pkl`**（与 ckpt 同目录，用于保证推理与训练阶段数据统计参数一致）：
> - 单文件直链：https://modelscope.cn/models/gsym236998/goai-embodied-solo-ckpt/resolve/master/dataset_stats.pkl
> - 命令行：`modelscope download gsym236998/goai-embodied-solo-ckpt dataset_stats.pkl`
> - 评测时请将 `dataset_stats.pkl` 与 `policy_epoch_3100_seed_0.ckpt` 放在同一目录下。

---

## 6. 协议

- 本仓库代码：**Apache License 2.0**
- 第三方组件（RoboDojo、XPolicyLab、ACT）：遵循各自开源协议

---

## 7. 合规审查清单（评测方参考）

- [x] **单一 ckpt**：`deploy.yml` 只声明一个 `ckpt_dir`，无任务→ckpt 映射表
- [x] **单一模型**：`model.py` 只持有 `self.model` 一个实例引用
- [x] **prepare_case 不切换**：测试覆盖 24 个 case 调用后 `id(self.model)` 恒定
- [x] **无禁用字段**：实例不存在 `task_models` / `fallback_map` / `active_model`
- [x] **单一动作类型**：`action_type=joint` 构造后不可变更
- [x] **单一协议**：`protocol=ws` 构造后不可变更
- [x] **合规自检**：提供 `compliance_self_check()` 方法供动态核查
- [x] **测试覆盖**：`tests/test_model_interface.py` 10 项 PASS

---

## 8. 联系方式

如评测过程中遇到问题，可通过提交邮件回复联系参赛者。

---

**致谢**：感谢 RoboDojo 团队开源仿真环境与 ACT baseline，本方案的所有训练数据
与 baseline 均来自官方提供的资源。设计依据为 XPolicyLab ACT 官方 README：

> "task_name is only the evaluation task; multi-task checkpoints can be
> evaluated on different tasks without renaming the checkpoint directory."
