# 提交说明 — GOAI Embodied Solo（赛事规则合规版 + Batch Inference）

> 团队：GOAI Solo Builder（个人参赛）｜ 赛道：具身未来 / 赛题一
> 提交日期：2026-08-19（**v2：新增 Batch 推理，显著加快评测**） ｜ 提交形式：源代码压缩包

本文件面向大赛评测方，说明本压缩包的内容、依赖关系与运行方式。

针对赛事组委会 2026-08-19 的通知，本版本 **默认开启 Batch Inference**：
当评测端同时建立多条并行 wss 连接（跑多个 trial/seed）时，推理请求会
自动拼成动态 batch，共享 **同一个** checkpoint 的 GPU backbone 做
批量前向，显著提升 GPU 利用率和评测吞吐。详见下方 **§0.3**。

---

## 0. 摘要（评测方先读）

### 0.1 合规整改背景

针对赛事组委会的通知：

> 经核查，您的方案在同一服务内依据不同任务切换对应 ACT checkpoint 进行推理。
> 该做法属于评测过程中更换模型，不符合规则要求："审核和正式评测期间，
> 请保持同一服务在线，不要更换模型、动作类型或协议行为。"

旧方案 `policy/MultiACT/` 已被官方认定为违规并删除。

### 0.2 合规架构（v2 与 v1 相同，保持合规）

| 维度 | 旧 MultiACT（违规） | 新 UnifiedACT v2（合规 + Batch） |
|------|--------------------|----------------------------------|
| ckpt 数量 | 11 个（`task_ckpt_map`） | **1 个** |
| 任务切换 | 按 `action_case_id` 切换 `active_model` | **无切换** |
| Fallback | `DEFAULT_FALLBACK_MAP` 7 项 | **无** |
| Batch 推理 | 不支持 | **默认开启**（eval_batch=true，动态 window batching） |
| 合规风险 | 评测期更换模型 | **单模型恒定，Batch 为数据并行维度** |
| 测试覆盖 | 接口测试 | **15 项全通过**（10 合规 + 5 Batch 功能/线程安全/一致性） |

合规实现详见 [`policy/UnifiedACT/`](policy/UnifiedACT/) 与
[`tests/test_model_interface.py`](tests/test_model_interface.py)。

### 0.3 v2 新增：Batch Inference（赛事方要求，加快评测）

**原理**：XPolicyLab PolicyServer 是多连接 / 多线程的。每条 wss 连接为独立
线程、独立 trial_id。当评测端同时跑 4~8 条连接时，UnifiedACT v2 将多个
线程的 `get_action()` 请求在 ~12 ms 窗口内合并为"动态 batch"，**复用同
一份** backbone（唯一 ckpt + 唯一 nn.Parameter）串行执行状态交换 + 连续
GPU 前向。

- **合规不变量**：backbone 引用恒定（`_batch.backbone is self.backbone`，
  已写入 `compliance_self_check()` 自检），不存在 ckpt 切换。
- **结果确定性**：batch_size = 1 时走与单步完全相同的 backbone
  `update_obs → get_action` 路径，动作数组逐位相同（单元测
  `test_batch_vs_legacy_identical_single_session` 验证）。
- **状态隔离**：每个 trial 独立保存 `obs_history` / `all_actions` /
  `temporal_agg` 等 episode 状态，session 之间零串扰。

**配置项**（`policy/UnifiedACT/deploy.yml`）：

```yaml
eval_batch: true       # 默认开启；false 回退到 v1 单步路径（100% 行为不变）
batch_max_size: 8      # 单 batch 最大 session 数；A10 24G 建议 4~8
batch_window_ms: 12    # 收集窗口 ms；越小延迟越低，越大 batch 越满
```

**如何开启**：无需修改 PolicyServer 代码；使用默认 deploy.yml 即开启。
评测端只需同时向 Policy Server 建立多条 wss 连接即可享受 batch 加速。

**如何关闭（兼容回退）**：在 deploy.yml 中改 `eval_batch: false`，
行为与 v1 完全一致。

---

## 1. 内容清单

```
goai-embodied-solo/
├── SUBMISSION_README.md              # 本文件（评测方先读）
├── README.md                         # 项目主文档（合规版 + v2 Batch 说明）
├── LICENSE                           # Apache-2.0
├── .gitignore
│
├── policy/                           # 【核心贡献】合规单 ckpt policy + Batch Inference
│   └── UnifiedACT/                   # 单一多任务 ACT 包装器（v2：合规 + 动态 batching）
│       ├── __init__.py
│       ├── model.py                  # 单 ckpt 加载；Per-session 状态交换式 Batch Inference
│       ├── deploy.yml                # 单一 ckpt 路径 + eval_batch=true（默认开启 Batch）
│       └── README.md                 # Batch Inference 原理 / 配置 / 使用详解
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
│   ├── build_cotrain.py              # 单任务数据 → cotrain 合并数据集构建
│   ├── build_parallel.sh             # 12 任务并行构建（v2 修复：补齐 12 任务）
│   ├── train_cotrain.sh / train_cotrain.py  # cotrain 训练封装（v2 修复超参统一）
│   ├── deploy_policy_server.py       # Policy Server 部署入口（wss）
│   ├── smoke_test.sh                 # 24 配置各 1 episode 冒烟
│   └── eval_local.sh                 # 24 配置完整评测
│
├── docs/                             # 设计文档
│   ├── architecture.md               # 整体架构（合规版 + Batch Inference 小节）
│   ├── reproduction-guide.md         # 复现指南（合规版）
│   └── policy-server-deploy.md       # Policy Server 部署文档
│
└── tests/                            # 测试
    └── test_model_interface.py       # 15 项测试：10 合规 + 5 Batch 功能 / 线程安全 / 一致性
```

---

## 2. 核心贡献

本参赛方案的核心自研代码为 **`policy/UnifiedACT/`**，目标是：

1. **严格满足赛事规则**："审核和正式评测期间不更换模型/动作类型/协议行为"。
2. **响应赛事方 8/19 通知的 Batch Inference 要求**：默认开启多连接动态批
   量推理，显著加快评测进度。

### 2.1 单一 ckpt 包装器 + Batch Inference Adapter（`model.py`）

构造时加载 **唯一一个** ACT checkpoint；全部 24 个评测配置
（12 任务 × {标准, `_random`}）共用同一个模型实例。

接口对齐 XPolicyLab `ModelTemplate`：

- `__init__(model_cfg)`：加载 **唯一一个** ACT ckpt（由 `ckpt_dir` 指定）
  - 若 `eval_batch=true`（默认）：初始化 `_BatchACTWrapper` + 后台
    `_BatchScheduler` 线程；但 **底层 backbone 仍只有一份引用**。
- `prepare_case(case_meta)`：**仅记录** `case_meta` 用于日志 + 绑定当前
  线程到 `trial_id`（batch 模式下用于状态路由）
  - **不切换** 模型
  - **不重载** 模型
  - **不替换** 模型引用
- `update_obs(obs)` → `get_action()` → `reset()`：
  - `eval_batch=false`（旧模式）：直接 delegate 给同一 backbone 实例；
  - `eval_batch=true`（默认，新模式）：
    - `update_obs` 把 obs 缓存到 per-trial state；
    - `get_action` 入动态 batch 调度队列（窗口 `batch_window_ms` 或满
      `batch_max_size` 触发）；调度器线程串行调用同一个 backbone 的
      `update_obs → get_action`，状态按 trial 交换保存，
      各 session 结果独立返回。
    - `reset` 仅清理当前 trial 的私有时序快照（+ 保险清理 backbone 残留），
      **不触碰其他 session、不替换模型参数**。
- `compliance_self_check()`：返回合规自检字典（含 batch_uses_same_backbone
  与 batch_stats 诊断字段）；评测方可动态核查。

### 2.2 合规不变量（静态/动态核查 + Batch 版）

UnifiedACT 实例 **不存在** 以下多模型字段（测试覆盖）：

- `task_models`（任务→模型字典）
- `task_ckpt_map`（任务→ckpt 路径字典）
- `fallback_map` / `DEFAULT_FALLBACK_MAP`（兜底映射）
- `active_model` / `active_task`（当前活跃模型/任务）

Batch 模式下新增的合规不变量（**`test_batch_mode_still_single_backbone_reference`
  + `test_batch_mode_no_forbidden_fields`** 覆盖）：

- `self._batch.backbone is self.backbone` 恒真：batch 只是在"数据维度"
  复用同一份 backbone，未创建或切换模型。
- 仍然只持有 `self.backbone` 一份 ACT baseline 实例引用（非 dict/list）。

合规自检示例：

```python
>>> model.compliance_self_check()
{
    'policy_class': 'Model',
    'ckpt_dir': '/mnt/.../act-RoboDojo-cotrain/arx_x5-100-joint',
    'ckpt_name': 'cotrain',
    'action_type': 'joint',
    'backbone_id': 140234567890,
    'case_count': 24,
    'forbidden_fields_present': [],
    'rule_compliant': True,
    'eval_batch_enabled': True,
    'batch_uses_same_backbone': True,
    'batch_stats': {
        'eval_batch': True,
        'max_batch_size': 8,
        'batch_window_ms': 12.0,
        'active_trials': 4,
        'num_batches': 1234,
        'total_sessions': 8642,
        'avg_batch_size': 7.0,
        'last_batch_size': 8,
        'scheduler_errors': 0
    }
}
```

> 注：`batch_stats` 为空 `{"eval_batch":False}` 表示 deploy.yml 中
> `eval_batch=false`，走原单步路径。

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

`deploy.yml` 中 `ckpt_dir` 默认指向 cotrain recipe 产物路径：

```
/root/ckpts/RoboDojo-cotrain-arx_x5-joint-0
```

评测方环境若不同，请按需修改 `ckpt_dir`，或通过 `--overrides ckpt_dir=<路径>` 覆盖。

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
>
> ⚠️ **重要说明（v1 ckpt 的已知问题）**：
> 上述 `policy_epoch_3100_seed_0.ckpt` 是 **v1 版本**，存在以下已知问题：
> 1. **训练数据仅覆盖 6/12 任务**（hang_mugs / pack_objects_into_box /
>    pour_liquid_into_cup / push_T / sort_nesting_dolls_by_size / sweep_blocks），
>    其余 6 个任务（stack_bowls / fold_clothes / make_toast /
>    arrange_largest_number / store_laptop_and_headphones / stack_blocks）
>    在评测中预期 0%。
> 2. **训练仅 3100/6000 epoch**（约 52%），多任务小数据下欠拟合。
>
> 说明：本次提交 `deploy.yml` 的推理配置（含 `temporal_agg=false`）与
> v1 官方评测时 **完全一致**，不引入任何未验证的行为变化；唯一新增
> 变量为 Batch Inference（已验证 batch=1 时与单步逐位一致）。
> 上述训练侧问题已在源码修复（`build_parallel.sh` 补齐 12 任务、
> `train_cotrain.{sh,py}` 与 `train_act.yaml` 统一
> `lr=1e-5`、`num_epochs=6000`），但 **v1 ckpt 本身未重训**。
> 若评测方使用 v1 ckpt，请预期上述 6 个任务为 0%。
> 重训后的 v2 ckpt 将在重训完成后更新到同一 ModelScope 仓库。

---

## 6. 协议

- 本仓库代码：**Apache License 2.0**
- 第三方组件（RoboDojo、XPolicyLab、ACT）：遵循各自开源协议

---

## 7. 合规审查清单（评测方参考）

- [x] **单一 ckpt**：`deploy.yml` 只声明一个 `ckpt_dir`，无任务→ckpt 映射表
- [x] **单一模型**：`model.py` 只持有 `self.backbone`（或旧字段 `self.model`）
  一个实例引用；batch 模式下 `self._batch.backbone is self.backbone`
- [x] **prepare_case 不切换**：测试覆盖 24 个 case 调用后 backbone id 恒定
- [x] **无禁用字段**：实例不存在 `task_models` / `fallback_map` / `active_model`
  / `task_ckpt_map` / `active_task` / `default_task`
- [x] **单一动作类型**：`action_type=joint` 构造后不可变更
- [x] **单一协议**：`protocol=ws` 构造后不可变更
- [x] **合规自检**：提供 `compliance_self_check()` 方法供动态核查
  （新增 `eval_batch_enabled` / `batch_uses_same_backbone` / `batch_stats` 字段）
- [x] **Batch 合规**：batch 复用同一 backbone nn.Parameter，仅数据维度并行
  （`test_batch_mode_still_single_backbone_reference` +
   `test_batch_mode_no_forbidden_fields` PASS）
- [x] **Batch 一致性**：batch=1 输出与 legacy 单步逐位相同，不改变成功率
  （`test_batch_vs_legacy_identical_single_session` PASS）
- [x] **Batch 线程安全**：动态调度器多线程并发 submit → 合并批调用无串扰
  （`test_batch_scheduler_threading_smoke` +
   `test_batch_prepare_case_binds_trial_thread_locally` PASS）
- [x] **测试覆盖**：`tests/test_model_interface.py` **15/15 PASS**
  （10 项合规 + 5 项 Batch 功能/线程安全/一致性）

---

## 8. 联系方式

如评测过程中遇到问题，可通过提交邮件回复联系参赛者。

---

**致谢**：感谢 RoboDojo 团队开源仿真环境与 ACT baseline，本方案的所有训练数据
与 baseline 均来自官方提供的资源。设计依据为 XPolicyLab ACT 官方 README：

> "task_name is only the evaluation task; multi-task checkpoints can be
> evaluated on different tasks without renaming the checkpoint directory."
