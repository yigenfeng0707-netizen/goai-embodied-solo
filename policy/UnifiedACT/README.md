# UnifiedACT — 单一多任务 ACT Policy 推理适配器（赛事规则合规版）

> 适用赛道：GOAI 2026 具身未来 / 赛题一：通用双臂协同操作能力测试
> 合规依据：赛事规则要求"审核和正式评测期间，请保持同一服务在线，
>           不要更换模型、动作类型或协议行为。"

`UnifiedACT` 是 XPolicyLab ACT baseline 的轻量包装层，用于满足赛事
"评测期不更换模型"的硬性规则。**全部 24 个评测配置（12 任务 ×
{标准, `_random`}）共用同一个 ACT checkpoint 与同一个模型实例**，
不存在任何任务切换、checkpoint 切换或 fallback 映射逻辑。

## 合规设计

| 规则要求 | UnifiedACT 实现 |
|---------|----------------|
| 不更换模型 | 构造时加载 **1 个** ACT checkpoint；`prepare_case` 不切换/重载模型 |
| 不更换动作类型 | `action_type` 由 `deploy.yml` 静态声明，构造后不可变 |
| 不更换协议 | `protocol` 由 `deploy.yml` 静态声明，构造后不可变 |
| 同一服务在线 | 单进程、单模型实例，无任务级模型替换 |

合规自检：实例提供 `compliance_self_check()` 方法，返回关键字段
（`ckpt_dir` / `ckpt_name` / `action_type` / `model_id`）以及
是否误含禁用字段（`task_models` / `fallback_map` / `active_model`
等），便于评测方动态核查。

## 任务覆盖（12 任务统一处理）

| 任务 | 处理方式 |
|------|---------|
| `stack_bowls` / `stack_bowls_random` | 同一模型推理 |
| `push_T` / `push_T_random` | 同一模型推理 |
| `pack_objects_into_box` / `_random` | 同一模型推理 |
| `fold_clothes` / `_random` | 同一模型推理 |
| `hang_mugs` / `_random` | 同一模型推理 |
| `sweep_blocks` / `_random` | 同一模型推理 |
| `pour_liquid_into_cup` / `_random` | 同一模型推理 |
| `make_toast` / `_random` | 同一模型推理 |
| `arrange_largest_number` / `_random` | 同一模型推理 |
| `sort_nesting_dolls_by_size` / `_random` | 同一模型推理 |
| `store_laptop_and_headphones` / `_random` | 同一模型推理 |
| `stack_blocks` / `_random` | 同一模型推理 |

> 单一 ckpt 训练时应在所有 12 任务的合并数据集上做 cotrain，
> 以最大化泛化能力。具体训练方式见下方"训练"小节。

## 接口

由 XPolicyLab `PolicyServer`（WebSocket 协议）调用：

| 方法 | 说明 | 是否切换模型 |
|------|------|--------------|
| `__init__(model_cfg)` | 加载唯一一个 ACT checkpoint | — |
| `prepare_case(case_meta)` | 仅记录 case_meta 用于日志 | **否** |
| `update_obs(obs)` | 转发观测到唯一模型 | — |
| `get_action()` | 返回 chunk_size 个动作（joint, 14-dim） | — |
| `reset()` | 重置唯一模型的时序状态（不替换模型） | **否** |
| `compliance_self_check()` | 返回合规自检字典（可选） | — |

## 部署

```bash
# 1. 启动 Policy Server（单 ckpt，单模型，默认开启 Batch Inference）
python XPolicyLab/setup_policy_server.py \
    --config_path policy/UnifiedACT/deploy.yml \
    --overrides host=0.0.0.0 port=19002

# 2. 公网暴露（DSW 上可选）
cloudflared tunnel --url http://localhost:19002 &
```

`deploy.yml` 的 `ckpt_dir` 默认指向 cotrain recipe 产物路径：

```
/root/ckpts/RoboDojo-cotrain-arx_x5-joint-0
```

评测方环境若不同，请按需修改 `ckpt_dir`，或通过 `--overrides ckpt_dir=<路径>` 覆盖。

## ★ Batch Inference（赛事方推荐，显著加快评测进度）

### 原理
XPolicyLab PolicyServer 是 **多连接 / 多线程** 的：每条 wss 连接为独立线程
（`evaluation_id + trial_id` 唯一标识 session），按 episode 循环调用：

    prepare_case → reset → (update_obs → get_action) × N 步 → TRIAL_END

当评测端同时跑多条连接时，不同线程的 `get_action()` 会在接近的时间点
各自请求一次 GPU 前向。**UnifiedACT 的 Batch Inference Adapter** 将这些
接近同时的请求合并为一个"动态 batch"，一次性处理，显著提升 GPU 利用率
和总评测吞吐。

### 合规保证（Batch 模式仍然满足全部规则）
- **唯一 ckpt**：仍然只有一份 `self.backbone`，全部 `nn.Parameter` 恒定；
  batch 只是"数据并行"维度的复用，不是切换 ckpt / 动作类型 / 协议。
- **Batch = 1 时结果与单步逐位相同**：当队列中只有 1 个 session、窗口
  超时触发时，完整走原 backbone 的 `update_obs → get_action` 流水线，
  不改变输出分布，**不影响评测成功率**。
- **Per-session 状态严格隔离**：`temporal_agg` 的 `all_actions`、
  `obs_history`、时间步等全部状态以 trial_id 为 key 独立存储，
  session 之间无串扰。

### 配置项（deploy.yml）

| 字段 | 默认 | 含义 |
|------|------|------|
| `eval_batch` | `true` | 总开关；`false` 回退到原单步路径（100% 行为不变）。 |
| `batch_max_size` | `8` | 单个 batch 的最大 session 数。A10 24G 建议 4~8；更大 GPU 可到 16。 |
| `batch_window_ms` | `12` | 收集窗口（毫秒）。越小延迟越低，越大 batch 越满。建议 8~20 ms。 |

### 推荐用法
- **评测端**：同时向 Policy Server 建立 4~8 条并行 wss 连接（跑不同
  trial / seed），会自动合并为 batch 推理。
- **诊断**：实例提供 `model.batch_stats()` 返回：
  - `num_batches` / `last_batch_size` / `avg_batch_size`：实际 batch 效果
  - `scheduler_errors`：批内部异常计数（应为 0）
  - `active_trials`：当前在线 session 数
- **兼容性回退**：如果环境有任何问题，把 `eval_batch` 改为 `false`
  即可完全恢复原单步行为，不影响合规与成功率。

### 与官方 `eval_batch` 字段的关系
XPolicyLab 在 deploy.yml 标准字段中预留了 `eval_batch`。UnifiedACT
对该字段做了原生支持：当 `eval_batch=true` 时启用动态 batch，否则走
旧路径。评测方无需修改 PolicyServer 代码，直接用标准字段即可。

## 训练（生产单一多任务 ckpt）

UnifiedACT 不引入任何额外训练逻辑，完全复用 XPolicyLab ACT 的官方
cotrain recipe：

```bash
cd XPolicyLab/policy/ACT

# 1. 合并数据处理（如官方提供合并数据集，可跳过此步）
bash process_data.sh RoboDojo cotrain arx_x5 joint

# 2. 多任务联合训练（产出单一 ckpt，覆盖 12 任务分布）
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

产物路径示例：

```
XPolicyLab/policy/ACT/checkpoints/RoboDojo-cotrain-arx_x5-joint-0/
```

将上方路径填入 `deploy.yml::ckpt_dir` 即完成接入。

> 说明：cotrain recipe 来自 `XPolicyLab/policy/ACT/README.md` 的官方示例，
> 该 README 明确写道："task_name is only the evaluation task;
> multi-task checkpoints can be evaluated on different tasks without
> renaming the checkpoint directory."  —— 这正是 UnifiedACT 的设计依据。

## 依赖

- XPolicyLab (`pip install -e XPolicyLab`)
- ACT policy (`XPolicyLab/policy/ACT`，含 `detr/` 子树）
- torch, cv2, numpy, mujoco（间接）

## 与旧方案（MultiACT）的差异

| 维度 | 旧 MultiACT（已弃用，不合规） | 新 UnifiedACT（合规） |
|------|------------------------------|----------------------|
| ckpt 数量 | 11 个（task_ckpt_map） | 1 个 |
| 任务切换 | 按 `action_case_id` 切换 `active_model` | 无切换 |
| Fallback 映射 | `DEFAULT_FALLBACK_MAP` 7 项 | 无 |
| 合规风险 | 评测期更换模型，违反规则 | 单模型恒定，合规 |
| 评测方异议 | 已被官方拒绝（见通知邮件） | 设计上即满足规则 |
