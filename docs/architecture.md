# 架构设计 - GOAI Embodied Solo（赛事规则合规版）

> 参赛团队：GOAI Solo Builder（个人参赛）｜ 赛道：具身未来
> 赛题选择：赛题一 通用双臂协同操作能力测试（基于 RoboDojo / X-Eval 平台）
> 文档版本：v2.0（2026-08-12 合规整改）｜ 维护者：GOAI Solo Builder

本文档描述 GOAI 具身未来赛道参赛项目的整体架构、模块划分、数据流与关键设计决策，
作为后续开发、部署与复现的顶层参考。

---

## 0. 合规整改摘要

针对赛事组委会通知中明确指出的违规问题（"在同一服务内依据不同任务切换对应
ACT checkpoint 进行推理属于评测过程中更换模型"），本架构已重构为：

- **唯一一个** ACT checkpoint（cotrain recipe 产物）
- **唯一一个** 模型实例（`policy/UnifiedACT/`）
- **不切换** 模型 / 动作类型 / 协议
- 满足规则："审核和正式评测期间，请保持同一服务在线，
  不要更换模型、动作类型或协议行为。"

旧 `policy/MultiACT/` 多 ckpt 切换方案已删除。

---

## 1. 整体架构图（合规版）

```
                +---------------------------------------------------+
                |                X-Eval 评测端（官方）               |
                |   xsparkai.com/goai-2026  |  最多 3 次取最高       |
                |   >10 分晋级  |  1-8 个 wss:// 端点                 |
                +---------------------------------------------------+
                                          ^
                                          | (wss://host:port, TLS)
                                          v
                +---------------------------------------------------+
                |              Policy Server（公网部署）             |
                |   WebSocket(wss) + TLS  |  动作类型 joint          |
                |   ★ 单一 ckpt，审核与评测期不更换模型/动作类型/协议  |
                +---------------------------------------------------+
                                          ^
                                          | (加载唯一 checkpoint)
                                          v
                +---------------------------------------------------+
                |          UnifiedACT 推理（XPolicyLab ACT）         |
                |   单一多任务 ACT checkpoint（cotrain recipe）       |
                |   ★ 全部 24 配置共用同一组 nn.Parameter            |
                +---------------------------------------------------+
                                          ^
                                          | (obs -> action)
                                          v
                +---------------------------------------------------+
                |     RoboDojo 仿真（Isaac Sim / IsaacLab）           |
                |   12 任务 × (标准 + _random) = 24 配置              |
                |   本地成功率评测需 Isaac；Policy Server 不需要      |
                +---------------------------------------------------+
```

数据流主链路自下而上：RoboDojo 仿真环境产出观测 obs → UnifiedACT 根据 obs
计算 action（joint）→ Policy Server 经由 wss:// 将 action 返回 → X-Eval 评测端
在仿真中执行 action 并产出下一帧 obs 与任务进度，循环直至 episode 结束并打分。

**关键合规不变量**：无论 `action_case_id` 在 24 个配置间如何变化，
UnifiedACT 的 `self.model` 引用与底层 `nn.Parameter` 集合始终保持恒定。

---

## 2. 模块说明

### 2.1 仿真层：RoboDojo

- 基础：官方本地评测客户端（`robodojo.sh client` / `eval_policy.sh` / `src/eval_client`）
  基于 **Isaac Sim + IsaacLab**（`from isaaclab.app import AppLauncher`）。
  仓库 `env/environment/` 仅有 `isaac/` 后端，**无独立 MuJoCo 成功率评测入口**。
- 与 Policy Server 的分工：参赛者部署的 Policy Server（XPolicyLab WebSocket）
  **不需要** IsaacLab；X-Eval 平台侧跑仿真。本地若要自己算 episode 成功率，
  则必须安装 Isaac。
- 任务规模：12 个基础任务，每个任务提供标准与 `_random` 配置，合计 24 个可运行配置。
- 随机化：`_random` 配置对初始状态、物体位姿、纹理/光照等施加随机化，
  用于评测 Generalization（泛化）维度。
- 资源：Assets 与 GOAI-2026 数据从 HuggingFace 下载，
  需执行 `utils/update_embodiment_config_path.py` 修正路径。

### 2.2 Policy 框架：XPolicyLab

- 来源：官方提供的 Policy 框架，内含 30+ baseline。
- 代表 baseline：ACT、Diffusion Policy、RDT 等 SOTA 双臂操作策略。
- 适配规范：参照 `XPolicyLab/policy/ACT/README.md`，自有 policy 需接入统一
  目录结构、环境安装、训练与服务启动流程。
- 输出：动作类型由 `deploy.yml::action_type` 静态声明（本方案固定为 `joint`），
  评测期间不可变更。

### 2.3 训练层：DSW GPU 远程训练 + cotrain 单一 ckpt（合规关键）

- 算力：魔搭 ModelScope DSW GPU 实例（A10/V100，含免费额度）。
- **合规训练策略**：调用 XPolicyLab ACT 官方 cotrain recipe，
  在合并的 12 任务数据集上联合训练，产出 **唯一一个** 多任务 ckpt：

  ```bash
  cd XPolicyLab/policy/ACT
  bash train.sh RoboDojo cotrain arx_x5 joint 0 0
  ```

  产物路径示例：`checkpoints/RoboDojo-cotrain-arx_x5-joint-0/`

- **不再使用**：按任务独立训练 + 按任务切换 ckpt 的多模型方案（违规）。
- 远程执行：复用 `dsw-remote-execution` Skill，通过 Chrome CDP + JupyterLab
  REST API + nohup 长任务实现远程训练与文件上传下载。
- 监控：训练 loss 与收敛通过 JupyterLab 实时查看，
  checkpoint 落盘后下载至本地或 Policy Server 主机。

### 2.4 服务层：Policy Server（UnifiedACT）

- 协议：WebSocket Secure（wss://），启用 TLS（评测期不可变更）。
- 部署：公网固定 IP 或稳定域名，1-8 个端点，
  禁止 localhost/127.0.0.1/局域网 IP。
- 启动：`python scripts/deploy_policy_server.py` 加载 **唯一一个** ckpt。
- **合规不变量**：
  - 单一 ckpt 路径（`deploy.yml::ckpt_dir`）
  - 单一动作类型（`deploy.yml::action_type = joint`）
  - 单一协议（`deploy.yml::protocol = ws`）
  - 单进程、单模型实例
- 稳定性：审核期与正式评测期间不更换模型、动作类型与协议行为；
  建议 systemd/supervisor 守护与自动重启。

### 2.5 推理适配层：UnifiedACT（合规核心）

- 路径：`policy/UnifiedACT/`
- 接口：完全对齐 XPolicyLab `ModelTemplate`
- **关键合规字段**（构造后不可变）：

  | 字段 | 类型 | 说明 |
  |------|------|------|
  | `model` | ACT Model | 唯一底层模型实例（`id()` 恒定） |
  | `ckpt_dir` | str | 唯一 ckpt 路径 |
  | `ckpt_name` | str | 唯一 ckpt 名称（默认 `cotrain`） |
  | `action_type` | str | 动作类型（默认 `joint`） |

- **关键合规方法**：

  | 方法 | 是否切换模型 | 说明 |
  |------|------------|------|
  | `__init__(model_cfg)` | — | 加载唯一 ckpt |
  | `prepare_case(case_meta)` | **否** | 仅记录日志，不调用模型加载/重载 |
  | `update_obs(obs)` | — | 委托给唯一模型 |
  | `get_action()` | — | 委托给唯一模型 |
  | `reset()` | **否** | 仅重置时序状态，不替换模型实例 |
  | `compliance_self_check()` | — | 返回合规自检字典（评测方动态核查用） |

- **禁用字段**（实例不应存在）：
  - `task_models`（任务→模型字典）
  - `task_ckpt_map`（任务→ckpt 路径字典）
  - `fallback_map` / `DEFAULT_FALLBACK_MAP`（兜底映射）
  - `active_model` / `active_task`（当前活跃模型/任务）

### 2.6 评测层：X-Eval 平台

- 入口：xsparkai.com/goai-2026（提交表单 `/apply`）。
- 提交规则：最多 3 次有效提交，取最高分；综合得分 >10 分晋级。
- 提交字段：队伍名称、联系人、手机号、邮箱、Policy Server 主机与端口（1-8）、
  策略名称（字母数字下划线）、动作类型（joint/ee）。
- 真机：初赛晋级后进入 8 项真机任务评测（1 月云调试 + 3 天线下）。

---

## 3. 数据流

主循环（每个 episode）：

1. X-Eval 评测端从 RoboDojo 仿真环境获取当前观测 `obs`（含视觉、本体感觉等）。
2. 评测端通过 `wss://host:port` 将 `obs` 发送至 Policy Server。
3. Policy Server 调用 **同一** UnifiedACT 模型推理（加载 cotrain ckpt），
   生成动作 `action`（joint）。
4. Policy Server 经由 wss 将 `action` 返回评测端。
5. 评测端将 `action` 注入 RoboDojo 仿真环境，步进物理仿真，
   产出下一帧 `obs` 与任务进度。
6. 循环 2-5 直至 episode 结束（成功 / 失败 / 超时），由 X-Eval 记录该配置得分。

```
obs ──> X-Eval 评测端 ──wss──> Policy Server ──> UnifiedACT (cotrain ckpt)
                                                                |
action (joint, 14-dim) <──wss── Policy Server <─────────────────┘
   |
   v
RoboDojo 仿真环境 ──> 下一帧 obs + 任务进度 ──> X-Eval 评测端
```

---

## 4. 关键设计决策

### 决策 1：选择赛题一（标准化、个人可行、有 baseline）

赛题一基于 RoboDojo + X-Eval 标准化基准，提供 30+ 现成 baseline，
个人参赛者无需真机硬件即可完成初赛全流程。标准化评测与可复现链路显著降低
个人参赛门槛。

### 决策 2：单一多任务 ckpt（cotrain）替代多 ckpt 切换（合规必选）

**背景**：旧 MultiACT 方案在评测期按任务切换 ckpt，被赛事方认定为违规。

**新方案**：调用 XPolicyLab ACT 官方 cotrain recipe，在合并的 12 任务数据上
联合训练，产出 **唯一一个** 多任务 ckpt。这符合官方 ACT README 的明确说明：
"task_name is only the evaluation task; multi-task checkpoints can be
evaluated on different tasks without renaming the checkpoint directory."

**好处**：
- 单一 ckpt 满足"评测期不更换模型"的硬性规则
- 多任务联合训练有助于跨任务共享表示，提升 `_random` 泛化
- 减少部署复杂度（无任务路由/fallback 逻辑）

### 决策 3：Generalization 维度作为优化重点

12 个任务各提供 `_random` 随机化配置，泛化能力是晋级与冲分的关键。
通过 cotrain 数据合并、状态扰动、动作平滑等手段提升泛化性。

### 决策 4：跨实例续训解决 DSW GPU 时限问题

DSW GPU 免费实例存在单次时限。采用 `--ckpt_path` 在实例到期后将
checkpoint 迁移至新实例续训，保持训练连续性。

### 决策 5：Policy Server 公网部署（合规服务化）

赛题一要求以公网 Policy Server 形式提交。参赛者通过 `wss://` 暴露推理服务，
X-Eval 评测端直接连接调用。**合规要点**：单一 ckpt + 单一动作类型 + 单一协议。

---

## 5. 模块依赖关系

```
[RoboDojo 本地评测] ──依赖──> [Isaac Sim + IsaacLab + Assets]
[Policy Server]    ──不依赖──> Isaac（仅需 XPolicyLab + ckpt）
[UnifiedACT 推理]  ──依赖──> [XPolicyLab 框架 + 唯一 cotrain checkpoint]
[Policy Server]    ──依赖──> [UnifiedACT + TLS 证书 + 公网端点]
[X-Eval 评测]      ──依赖──> [Policy Server (wss) + 24 配置]
[训练层 DSW]       ──产出──> [cotrain ckpt] ──喂给──> [Policy Server]
```

环境搭建（仿真层 + Policy 框架）是所有后续阶段的前提；cotrain 训练层产出
**唯一 ckpt** 后方可服务化；服务层与评测层强依赖 ckpt 与公网端点就绪。

---

## 6. 与评审维度的对齐

| 大赛评审维度 | 架构对应措施 |
|------------|--------------|
| 赛题契合度与任务价值 | 赛题一，12 任务泛化，真机闭环 |
| 技术方案与模型能力 | cotrain 单一多任务 ACT + 泛化优化 |
| 工程实现与可运行性 | 冒烟测试 24 配置 + Policy Server 稳定 + 合规自检 |
| 评测结果与证据质量 | 3 次提交取最高 + 本地评测报告 + 真机结果 |
| 开放/开源贡献与长期成长性 | 完整开源仓库 + 复现文档 + 合规测试套件 |

---

## 7. 合规审查清单（评测方参考）

- [x] **单一 ckpt**：`UnifiedACT/deploy.yml` 只声明一个 `ckpt_dir`
- [x] **单一模型**：`UnifiedACT/model.py` 只持有 `self.model` 一个实例引用
- [x] **prepare_case 不切换**：测试覆盖 24 个 case 调用后 `id(self.model)` 恒定
- [x] **无禁用字段**：实例不存在 `task_models` / `fallback_map` / `active_model`
- [x] **单一动作类型**：`action_type=joint` 构造后不可变更
- [x] **单一协议**：`protocol=ws` 构造后不可变更
- [x] **合规自检**：提供 `compliance_self_check()` 方法供动态核查
- [x] **测试覆盖**：`tests/test_model_interface.py` 10 项 PASS

---

## 8. 参考资料

- 官网：goaihz.com/tracks?track=embodied
- X-Eval 平台：xsparkai.com/goai-2026
- RoboDojo：https://robodojo-benchmark.com
- XPolicyLab：GitHub 仓库（含 30+ baseline）
- 参赛手册：参赛手册.pdf / 具身未来 参赛手册.pdf
- 赛事方通知邮件（2026-08-12，关于违规判定与整改要求）
