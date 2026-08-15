# 复现指南 - GOAI Embodied Solo（赛事规则合规版）

> 参赛团队：GOAI Solo Builder（个人参赛）｜ 赛题一：RoboDojo / X-Eval 平台
> 目标：在标准环境完成 环境搭建 → cotrain 单 ckpt 训练 → 部署 → 评测 → 提交 全流程复现

本指南面向评审者与社区开发者，按步骤复现参赛方案。所有命令均在项目根目录
`goai-embodied-solo/` 下执行。

---

## 0. 合规整改摘要

旧 MultiACT 多 ckpt 切换方案已被赛事方认定为违规（违反
"审核和正式评测期间不更换模型"规则）。本指南对应整改后的 **单一 ckpt**
UnifiedACT 方案：

- 训练阶段：调用 XPolicyLab ACT 官方 cotrain recipe，产出 **唯一一个** 多任务 ckpt
- 部署阶段：单一 ckpt 加载到单一模型实例
- 评测阶段：全部 24 配置使用同一模型，无切换

---

## 前置要求

- 操作系统：Linux / macOS，或 Windows 下的 WSL2（推荐 Ubuntu 22.04）
- Python：3.8 及以上（推荐 3.10）
- Conda：Miniconda 或 Anaconda 已安装
- Git LFS：`git lfs install` 已初始化（RoboDojo 含大文件）
- GPU：训练阶段必需（cotrain 数据量大）；推理服务可在 CPU 运行但 GPU 更佳
- 网络：可访问 GitHub、HuggingFace、魔搭 ModelScope（DSW 训练用）
- 公网端点：仅步骤 7 需要（固定公网 IP 或稳定域名 + TLS 证书）

---

## 步骤 1：环境搭建（RoboDojo + Assets + GOAI 数据）

执行一键环境搭建脚本：

```bash
bash scripts/setup_env.sh
```

该脚本将完成：

1. 克隆 RoboDojo 仓库并初始化子模块
2. 安装 RoboDojo conda 环境（`conda activate RoboDojo`）
3. 下载 GOAI 专用 Assets（来自 HuggingFace `RoboDojo-Benchmark/RoboDojo`）
4. 下载 GOAI-2026 训练/评测数据（`data/hdf5`）
5. 执行路径修正（`python utils/update_embodiment_config_path.py`）

验证：`conda activate RoboDojo` 成功，且 24 个配置均可启动 episode。

---

## 步骤 2：Policy 框架（XPolicyLab + 30+ baseline）

```bash
bash scripts/setup_xpolicylab.sh
```

该脚本将完成：

1. 克隆 XPolicyLab 仓库
2. 浏览 30+ baseline 列表（ACT / Diffusion Policy / RDT 等 SOTA 双臂操作策略）
3. 安装 XPolicyLab 通用依赖

验证：`XPolicyLab/policy/` 下至少 1 个 baseline（推荐 ACT）可加载并产出动作。

---

## 步骤 3：训练单一多任务 ckpt（★ 合规关键步骤）

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

产物路径示例：

```
XPolicyLab/policy/ACT/checkpoints/RoboDojo-cotrain-arx_x5-joint-0/
```

> **合规关键**：必须使用单一 ckpt（cotrain recipe），不得退回到
> 按任务独立训练 + 按任务切换 ckpt 的多模型方案。
>
> 官方依据（XPolicyLab ACT README）：
> "task_name is only the evaluation task; multi-task checkpoints can be
> evaluated on different tasks without renaming the checkpoint directory."

验证：产物目录下存在 `policy_*.ckpt` 文件（至少 1 个）。

---

## 步骤 4：接入 UnifiedACT 包装器

将本仓库的 `policy/UnifiedACT/` 复制到 XPolicyLab：

```bash
cp -r policy/UnifiedACT XPolicyLab/policy/
```

修改 `policy/UnifiedACT/deploy.yml::ckpt_dir` 指向步骤 3 的产物绝对路径：

```yaml
ckpt_dir: /path/to/XPolicyLab/policy/ACT/checkpoints/RoboDojo-cotrain-arx_x5-joint-0
```

验证：

```bash
python -c "
import sys; sys.path.insert(0, 'policy')
from UnifiedACT.model import Model
m = Model({'ckpt_dir': '/path/to/cotrain', 'ckpt_name': 'cotrain', 'action_type': 'joint'})
print(m.compliance_self_check())
"
```

期望输出包含 `'rule_compliant': True` 与 `'forbidden_fields_present': []`。

---

## 步骤 5：合规性单元测试

```bash
python tests/test_model_interface.py
```

期望输出：`10/10 tests passed`，验证：

- 不存在 `task_models` / `fallback_map` / `active_model` 等多模型字段
- 24 个 case 调用 `prepare_case` 后 `id(self.model)` 恒定
- `prepare_case` 不触发底层模型 reset/load
- 动作类型在构造后不被改变

---

## 步骤 6：本地评测（24 配置，单一 ckpt）

```bash
POLICY_DIR=XPolicyLab/policy/UnifiedACT \
POLICY_ENV=RoboDojo \
RUN_LABEL=cotrain \
ACTION_TYPE=joint \
bash scripts/eval_local.sh
```

该脚本将：

1. 读取 `configs/tasks.yaml` 中 12 任务列表
2. 对每个任务的标准配置与 `_random` 配置分别评测（全部使用同一 cotrain ckpt）
3. 汇总得分至本地评测报告

验证：综合得分稳定，弱任务已识别，作为后续泛化优化的基线。

---

## 步骤 7：冒烟测试（24 配置各 1 episode）

```bash
POLICY_DIR=XPolicyLab/policy/UnifiedACT \
POLICY_ENV=RoboDojo \
RUN_LABEL=cotrain \
ACTION_TYPE=joint \
bash scripts/smoke_test.sh
```

验证端到端链路通畅：

- RoboDojo 仿真环境可正常启动与步进
- UnifiedACT 可正常加载 cotrain ckpt 并响应（日志中应看到 `SAME model (no switching)`）
- wss:// 链路（本地模拟）可正常收发 obs/action

注意：冒烟测试仅验证链路通畅，不代表正式成绩。

---

## 步骤 8：Policy Server 部署（公网 wss://）

```bash
python scripts/deploy_policy_server.py \
  --config-path policy/UnifiedACT/deploy.yml \
  --host 0.0.0.0 \
  --port 8443 \
  --overrides protocol=ws
```

部署要点：

- 公网固定 IP 或稳定域名，启用 TLS，提供 `wss://` 端点
- 1-8 个端点，端口映射 / 防火墙 / 反向代理配置正确
- 禁止使用 `localhost` / `127.0.0.1` / 局域网 IP
- 审核与正式评测期间不更换 ckpt / 动作类型 / 协议行为

详细部署方案见 `docs/policy-server-deploy.md`。

验证：从公网客户端连接 `wss://host:port` 响应正常，长时间稳定不掉线。

---

## 步骤 9：正式提交

在 X-Eval 平台提交表单填写 Policy Server 信息：

1. 访问 https://xsparkai.com/goai-2026/apply
2. 填写提交表单字段：
   - 队伍名称
   - 联系人
   - 手机号
   - 邮箱
   - Policy Server 主机与端口（1-8 个端点）
   - 策略名称：**UnifiedACT**（仅字母、数字、下划线）
   - 动作类型：**joint**
3. 提交后等待审核与正式评测

---

## 注意事项

1. **提交次数**：最多 3 次有效提交，取最高分；综合得分 >10 分晋级。
   请勿用正式提交机会调试。
2. **合规硬约束**：评测期间不得更换 ckpt / 动作类型 / 协议行为。
   UnifiedACT 架构已从源头保证此约束。
3. **审核期**：保持 Policy Server 在线，不更换模型/动作类型/协议行为。
4. **端点合规**：不要提交 `localhost` / `127.0.0.1` / 局域网 IP，
   X-Eval 评测端无法访问。
5. **提交前自测**：每次正式提交前务必完成本地评测与冒烟测试，
   避免浪费提交机会。
6. **跨实例续训**：DSW GPU 实例到期后使用 `--ckpt_path` 在新实例续训，
   避免训练重置。
7. **资源来源**：Assets 与 GOAI-2026 数据均来自 HuggingFace 官方仓库，
   复现时请遵循其许可协议。

---

## 复现验证清单

- [ ] `conda activate RoboDojo` 成功，24 配置均可启动 episode
- [ ] XPolicyLab 已克隆，ACT baseline 可加载并产出动作
- [ ] cotrain recipe 已运行，产物 ckpt 存在
- [ ] UnifiedACT 已接入 XPolicyLab，`deploy.yml::ckpt_dir` 指向 cotrain 产物
- [ ] `tests/test_model_interface.py` 10/10 PASS
- [ ] 24 配置本地评测完成，综合得分记录在案
- [ ] 冒烟测试 24 配置各 1 episode 通过
- [ ] 公网 Policy Server wss:// 可访问，长时间稳定
- [ ] X-Eval 提交表单填写完成，Policy=UnifiedACT，Action=joint
- [ ] 最高成绩 >10 分，确认晋级
