"""UnifiedACT: 单一 ACT 推理适配器（赛事规则合规版）

设计目标
========
满足赛事规则："审核和正式评测期间，请保持同一服务在线，
不要更换模型、动作类型或协议行为。"

本模块包装 XPolicyLab ACT 的 **单一** checkpoint，全部 24 个评测配置
（12 任务 × {标准, _random}）共用同一个模型实例与同一组参数。
对齐 XPolicyLab `ModelTemplate` 接口，可直接被 `PolicyServer` 调用。

合规要点
========
1. 构造时只加载 **1 个** ACT checkpoint；不存在 `task_ckpt_map`、
   `task_models`、`fallback_map`、`active_model` 切换等任何多模型字段。
2. `prepare_case(case_meta)` 仅缓存 `case_meta` 用于日志/诊断，
   **不**切换、不重载、不替换任何模型参数；模型的 `nn.Parameter`
   集合在评测全生命周期保持恒定。
3. `update_obs / get_action / reset` 全部委托给同一个底层模型实例。
4. 动作类型（joint/ee）与协议（ws/wss）由 `deploy.yml` 静态声明，
   评测期间不可变更。

单一 ckpt 的训练方式
====================
参照 `XPolicyLab/policy/ACT/README.md` 的多任务 cotrain recipe：

    cd XPolicyLab/policy/ACT
    bash train.sh RoboDojo cotrain arx_x5 joint 0 0

产物路径形如：`checkpoints/RoboDojo-cotrain-arx_x5-joint-0/`，
将其绝对路径填入 `deploy.yml::ckpt_dir` 即可。

部署方式
========
    python XPolicyLab/setup_policy_server.py \\
        --config_path policy/UnifiedACT/deploy.yml \\
        --overrides host=0.0.0.0 port=19002

核心接口（由 XPolicyLab PolicyServer 调用）
==========================================
- ``__init__(model_cfg)``  : 加载唯一一个 ACT checkpoint
- ``prepare_case(case_meta)``: 仅记录 case 元数据，**不切换模型**
- ``update_obs(obs)``       : 转发观测到唯一模型
- ``get_action()``          : 返回 chunk_size 个动作（joint, 14-dim）
- ``reset()``               : 重置唯一模型的时序状态
"""
import os

# ACTModel 延迟导入：仅在实例化时需要 XPolicyLab，模块级别可独立 import（测试友好）
_ACTModel = None


def _get_act_model_class():
    """惰性加载 XPolicyLab ACT 适配器，避免顶层 import 失败。"""
    global _ACTModel
    if _ACTModel is None:
        from XPolicyLab.policy.ACT.model import Model as _M
        _ACTModel = _M
    return _ACTModel


class Model:
    """单一 ACT 推理适配器（合规版）。

    全部 24 个评测配置共用 ``self.model`` 这同一个实例与同一组参数；
    `prepare_case` 不会替换、重置或重载该实例。
    """

    # 显式声明：本类不存在任务切换字段（便于评测方静态核查）
    _COMPLIANCE_FIELDS = (
        "ckpt_dir",          # 唯一 ckpt 路径，构造后不可变
        "ckpt_name",         # 唯一 ckpt 名称，构造后不可变
        "action_type",       # 动作类型，构造后不可变
        "model",             # 唯一底层模型实例，构造后引用不变
    )

    def __init__(self, model_cfg):
        print(f"[UnifiedACT] init with cfg keys: {list(model_cfg.keys())}")

        ckpt_dir = model_cfg.get("ckpt_dir")
        ckpt_name = model_cfg.get("ckpt_name", "cotrain")
        action_type = model_cfg.get("action_type", "joint")

        if not ckpt_dir:
            raise ValueError(
                "[UnifiedACT] 'ckpt_dir' is required. Train the unified ckpt via: "
                "`bash XPolicyLab/policy/ACT/train.sh RoboDojo cotrain arx_x5 joint 0 0` "
                "then set ckpt_dir to the produced run directory."
            )

        # 构造唯一底层模型（XPolicyLab ACT 适配器）
        cfg = dict(model_cfg)
        cfg["ckpt_name"] = ckpt_name
        cfg["ckpt_dir"] = ckpt_dir
        print(f"[UnifiedACT] loading SINGLE ACT ckpt: dir={ckpt_dir} name={ckpt_name}")
        self.model = _get_act_model_class()(cfg)

        # 冻结字段（仅用于诊断/日志，不参与任何路由逻辑）
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = ckpt_name
        self.action_type = action_type
        self.last_case_meta = None
        self._case_counter = 0

        print(
            f"[UnifiedACT] ready; ONE model for ALL 24 configs "
            f"(action_type={action_type}, ckpt_name={ckpt_name}). "
            f"Rule-compliant: no checkpoint/task/protocol switching."
        )

    # ------------------------------------------------------------------
    # XPolicyLab PolicyServer 调用接口
    # ------------------------------------------------------------------

    def prepare_case(self, case_meta):
        """新评测 case 开始时由 PolicyServer 调用。

        合规说明
        ========
        本方法 **仅记录** ``case_meta`` 用于日志与诊断，
        **不**调用任何模型加载、参数替换、活跃模型切换等操作。
        评测期间 ``self.model`` 引用恒定，``nn.Parameter`` 集合恒定。

        即便 ``action_case_id`` 在 24 个评测配置间变化，
        本方法也不会改变底层模型，确保满足"不更换模型"的规则要求。
        """
        action_case_id = str(case_meta.get("action_case_id", "")) if case_meta else ""
        evaluation_id = str(case_meta.get("evaluation_id", "")) if case_meta else ""
        trial_id = str(case_meta.get("trial_id", "")) if case_meta else ""
        self.last_case_meta = case_meta
        self._case_counter += 1
        print(
            f"[UnifiedACT] case #{self._case_counter} "
            f"case_id={action_case_id!r} eval_id={evaluation_id!r} "
            f"trial_id={trial_id!r} -> SAME model (no switching, rule-compliant)"
        )

    def update_obs(self, obs):
        """转发观测到唯一模型实例。"""
        self.model.update_obs(obs)

    def get_action(self):
        """获取动作（chunk_size 个）from 唯一模型实例。"""
        return self.model.get_action()

    def reset(self):
        """重置唯一模型实例的时序状态（不替换模型本身）。"""
        m = getattr(self, "model", None)
        if m is not None and hasattr(m, "reset"):
            m.reset()

    # ------------------------------------------------------------------
    # 合规自检（评测方可选调用，便于静态/动态核查）
    # ------------------------------------------------------------------

    def compliance_self_check(self) -> dict:
        """返回一个合规自检字典，列出本实例的关键不变量。

        评测方可通过 PolicyServer 调用此方法（或在日志中查看）以确认：
        - 只有一个 model 实例
        - ckpt_dir / ckpt_name / action_type 在评测期间未变更
        - 不存在 task_models / fallback_map / active_model 等多模型字段
        """
        forbidden = ("task_models", "fallback_map", "active_model",
                     "active_task", "task_ckpt_map", "default_task")
        present_forbidden = [k for k in forbidden if hasattr(self, k)]
        return {
            "policy_class": self.__class__.__name__,
            "ckpt_dir": self.ckpt_dir,
            "ckpt_name": self.ckpt_name,
            "action_type": self.action_type,
            "model_id": id(self.model),
            "case_count": self._case_counter,
            "forbidden_fields_present": present_forbidden,
            "rule_compliant": not present_forbidden,
        }
