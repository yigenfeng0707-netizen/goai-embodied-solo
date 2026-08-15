"""UnifiedACT Model 接口单元测试（mock，无需真实 ckpt / XPolicyLab）。

运行:
    python -m pytest tests/test_model_interface.py -v
或:
    python tests/test_model_interface.py

合规测试目标
============
验证UnifiedACT满足赛事规则："评测期不更换模型/动作类型/协议行为"：
1. 构造时只加载 **1 个** 模型（不存在 task_models 字典）；
2. `prepare_case` 在不同 action_case_id 之间 **不**切换模型实例；
3. 不存在 fallback_map / active_model / task_ckpt_map 等多模型字段；
4. update_obs / get_action / reset 始终委托给同一个模型实例。
"""
import sys
import os
from unittest.mock import MagicMock, patch

# 将 policy/ 加入 import 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "policy"))


def _make_mock_act_model():
    """创建一个 mock 的底层 ACT Model（XPolicyLab ACT 适配器实例）。"""
    m = MagicMock()
    m.reset = MagicMock()
    m.update_obs = MagicMock()
    m.get_action = MagicMock(return_value=[{"left_arm_joint_state": [0.0] * 6}])
    return m


def _make_unifiedact(ckpt_dir="/fake/cotrain", ckpt_name="cotrain",
                     action_type="joint"):
    """构造 UnifiedACT 实例，绕过真实 ACTModel 加载。"""
    with patch.dict(sys.modules, {
        "torch": MagicMock(),
        "XPolicyLab.policy.ACT.model": MagicMock(),
    }):
        # 重新导入确保拿到当前模块
        if "UnifiedACT.model" in sys.modules:
            del sys.modules["UnifiedACT.model"]
        from UnifiedACT.model import Model
        m = Model.__new__(Model)
        m.model = _make_mock_act_model()
        m.ckpt_dir = ckpt_dir
        m.ckpt_name = ckpt_name
        m.action_type = action_type
        m.last_case_meta = None
        m._case_counter = 0
        return m


# ---------------------------------------------------------------------------
# 测试 1：合规性 — 不存在多模型字段
# ---------------------------------------------------------------------------

def test_no_multi_model_fields():
    """UnifiedACT 实例不应有任何任务切换相关字段。"""
    m = _make_unifiedact()
    forbidden = ("task_models", "fallback_map", "active_model",
                 "active_task", "task_ckpt_map", "default_task")
    for field in forbidden:
        assert not hasattr(m, field), (
            f"UnifiedACT 不应包含多模型字段 '{field}'（违反单 ckpt 合规约束）"
        )


def test_compliance_self_check_passes():
    """compliance_self_check() 应回报 rule_compliant=True。"""
    m = _make_unifiedact()
    report = m.compliance_self_check()
    assert report["rule_compliant"] is True
    assert report["forbidden_fields_present"] == []
    assert report["ckpt_dir"].endswith("cotrain")
    assert report["action_type"] == "joint"


def test_only_one_model_instance():
    """实例只应持有唯一一个 model 引用，不存在模型字典/列表。"""
    m = _make_unifiedact()
    assert hasattr(m, "model")
    assert not isinstance(getattr(m, "model", None), (dict, list, tuple))


# ---------------------------------------------------------------------------
# 测试 2：prepare_case 不切换模型
# ---------------------------------------------------------------------------

def test_prepare_case_does_not_switch_model():
    """跨 12 任务 × {_random} 共 24 个 case 调用 prepare_case，
    self.model 引用必须保持恒定（id 不变）。"""
    m = _make_unifiedact()
    initial_model_id = id(m.model)
    initial_model_obj = m.model

    tasks = [
        "stack_bowls", "push_T", "pack_objects_into_box", "fold_clothes",
        "hang_mugs", "sweep_blocks", "pour_liquid_into_cup", "make_toast",
        "arrange_largest_number", "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones", "stack_blocks",
    ]
    for task in tasks:
        # 标准配置
        m.prepare_case({"action_case_id": task,
                        "evaluation_id": f"eval_{task}",
                        "trial_id": f"trial_{task}"})
        assert id(m.model) == initial_model_id, (
            f"prepare_case({task}) 不应改变 self.model 引用"
        )
        assert m.model is initial_model_obj
        # _random 配置
        m.prepare_case({"action_case_id": f"{task}_random",
                        "evaluation_id": f"eval_{task}_random",
                        "trial_id": f"trial_{task}_random"})
        assert id(m.model) == initial_model_id
        assert m.model is initial_model_obj

    assert m._case_counter == 24


def test_prepare_case_records_meta_but_no_model_call():
    """prepare_case 不应在底层模型上调用 reset/load_state_dict 等方法。"""
    m = _make_unifiedact()
    m.model.reset = MagicMock()
    m.prepare_case({"action_case_id": "hang_mugs",
                    "evaluation_id": "e1", "trial_id": "t1"})
    # prepare_case 不应触发底层模型 reset
    m.model.reset.assert_not_called()
    assert m.last_case_meta is not None
    assert m.last_case_meta["action_case_id"] == "hang_mugs"


# ---------------------------------------------------------------------------
# 测试 3：update_obs / get_action / reset 委托给同一模型
# ---------------------------------------------------------------------------

def test_update_obs_forwards_to_single_model():
    m = _make_unifiedact()
    obs = {"vision": {}, "state": {}}
    m.update_obs(obs)
    m.model.update_obs.assert_called_once_with(obs)


def test_get_action_returns_single_model_result():
    m = _make_unifiedact()
    expected = [{"action": [0.0] * 14}]
    m.model.get_action = MagicMock(return_value=expected)
    result = m.get_action()
    assert result is expected


def test_reset_calls_single_model_reset():
    m = _make_unifiedact()
    m.reset()
    m.model.reset.assert_called_once()


# ---------------------------------------------------------------------------
# 测试 4：构造时强制要求 ckpt_dir（单 ckpt 配置）
# ---------------------------------------------------------------------------

def test_init_requires_ckpt_dir():
    """没有 ckpt_dir 应在构造时立即报错（防止意外退化到多 ckpt 行为）。"""
    with patch.dict(sys.modules, {
        "torch": MagicMock(),
        "XPolicyLab.policy.ACT.model": MagicMock(),
    }):
        if "UnifiedACT.model" in sys.modules:
            del sys.modules["UnifiedACT.model"]
        from UnifiedACT.model import Model
        import pytest
        with pytest.raises(ValueError, match="ckpt_dir"):
            Model({"ckpt_name": "cotrain", "action_type": "joint"})


# ---------------------------------------------------------------------------
# 测试 5：动作类型与协议恒定
# ---------------------------------------------------------------------------

def test_action_type_immutable_after_construction():
    """action_type 在构造后不应被 prepare_case 改变。"""
    m = _make_unifiedact(action_type="joint")
    assert m.action_type == "joint"
    for task in ("hang_mugs", "fold_clothes", "make_toast"):
        m.prepare_case({"action_case_id": task})
        assert m.action_type == "joint"


if __name__ == "__main__":
    # 简单运行器（不依赖 pytest）
    try:
        import pytest
        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        tests = [v for k, v in sorted(globals().items())
                 if k.startswith("test_") and callable(v)]
        passed = 0
        failed = []
        for t in tests:
            try:
                t()
                print(f"  PASS  {t.__name__}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {t.__name__}: {e}")
                failed.append(t.__name__)
        print(f"\n{passed}/{len(tests)} tests passed")
        if failed:
            print(f"Failed: {failed}")
        sys.exit(0 if passed == len(tests) else 1)
