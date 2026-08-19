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


# ---------------------------------------------------------------------------
# 测试 6：Batch Inference — 合规性 & 功能正确性（mock，无需真实 ckpt）
# ---------------------------------------------------------------------------

def _make_unifiedact_batch(mock_backbone=None, eval_batch=True,
                           batch_max_size=4, batch_window_ms=10.0):
    """构造带 batch 支持的 UnifiedACT 实例，绕过真实 ACTModel 加载。

    注入 mock backbone：单步行为完全可控。
    """
    with patch.dict(sys.modules, {
        "torch": MagicMock(),
        "numpy": MagicMock(),
        "XPolicyLab.policy.ACT.model": MagicMock(),
    }):
        if "UnifiedACT.model" in sys.modules:
            del sys.modules["UnifiedACT.model"]
        from UnifiedACT.model import Model, _BatchACTWrapper, _BatchScheduler
        m = Model.__new__(Model)
        # 合规字段
        m.ckpt_dir = "/fake/cotrain"
        m.ckpt_name = "cotrain"
        m.action_type = "joint"
        m.last_case_meta = None
        m._case_counter = 0
        # 注入共享的 mock backbone（唯一实例引用）
        if mock_backbone is None:
            bb = MagicMock()
            bb.chunk_size = 50
            bb.temporal_agg = True
            bb.action_dim = 14
            bb.device = "cpu"
            bb.camera_names = ["cam_head", "cam_right_wrist", "cam_left_wrist"]
            bb.stats = None
            def _seq_action():
                # 返回带 deterministic 计数的动作，用于验证 session 间状态隔离
                _seq_action.calls = getattr(_seq_action, "calls", 0) + 1
                import numpy as np  # local; outer mock may not have this
                arr = np.full((50, 14), float(_seq_action.calls), dtype=np.float32)
                return {"action": arr}
            bb.get_action = MagicMock(side_effect=lambda: _seq_action())
            bb.update_obs = MagicMock()
            bb.reset = MagicMock()
            mock_backbone = bb
        m.backbone = mock_backbone
        # Batch 支持
        m.eval_batch = bool(eval_batch)
        m._batch_max_size = int(batch_max_size)
        m._batch_window_ms = float(batch_window_ms)
        if m.eval_batch:
            # 构造 cfg dict（与 _BatchACTWrapper.__init__ 参数匹配）
            cfg = {"ckpt_dir": m.ckpt_dir, "ckpt_name": m.ckpt_name,
                   "chunk_size": 50, "temporal_agg": True, "action_dim": 14,
                   "device": "cpu",
                   "camera_names": ["cam_head", "cam_right_wrist", "cam_left_wrist"]}
            m._batch = _BatchACTWrapper(m.backbone, cfg)
            m._scheduler = _BatchScheduler(
                batch_forward_fn=m._batch_forward_sequential,
                max_batch_size=m._batch_max_size,
                batch_window_ms=m._batch_window_ms,
                name="UT-UnifiedACT",
            )
        else:
            m._batch = None
            m._scheduler = None
        return m


def test_batch_mode_still_single_backbone_reference():
    """batch 模式下仍然只有一个 backbone 引用，合规。"""
    m = _make_unifiedact_batch(eval_batch=True)
    assert m._batch is not None
    assert m._batch.backbone is m.backbone
    # compliance_self_check 须通过
    r = m.compliance_self_check()
    assert r["rule_compliant"] is True, r
    assert r["eval_batch_enabled"] is True
    assert r["batch_uses_same_backbone"] is True


def test_batch_mode_no_forbidden_fields():
    """batch 模式下实例仍然没有多模型切换字段。"""
    m = _make_unifiedact_batch(eval_batch=True)
    forbidden = ("task_models", "fallback_map", "active_model",
                 "active_task", "task_ckpt_map", "default_task")
    for f in forbidden:
        assert not hasattr(m, f), f"batch 模式不应包含多模型字段 '{f}'"


def test_batch_scheduler_threading_smoke():
    """_BatchScheduler 的 batching 逻辑：多线程并发 submit → 合并为 1 次批调用。"""
    import threading, time
    call_log = []  # 每次 batch 调用记录 (batch_size, return_values)
    def _fn(states):
        call_log.append(len(states))
        # 返回每个 session 的动作（与 index 绑定的递增值）
        return [{"action_in_batch": i} for i in range(len(states))]

    from UnifiedACT.model import _BatchScheduler, _PerTrialState
    sched = _BatchScheduler(_fn, max_batch_size=8, batch_window_ms=40.0,
                            name="UT-Sched")
    # 启动 N 个线程同时 submit（用 barrier 保证尽可能接近同时）
    N = 6
    barrier = threading.Barrier(N)
    results = [None] * N
    errors = []

    def worker(idx):
        try:
            barrier.wait(timeout=2)
            st = _PerTrialState(50, True)
            st.last_obs_raw = {"idx": idx}
            r = sched.submit(st)
            results[idx] = r
        except Exception as e:
            errors.append((idx, repr(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=5)

    assert not errors, f"errors = {errors}"
    # 由于窗口 40ms + barrier ≈ 同时唤醒，全部 N 应合并进 1~2 个 batch
    assert sum(call_log) == N, f"total sessions processed: {call_log}, expected {N}"
    # 结果按 index 对应
    assert all(r is not None for r in results), "有线程未获结果"
    assert sorted(r["action_in_batch"] for r in results) == list(range(N)), \
        f"结果错位: {results}"
    sched.shutdown()


def test_batch_prepare_case_binds_trial_thread_locally():
    """prepare_case 按 thread-local 绑定 trial_id；不同线程各自隔离。"""
    import threading
    m = _make_unifiedact_batch(eval_batch=True)
    seen_ids = {}
    lock = threading.Lock()

    def run_thread(trial):
        m.prepare_case({"action_case_id": "task_a",
                        "evaluation_id": "E1", "trial_id": trial})
        tid = getattr(m._batch._tls, "trial_id", None)
        with lock:
            seen_ids[trial] = tid

    ts = [threading.Thread(target=run_thread, args=(f"T{i}",)) for i in range(4)]
    for t in ts: t.start(); t.join(timeout=3)
    # 每个 trial 绑定的 id 应唯一且包含其 trial_id
    vals = list(seen_ids.values())
    assert len(set(vals)) == 4, f"绑定应唯一: {vals}"
    for t, v in seen_ids.items():
        assert t in v, f"trial_id={t} 未出现在绑定 key {v}"
    # active_trials 应 >= 4
    assert m._batch.num_active_trials() >= 4


def test_batch_vs_legacy_identical_single_session():
    """batch 模式（单 session → batch_size=1）与 legacy 模式在相同输入
    下通过完全相同的 backbone 调用路径，得到同一返回值，保证不会改变成功率。
    """
    import numpy as np  # 必须在 patch sys.modules 之前真实导入
    # 共享同一个 deterministic backbone 实例：backbone.get_action 有可预测输出
    with patch.dict(sys.modules, {
        "torch": MagicMock(),
        "XPolicyLab.policy.ACT.model": MagicMock(),
    }):
        if "UnifiedACT.model" in sys.modules:
            del sys.modules["UnifiedACT.model"]
        from UnifiedACT.model import Model, _BatchACTWrapper, _BatchScheduler

        call_counter = {"n": 0}
        def _get_action():
            call_counter["n"] += 1
            return {"action": np.full((50, 14), call_counter["n"], dtype=np.float32)}

        # build deterministic shared backbone (NOT MagicMock so our call_counter side effect works)
        class _FakeBB:
            chunk_size = 50
            temporal_agg = True
            action_dim = 14
            device = "cpu"
            camera_names = ["cam_head", "cam_right_wrist", "cam_left_wrist"]
            stats = None
            def update_obs(self_inner, obs):
                # record last obs to verify path
                self_inner._last = obs
            def get_action(self_inner):
                return _get_action()
            def reset(self_inner):
                pass

        bb_leg = _FakeBB()
        bb_bat = _FakeBB()

        # ---- legacy path ----
        leg = Model.__new__(Model)
        leg.ckpt_dir = "/c"; leg.ckpt_name = "n"; leg.action_type = "joint"
        leg.last_case_meta = None; leg._case_counter = 0
        leg.backbone = bb_leg
        leg.eval_batch = False
        leg._batch = None; leg._scheduler = None

        # ---- batch path (B=1: window small, submit alone) ----
        bat = Model.__new__(Model)
        bat.ckpt_dir = "/c"; bat.ckpt_name = "n"; bat.action_type = "joint"
        bat.last_case_meta = None; bat._case_counter = 0
        bat.backbone = bb_bat
        bat.eval_batch = True
        bat._batch_max_size = 2
        bat._batch_window_ms = 1.0  # 极短窗口 → 单个 session 会立即执行
        cfg = {"ckpt_dir": "/c", "chunk_size": 50, "temporal_agg": True,
               "action_dim": 14, "device": "cpu",
               "camera_names": ["cam_head", "cam_right_wrist", "cam_left_wrist"]}
        bat._batch = _BatchACTWrapper(bat.backbone, cfg)
        bat._scheduler = _BatchScheduler(
            batch_forward_fn=bat._batch_forward_sequential,
            max_batch_size=2,
            batch_window_ms=1.0,
            name="UT-Ident",
        )

        # Run a mini episode on both sides with same obs
        obs = {"state": {"left_arm_joint_state": [0.1]*6, "left_ee_joint_state": [0.0],
                         "right_arm_joint_state": [0.2]*6, "right_ee_joint_state": [0.0]}}
        # Legacy
        call_counter["n"] = 0
        leg.reset()
        leg.update_obs(obs)
        a_leg = leg.get_action()
        # Batch (bind first via prepare_case then run)
        bat.prepare_case({"action_case_id": "t1", "evaluation_id": "e1", "trial_id": "single"})
        call_counter["n"] = 0
        bat.reset()
        bat.update_obs(obs)
        a_bat = bat.get_action()

        # 两条路径都会执行 backbone.update_obs + get_action 各一次
        assert hasattr(bb_leg, "_last"), "legacy update_obs 未调到 backbone.update_obs"
        # batch 模式下 _batch_forward_sequential 内部会调用 bb.update_obs(obs)
        assert hasattr(bb_bat, "_last"), "batch update_obs 未在 forward 时调到 backbone.update_obs"
        # 返回值结构与大小完全相同（逐值相同 → batch=1 逐位等价）
        assert set(a_leg.keys()) == set(a_bat.keys()), (a_leg.keys(), a_bat.keys())
        np.testing.assert_array_equal(a_leg["action"], a_bat["action"],
                                      err_msg="batch=1 时动作应与 legacy 逐位相同")
        # batch stats 正常
        bs = bat.batch_stats()
        assert bs["eval_batch"] is True
        assert bs["num_batches"] >= 1
        bat._scheduler.shutdown()


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
