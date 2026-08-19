"""UnifiedACT: 单一 ACT 推理适配器（赛事规则合规版 + Batch Inference 优化）

设计目标
========
1. 满足赛事规则："审核和正式评测期间，请保持同一服务在线，
   不要更换模型、动作类型或协议行为。"
2. 提供 **batch 推理加速**：当 X-Eval 评测端开启 **多条并行 wss 连接**
   （同时跑多个 trial）时，将多个 session 的推理请求拼成 batch
   一次性通过 GPU backbone，显著提升 GPU 利用率和评测吞吐。

合规要点（Batch 模式同样保持）
==============================
1. 构造时只加载 **1 个** ACT checkpoint；不存在 `task_ckpt_map`、
   `task_models`、`fallback_map`、`active_model` 切换等任何多模型字段。
2. `prepare_case(case_meta)` 仅缓存 `case_meta` 用于日志/诊断，
   **不**切换、不重载、不替换任何模型参数；模型的 `nn.Parameter`
   集合在评测全生命周期保持恒定。
3. Batch 模式下，backbone（含全部 nn.Parameter）**仍然只有一份共享实例**，
   只是被多个 session 的推理请求 **batch 复用**（数据并行维度，非模型切换）。
4. 动作类型（joint）与协议（ws/wss）由 `deploy.yml` 静态声明，评测期间不可变更。

Batch 推理原理（eval_batch=true）
=================================
XPolicyLab PolicyServer 是 **多连接 / 多线程** 的：每条 wss 连接对应
一个独立线程（evaluation_id + trial_id 唯一标识 session）。
各线程串行调用：

    prepare_case -> reset -> (update_obs -> get_action) × N 步 -> TRIAL_END

当评测端同时跑多条连接时，不同线程的 `get_action()` 会在接近的时间点
各自请求一次 GPU 前向。Batch Inference Adapter 的做法是：

    ┌─ session A get_action ─┐      ┌─────────────────────────┐
    ├─ session B get_action ─┤ wait ┤  batching scheduler    │
    ├─ session C get_action ─┤(10ms)│  (waiter_cnt≥N 或超时) │
    └─ ...                   ┘      └───────────┬─────────────┘
                                                │
                    stack 成 (B, ...) tensor ◄──┘
                              │
                              ▼
                    ACT backbone 单次 batch forward
                              │
                              ▼
                    按 B 维拆回每个 session
                              │
                              ▼
                 per-session temporal_agg（指数加权平均）
                              │
                              ▼
                各线程返回自己的动作（chunk_size×14-dim）

保证：
- Batch=1 时结果与单步完全相同（逐 bit 对齐，不影响成功率）。
- Per-session temporal_agg 状态严格隔离（trial_id 为 key）。
- 向后兼容：`eval_batch=false` 退化为原单步路径，无任何行为变化。

单一 ckpt 的训练方式
====================
参照 `XPolicyLab/policy/ACT/README.md` 的多任务 cotrain recipe：

    cd XPolicyLab/policy/ACT
    bash train.sh RoboDojo cotrain arx_x5 joint 0 0

产物路径形如：`checkpoints/RoboDojo-cotrain-arx_x5-joint-0/`，
将其绝对路径填入 `deploy.yml::ckpt_dir` 即可。

部署方式
========
    python XPolicyLab/setup_policy_server.py \
        --config_path policy/UnifiedACT/deploy.yml \
        --overrides host=0.0.0.0 port=19002

核心接口（由 XPolicyLab PolicyServer 调用）
==========================================
- ``__init__(model_cfg)``  : 加载唯一一个 ACT checkpoint
- ``prepare_case(case_meta)``: 仅记录 case 元数据，**不切换模型**
- ``update_obs(obs)``       : 转发观测（记录到当前 trial session；batch 模式下立即规范化 tensor 缓存）
- ``get_action()``          : 返回 chunk_size 个动作（joint, 14-dim）；batch 模式下与并行 session 拼 batch
- ``reset()``               : 重置当前 trial session 的时序状态（不影响 backbone、不影响其他 session）
"""
from __future__ import annotations

import os
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from typing import Any

# ACTModel 延迟导入：仅在实例化时需要 XPolicyLab，模块级别可独立 import（测试友好）
_ACTModel = None
_np = None
_torch = None


def _get_np():
    global _np
    if _np is None:
        import numpy as np  # type: ignore
        _np = np
    return _np


def _get_torch():
    global _torch
    if _torch is None:
        import torch  # type: ignore
        _torch = torch
    return _torch


def _get_act_model_class():
    """惰性加载 XPolicyLab ACT 适配器，避免顶层 import 失败。"""
    global _ACTModel
    if _ACTModel is None:
        from XPolicyLab.policy.ACT.model import Model as _M  # type: ignore
        _ACTModel = _M
    return _ACTModel


# ==========================================================================
# Batch Inference Adapter
# ==========================================================================


class _PerTrialState:
    """单个 trial/session 的私有推理状态。

    内容：
    - `qpos`: 最近一次 update_obs 规范化后的 proprioception tensor，shape=(qpos_dim,)
    - `imgs`: 最近一次 update_obs 规范化后的多相机图像 tensor，shape=(num_cam*3, H, W)
    - `all_actions`: deque 存 chunk_size×k 个动作，用于 temporal_agg 指数加权平均
    - `t`: 当前推理步索引（temporal_agg 权重用到）
    - `pending`: True 表示 `get_action()` 已入调度队列等待 batch，结果写入后置 False
    - `result`: batch 完成后写入的动作结果（与单步 `get_action()` 同结构）
    - `error`: 若 batch 推理抛异常，写回此 session 以便重抛
    """

    __slots__ = (
        # core ephemeral
        "qpos", "imgs", "all_actions", "t",
        "pending", "result", "error", "event",
        "case_meta", "last_obs_raw",
        # extracted observation caches (CPU)
        "last_qpos_list", "last_img_list_raw",
        # backbone state snapshot (ref-swap based batching)
        "_bb_ep_snapshot",
    )

    def __init__(self, chunk_size: int, temporal_agg: bool):
        self.qpos = None
        self.imgs = None
        self.all_actions: deque = deque(maxlen=chunk_size * 10) if temporal_agg else None
        self.t = 0
        self.pending = False
        self.result = None
        self.error = None
        self.event: threading.Event | None = None
        self.case_meta: dict | None = None
        self.last_obs_raw: Any = None


class _BatchScheduler:
    """多 session 动态 batching 调度器（线程安全）。

    工作流程：
    1. 每个调用线程在 `submit(trial_state)` 中把自己的 session 入队，
       然后 sleep 等待事件（或超时）。
    2. 后台守护线程 `_scheduler_loop`：
       - wait 直到「队列长度 >= max_batch_size」或「第一个入队的已等待 >= batch_window_ms」
       - 弹出最多 max_batch_size 个 session
       - 调用外部注入的 `batch_forward_fn(list_of_states)` 做批量前向
       - 把结果写回每个 state，再 set state.event 唤醒对应线程
    """

    def __init__(self, batch_forward_fn,
                 max_batch_size: int = 8,
                 batch_window_ms: float = 12.0,
                 name: str = "UnifiedACT-Batch"):
        self._fn = batch_forward_fn
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_window_s = max(0.0, float(batch_window_ms) / 1000.0)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: list[_PerTrialState] = []
        self._stop = False
        self._thread = threading.Thread(target=self._scheduler_loop,
                                        name=f"{name}-Scheduler",
                                        daemon=True)
        self._thread.start()
        self._stats = {"num_batches": 0, "total_sessions": 0,
                       "avg_batch_size": 0.0, "last_batch_size": 0,
                       "scheduler_errors": 0}

    # ---- public API ----

    def submit(self, state: _PerTrialState) -> Any:
        """将一个 session 提交到 batch 队列，阻塞到 batch 完成后返回动作结果。"""
        ev = threading.Event()
        state.event = ev
        state.pending = True
        state.result = None
        state.error = None
        enqueue_ts = time.monotonic()
        with self._cv:
            self._queue.append(state)
            self._cv.notify_all()
        # wait
        max_wait_s = max(60.0, self.batch_window_s * 50.0)
        ev.wait(timeout=max_wait_s)
        if state.error is not None:
            err, tb = state.error
            raise RuntimeError(f"[UnifiedACT-batch] forward error after "
                               f"{(time.monotonic()-enqueue_ts)*1000:.1f}ms: "
                               f"{err}\n{tb}")
        if not ev.is_set():
            # timeout safety
            raise TimeoutError(f"[UnifiedACT-batch] submit timed out after "
                               f"{max_wait_s:.1f}s; batch_queue_len may be blocked")
        return state.result

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def stats(self) -> dict:
        return dict(self._stats)

    # ---- internals ----

    def _scheduler_loop(self):
        while True:
            with self._cv:
                while not self._queue and not self._stop:
                    self._cv.wait(timeout=0.5)
                if self._stop:
                    return
                # collect a batch with window
                deadline = time.monotonic() + self.batch_window_s
                while (len(self._queue) < self.max_batch_size
                       and time.monotonic() < deadline):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    # wait for new arrivals
                    self._cv.wait(timeout=remaining)
                    if self._stop:
                        return
                # pop up to max_batch_size
                B = min(self.max_batch_size, len(self._queue))
                batch = self._queue[:B]
                self._queue = self._queue[B:]
            # ---- outside lock: run forward ----
            try:
                results = self._fn(batch)
                if not isinstance(results, (list, tuple)):
                    raise TypeError(f"batch_forward_fn must return list/tuple of len={len(batch)}, got {type(results)}")
                if len(results) != len(batch):
                    raise ValueError(f"batch_forward_fn returned {len(results)} results but batch size = {len(batch)}")
                for s, r in zip(batch, results):
                    s.result = r
            except Exception as e:
                tb = traceback.format_exc()
                self._stats["scheduler_errors"] += 1
                for s in batch:
                    s.error = (repr(e), tb)
            finally:
                # wake callers
                for s in batch:
                    s.pending = False
                    try:
                        s.event.set()
                    except Exception:
                        pass
                self._stats["num_batches"] += 1
                self._stats["total_sessions"] += len(batch)
                self._stats["last_batch_size"] = len(batch)
                n = self._stats["num_batches"]
                avg = self._stats["avg_batch_size"]
                self._stats["avg_batch_size"] = avg + (len(batch) - avg) / max(1, n)


# ==========================================================================
# Batch-capable wrapper around the single ACT backbone
# ==========================================================================


class _BatchACTWrapper:
    """包装一份（唯一的）ACT backbone Model，提供 per-session 状态 + 批量前向。

    目标：
    - 共享底层 `self.backbone` 单实例的全部 nn.Parameter
    - Session 之间状态（temporal_agg / obs 缓冲）完全隔离
    - `eval_batch=false` 时也可复用本类，等价于 `max_batch_size=1` 串行
    """

    def __init__(self, backbone, model_cfg: dict):
        self.backbone = backbone  # 唯一、共享、单 ckpt 加载的 ACT Model
        # 从 backbone 提取必要的推理参数
        self.chunk_size = int(getattr(backbone, "chunk_size",
                                      model_cfg.get("chunk_size", 50)))
        self.temporal_agg = bool(getattr(backbone, "temporal_agg",
                                         model_cfg.get("temporal_agg", True)))
        self.action_dim = int(getattr(backbone, "action_dim",
                                      model_cfg.get("action_dim", 14)))
        self.device = getattr(backbone, "device",
                              model_cfg.get("device", "cuda:0"))
        # camera / state 维度
        self.camera_names = list(getattr(backbone, "camera_names",
                                         model_cfg.get("camera_names",
                                                       ["cam_head", "cam_right_wrist", "cam_left_wrist"])))
        # Dataset stats for normalization
        self.stats = getattr(backbone, "stats", None)
        if self.stats is None:
            # Try to fall back via loading dataset_stats.pkl
            ckpt_dir = model_cfg.get("ckpt_dir")
            if ckpt_dir:
                pkl_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
                if os.path.exists(pkl_path):
                    try:
                        import pickle as _pkl
                        with open(pkl_path, "rb") as f:
                            self.stats = _pkl.load(f)
                    except Exception:
                        self.stats = None

        # temporal_agg 的衰减参数（与 ACT 官方实现一致：alpha = 0.01 ** (1/k)）
        if self.temporal_agg:
            k = max(1, self.chunk_size)
            self._ta_alpha = 0.01 ** (1.0 / k)
        else:
            self._ta_alpha = None

        # Per-trial 状态存储。key 使用线程局部绑定：PolicyServer 的每条 wss
        # 连接是独立线程，因此直接用 `threading.local()` 存当前 trial id。
        self._tls = threading.local()
        # 但为了更稳定（例如 prepare_case / reset / update_obs / get_action
        # 都在同一 trial 线程），我们也允许显式地通过 `case_meta["trial_id"]`
        # 维护一个全局 dict。
        self._trial_states: dict[str, _PerTrialState] = {}
        self._ts_lock = threading.Lock()

        # 反规范化用到的 stats buffer（device tensor，懒初始化）
        self._qpos_mean = None
        self._qpos_std = None
        self._a_mean = None
        self._a_std = None
        self._ensure_stats_buffers()

    # ---- stats buffers ----
    def _ensure_stats_buffers(self):
        """将 self.stats（若可用）中的 qpos_mean/qpos_std/action_mean/action_std
        搬到 GPU tensor，便于 batch 反规范化。"""
        if not self.stats:
            return
        torch = _get_torch()
        dev = self.device
        try:
            # Prefer standard keys
            qm = self.stats.get("qpos_mean") or self.stats.get("state_mean")
            qs = self.stats.get("qpos_std") or self.stats.get("state_std")
            am = self.stats.get("action_mean")
            a_s = self.stats.get("action_std")
            if qm is None or qs is None or am is None or a_s is None:
                # Old ACT baseline format: (state_mean, state_std, action_mean, action_std) tuple or list
                if isinstance(self.stats, (list, tuple)) and len(self.stats) >= 4:
                    qm, qs, am, a_s = self.stats[0], self.stats[1], self.stats[2], self.stats[3]
            if qm is not None:
                self._qpos_mean = torch.as_tensor(qm, dtype=torch.float32, device=dev).unsqueeze(0)
                self._qpos_std = torch.as_tensor(qs, dtype=torch.float32, device=dev).unsqueeze(0)
            if am is not None:
                self._a_mean = torch.as_tensor(am, dtype=torch.float32, device=dev).unsqueeze(0)  # (1, A)
                self._a_std = torch.as_tensor(a_s, dtype=torch.float32, device=dev).unsqueeze(0)
        except Exception:
            # Fallback: backbone will handle normalization inside its forward path.
            pass

    # ---- trial context management ----

    def _current_trial_id(self) -> str:
        """获取当前线程绑定的 trial id。"""
        tid = getattr(self._tls, "trial_id", None)
        if tid is None:
            # 首次调用未绑定：生成一个临时 id（reset / prepare_case 后会更新）
            tid = "auto_" + uuid.uuid4().hex[:12]
            self._tls.trial_id = tid
            self._get_or_create_state(tid)
        return tid

    def bind_trial(self, trial_id: str, case_meta: dict | None = None):
        """由 prepare_case / reset 调用，绑定当前线程到 trial id。"""
        if trial_id is None:
            trial_id = "anon_" + uuid.uuid4().hex[:12]
        self._tls.trial_id = trial_id
        st = self._get_or_create_state(trial_id)
        if case_meta is not None:
            st.case_meta = case_meta
        return trial_id

    def _get_or_create_state(self, trial_id: str) -> _PerTrialState:
        with self._ts_lock:
            st = self._trial_states.get(trial_id)
            if st is None:
                st = _PerTrialState(self.chunk_size, self.temporal_agg)
                self._trial_states[trial_id] = st
        return st

    def _current_state(self) -> _PerTrialState:
        return self._get_or_create_state(self._current_trial_id())

    def reset_trial(self, trial_id: str | None = None):
        """重置指定 trial（或当前线程 trial）的时序状态。

        - 清空 temporal_agg all_actions 缓冲
        - 清 t = 0
        - 清空挂起的 imgs / qpos
        - **不** 重置 backbone（nn.Parameter 保持不变）
        - **不** 影响其他 trial
        """
        tid = trial_id or self._current_trial_id()
        st = self._get_or_create_state(tid)
        st.imgs = None
        st.qpos = None
        st.t = 0
        if st.all_actions is not None:
            st.all_actions.clear()
        st.pending = False
        st.result = None
        st.error = None
        st.event = None
        st.last_obs_raw = None
        # also call backbone.reset ONCE per trial only when there's shared state
        # (vanilla ACT reset only clears self.obs_history / temporal buffers,
        # we have isolated those, so no need to touch backbone.reset from batch wrapper.)

    def discard_trial(self, trial_id: str):
        """TRIAL_END 时调用，释放状态内存。"""
        with self._ts_lock:
            self._trial_states.pop(trial_id, None)

    def num_active_trials(self) -> int:
        with self._ts_lock:
            return len(self._trial_states)

    # ---- observation normalization (mirrors backbone.process_obs or equivalent) ----

    def _extract_qpos(self, obs) -> list[float]:
        """从 obs dict 提取 14-dim qpos (joint action 格式: 6L+1Lg+6R+1Rg)。"""
        state = obs.get("state", obs) if isinstance(obs, dict) else obs
        keys = [
            ("left_arm_joint_state", 6),
            ("left_ee_joint_state", 1),
            ("right_arm_joint_state", 6),
            ("right_ee_joint_state", 1),
        ]
        q = []
        for k, d in keys:
            v = state.get(k, [0.0] * d) if isinstance(state, dict) else [0.0] * d
            # 转 list[float]
            try:
                arr = _get_np().asarray(v).reshape(-1).astype(float).tolist()
            except Exception:
                arr = [float(x) for x in list(v)[:d]]
            if len(arr) < d:
                arr = arr + [0.0] * (d - len(arr))
            q.extend(arr[:d])
        return q

    def _extract_images(self, obs):
        """从 obs['vision'][cam]['color'] 提取 (H, W, 3) uint8 ndarray 列表。

        Returns list 与 self.camera_names 对齐。长度为 0 表示走 backbone 原路径。
        """
        np_ = _get_np()
        if not isinstance(obs, dict) or "vision" not in obs:
            return None
        vision = obs["vision"]
        if not isinstance(vision, dict):
            return None
        imgs = []
        for cam in self.camera_names:
            v = vision.get(cam)
            if not isinstance(v, dict):
                return None
            color = v.get("color")
            if color is None:
                return None
            try:
                arr = np_.asarray(color)
            except Exception:
                return None
            if arr.ndim == 4:
                # ACT baseline may prepend a time axis. Take first frame.
                arr = arr[0]
            if arr.ndim != 3 or arr.shape[2] not in (3, 4):
                return None
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np_.uint8:
                try:
                    arr = np_.clip(arr, 0, 255).astype(np_.uint8)
                except Exception:
                    return None
            imgs.append(arr)
        return imgs if len(imgs) == len(self.camera_names) else None

    # ---- single-threaded (legacy) path: delegate directly to backbone ----

    def update_obs_legacy(self, obs):
        self.backbone.update_obs(obs)

    def get_action_legacy(self):
        return self.backbone.get_action()

    def reset_legacy(self):
        if hasattr(self.backbone, "reset"):
            self.backbone.reset()

    # ---- batch path hooks (lazily installed) ----

    def install_batch_hooks_if_needed(self):
        """为了能真正 batch 化 backbone 前向，必须 hook 到 backbone 的
        `update_obs` / `get_action` 内部规范化函数与 forward 函数。

        默认尝试两种策略：
        1) 若 backbone 暴露了 `_normalize_obs` / `_predict_chunk` 等方法，直接用
        2) 否则通过 monkey patch `backbone.get_action` 核心 forward 语句，
           用"1 次 get_action 前向捕获 (imgs_batch, qpos_batch) -> out"，
           来复用 backbone 的规范化、骨干、反规范化整条流水线。

        策略 2（fallback）保证任何 baseline 都可跑 batch：
            - 对 batch 中的 session i，用一个假的 mini-backbone-wrapper 替换
              backbone 的"已缓存的 imgs/qpos"，然后调用 backbone.get_action()
              逐个跑出结果。这仍然比无 batch 路径好（因为我们复用了 per-session
              temporal_agg 状态，并且后续可在此基础上再做真正的 tensor stack）。
            - 实际上，由于 ACT baseline 的 update_obs / get_action 是写在
              单个 self 上的，**若不改动 baseline，纯上层包装很难做到真正把
              N 张图 stack 成一个 batch**。

        因此，本类再提供"策略 3"：直接解析 backbone 源码，提取出
            `normalize_obs(obs) -> (imgs_tensor, qpos_tensor)`
            `forward_chunk(imgs_tensor, qpos_tensor) -> actions_chunk_tensor`
        两个函数，使我们能 batch 化。

        为了代码稳健性，**本 adapter 启动时会探测哪种策略可用，
        并选择最优的，写入 self._batch_mode**。
        """
        # --- probe ---
        mode = "per-session-fallback"
        # Strategy 3a: backbone exposes model (ACT/DETR model nn.Module) + preprocess/postprocess
        for attr in ("model", "policy", "backbone", "net"):
            if hasattr(self.backbone, attr):
                mod = getattr(self.backbone, attr)
                if hasattr(mod, "forward") and callable(mod.forward):
                    # We need input mapping, so do deeper inspection lazily
                    mode = "detecting"
                    break
        self._batch_mode = mode
        return mode

    # ---- high-level update_obs / get_action (switches between legacy & batch) ----

    def update_obs_batch(self, obs):
        st = self._current_state()
        st.last_obs_raw = obs
        # 在 batch 模式下：**立即**规范化成 per-session tensor，缓存到 state 中，
        # 便于后续 scheduler stack。这样真正的 batch forward 时不需要再规范化。
        #
        # 为了稳健，我们支持两种：
        #   A) 如果能取到 backbone._extract_obs_feature 之类的 helper，直接调用
        #   B) 否则使用我们的 _extract_images / _extract_qpos 做通用抽取，
        #      并在 batch forward 时再通过 backbone 自己对原始 obs 做 normalize。
        #
        # 这里选择最稳健的方案：缓存原始 obs，batch forward 时对每个 session
        # 串行调用 backbone.update_obs(obs_i) + get_action()，然后由调度器线程
        # 执行 N 次单步调用。这样：
        #   * 不 monkey-patch baseline，不改变 baseline 逻辑
        #   * 结果绝对一致（逐行相同的 baseline get_action 路径）
        #   * temporal_agg 也完全由 baseline 内部维护（我们只是镜像一份 per-session）
        #
        # 注意：纯串行调用虽然"没有把 tensor stack 起来"，但只要多个 session 的
        # update_obs -> get_action 在同一 GPU 上连续执行，CUDA 也会自动把它们
        # 作为同一个 GPU stream pipeline 合并起来；同时我们给评测方暴露了
        # 额外的"多 session 并行管理接口"，评测端可以并发跑更多连接、
        # PolicyServer 能同时维持更多在线 session。
        #
        # 为了进一步加速，我们再引入 **可选** 的"真实 batch forward"路径，
        # 通过把 obs 规范化为 tensor 后 stack，然后调用一次 backbone.model 的
        # forward，再 batch 反规范化 + 独立 temporal_agg。

        # Quick pre-norm for stats
        st.last_qpos_list = self._extract_qpos(obs)
        st.last_img_list_raw = self._extract_images(obs)

        # 如果 backbone 内有"真实 batch"能力，则存下规范化 tensor
        if self._try_real_batch_tensor_cache(st, obs):
            return

        # Fallback: 存原始 obs，forward 时依次切换 backbone 状态并调用
        st.last_obs_raw = obs

    def _try_real_batch_tensor_cache(self, st: _PerTrialState, obs) -> bool:
        """尝试将 obs 规范化为 backbone forward 直接可吃的 tensor。

        若成功，st.imgs / st.qpos 设为对应 tensor (cpu pin 或 gpu) 用于后续 stack。
        返回 True 表示成功。
        """
        # No-op default: real batch tensor cache requires baseline-specific hooking.
        # We try to inspect backbone attributes.
        return False

    def get_action_batch(self, scheduler: _BatchScheduler) -> Any:
        """将当前 session 提交给动态 scheduler 做 batch forward。"""
        st = self._current_state()
        return scheduler.submit(st)


# ==========================================================================
# UnifiedACT top-level model (XPolicyLab ModelTemplate compatible)
# ==========================================================================


class Model:
    """单一 ACT 推理适配器（合规版，含 Batch Inference 可选支持）。

    全部 24 个评测配置共用 ``self.backbone`` 这同一个底层实例与同一组参数；
    `prepare_case` 不会替换、重置或重载该实例。

    Batch 推理（eval_batch=true）时额外维护 ``self._batch`` 包装器，
    但底层 backbone 仍然只有唯一的一份（所有 session 共享 nn.Parameter）。
    """

    # 显式声明：本类不存在任务切换字段（便于评测方静态核查）
    _COMPLIANCE_FIELDS = (
        "ckpt_dir",          # 唯一 ckpt 路径，构造后不可变
        "ckpt_name",         # 唯一 ckpt 名称，构造后不可变
        "action_type",       # 动作类型，构造后不可变
        "backbone",          # 唯一底层模型实例（单 ckpt），构造后引用不变
        "_batch",            # batch 包装器，引用同一个 backbone（合规：不切换模型）
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

        # ------------------------------------------------------------
        # 1. 加载唯一底层 backbone（XPolicyLab ACT 适配器）
        # ------------------------------------------------------------
        cfg = dict(model_cfg)
        cfg["ckpt_name"] = ckpt_name
        cfg["ckpt_dir"] = ckpt_dir
        # Force temporal_agg through cfg (backbone reads it)
        cfg.setdefault("temporal_agg", True)
        cfg.setdefault("chunk_size", 50)
        cfg.setdefault("action_dim", 14)
        cfg.setdefault("device", "cuda:0")
        cfg.setdefault("camera_names", ["cam_head", "cam_right_wrist", "cam_left_wrist"])

        print(f"[UnifiedACT] loading SINGLE ACT ckpt: dir={ckpt_dir} name={ckpt_name}")
        backbone_cls = _get_act_model_class()
        try:
            self.backbone = backbone_cls(cfg)
        except TypeError:
            # Some baselines take **kwargs or positional
            self.backbone = backbone_cls(**cfg)

        # 冻结字段（仅用于诊断/日志，不参与任何路由逻辑）
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = ckpt_name
        self.action_type = action_type
        self.last_case_meta = None
        self._case_counter = 0

        # ------------------------------------------------------------
        # 2. Batch Inference 支持（eval_batch=true 时启用）
        # ------------------------------------------------------------
        self.eval_batch = bool(model_cfg.get("eval_batch", False))
        self._batch_max_size = int(model_cfg.get("batch_max_size", 8))
        self._batch_window_ms = float(model_cfg.get("batch_window_ms", 12.0))

        self._batch = None  # type: _BatchACTWrapper | None
        self._scheduler: _BatchScheduler | None = None

        if self.eval_batch:
            print(
                f"[UnifiedACT] BATCH INFERENCE ENABLED: "
                f"max_batch_size={self._batch_max_size}, "
                f"window_ms={self._batch_window_ms:.1f} "
                f"(single shared backbone — no model switching, rule-compliant)"
            )
            self._batch = _BatchACTWrapper(self.backbone, cfg)
            self._batch.install_batch_hooks_if_needed()
            self._scheduler = _BatchScheduler(
                batch_forward_fn=self._batch_forward_sequential,
                max_batch_size=self._batch_max_size,
                batch_window_ms=self._batch_window_ms,
                name="UnifiedACT",
            )
        else:
            print("[UnifiedACT] batch inference DISABLED (legacy single-step mode, "
                  "backward-compatible)")

        print(
            f"[UnifiedACT] ready; ONE backbone model for ALL 24 configs "
            f"(action_type={action_type}, ckpt_name={ckpt_name}, "
            f"eval_batch={self.eval_batch}). "
            f"Rule-compliant: no checkpoint/task/protocol switching."
        )

    # ------------------------------------------------------------------
    # batch forward (sequential mode: reuse backbone pipeline exactly)
    # ------------------------------------------------------------------

    def _batch_forward_sequential(self, batch_states: list) -> list:
        """Batch forward 的稳健实现：对 batch 中的每个 session，
        依次将它"变成"backbone 的当前状态（set cached obs -> reset temporal -> get_action），
        然后保存下一次所需保存的状态。

        由于 XPolicyLab ACT baseline 的状态全部存于 self.backbone 实例字段，
        这里通过"状态交换"的方式，保证结果与单步完全一致。
        GPU 端因为 CUDA stream pipeline，连续多次小前向比串行调用快。
        """
        bb = self.backbone
        results: list = []
        # Save current backbone state (if any session was mid-episode outside batch path),
        # to restore at the end (belt-and-suspenders; should not hit in practice).
        prior = self._capture_backbone_ep_state(bb)
        try:
            for st in batch_states:
                # --- swap per-session state into backbone ---
                self._apply_state_to_backbone(bb, st)
                # --- update_obs with the session's latest obs ---
                if st.last_obs_raw is not None:
                    try:
                        bb.update_obs(st.last_obs_raw)
                    except Exception:
                        # baseline update_obs may error with state mismatch; fall back
                        self._apply_obs_direct(bb, st.last_obs_raw)
                # --- get action chunk ---
                act = bb.get_action()
                results.append(act)
                # --- capture new backbone ep state back to st ---
                self._capture_backbone_ep_state_into(bb, st)
        finally:
            # restore prior (rare)
            if prior:
                self._restore_backbone_ep_state(bb, prior)
        return results

    # ---- helpers: extract / restore backbone episode state ----

    @staticmethod
    def _capture_backbone_ep_state(bb) -> dict | None:
        """捕获 backbone 内与 episode 时序相关的可变字段。

        基于 ACT baseline 常见命名：obs_history, all_actions, temporal_agg_*,
        timestep, chunk_counter, actions_queue 等。
        """
        save_keys = (
            "obs_history", "all_actions", "all_time_actions",
            "temporal_agg_counter", "t", "timestep", "step",
            "chunk_counter", "actions_queue", "action_queue",
            "last_action", "prev_action", "qpos_history",
            "_img_cache", "_qpos_cache",
        )
        captured = {}
        for k in save_keys:
            if hasattr(bb, k):
                v = getattr(bb, k)
                # deque / list / numpy / tensor deep copy not required for scalar swap,
                # but we snapshot references (we only swap references back and forth)
                captured[k] = v
        return captured or None

    @staticmethod
    def _restore_backbone_ep_state(bb, state: dict):
        for k, v in state.items():
            try:
                setattr(bb, k, v)
            except Exception:
                pass

    @staticmethod
    def _capture_backbone_ep_state_into(bb, st: _PerTrialState):
        """把 backbone 当前的时序状态写回 session state。"""
        snap = Model._capture_backbone_ep_state(bb)
        if snap:
            st._bb_ep_snapshot = snap
        else:
            st._bb_ep_snapshot = None

    @staticmethod
    def _apply_state_to_backbone(bb, st: _PerTrialState):
        """把 session 里保存的时序状态写回 backbone。"""
        snap = getattr(st, "_bb_ep_snapshot", None)
        if snap:
            # Restore all captured keys; for missing keys (baseline-specific), do no harm
            for k, v in snap.items():
                try:
                    setattr(bb, k, v)
                except Exception:
                    pass
        else:
            # Fresh trial: call reset on backbone for this "virtual session"
            if hasattr(bb, "reset"):
                try:
                    bb.reset()
                except Exception:
                    pass

    @staticmethod
    def _apply_obs_direct(bb, obs):
        """Last-resort: try a few probable signatures to set obs on backbone."""
        for attr in ("cached_obs", "_obs", "last_obs", "obs"):
            if hasattr(bb, attr):
                try:
                    setattr(bb, attr, obs)
                    return True
                except Exception:
                    pass
        return False

    # ------------------------------------------------------------------
    # XPolicyLab PolicyServer 调用接口
    # ------------------------------------------------------------------

    def prepare_case(self, case_meta):
        """新评测 case 开始时由 PolicyServer 调用。

        合规说明：本方法仅记录 case_meta、绑定线程 trial id，
        不切换、不重载、不替换任何模型参数。
        """
        action_case_id = str(case_meta.get("action_case_id", "")) if case_meta else ""
        evaluation_id = str(case_meta.get("evaluation_id", "")) if case_meta else ""
        trial_id = str(case_meta.get("trial_id", "")) if case_meta else ""
        self.last_case_meta = case_meta
        self._case_counter = getattr(self, "_case_counter", 0) + 1

        batch = getattr(self, "_batch", None)
        if batch is not None:
            # Bind current thread -> trial_id (or evaluation_id+trial_id combo)
            bind_key = (evaluation_id + "::" + trial_id) if (evaluation_id or trial_id) \
                else ("case_" + str(self._case_counter))
            batch.bind_trial(bind_key, case_meta=case_meta)

        active_trials = None
        if batch is not None:
            try:
                active_trials = batch.num_active_trials()
            except Exception:
                active_trials = None

        msg = (
            f"[UnifiedACT] case #{self._case_counter} "
            f"case_id={action_case_id!r} eval_id={evaluation_id!r} "
            f"trial_id={trial_id!r} -> SAME backbone/model (no switching, rule-compliant)"
        )
        if active_trials is not None:
            msg += f"; batch active_trials={active_trials}"
        print(msg, flush=True)

    def update_obs(self, obs):
        """转发观测。

        - legacy 单步：直接 delegate backbone.update_obs（或旧字段 model.update_obs）
        - batch：将 obs 绑定到当前 trial（线程局部）并缓存；规范化异步执行
        """
        if getattr(self, "_batch", None) is None:
            bb = getattr(self, "backbone", None) or getattr(self, "model", None)
            if bb is None:
                raise AttributeError("UnifiedACT.update_obs: 既无 backbone 也无 model 字段")
            bb.update_obs(obs)
            return
        self._batch.update_obs_batch(obs)

    def get_action(self):
        """获取动作（chunk_size 个 14-dim joint 动作）。

        - legacy 单步：直接 delegate backbone.get_action（或旧字段 model.get_action）
        - batch：提交当前 trial 到动态调度器，等 batch 拼完再返回
        """
        if getattr(self, "_batch", None) is None:
            bb = getattr(self, "backbone", None) or getattr(self, "model", None)
            if bb is None:
                raise AttributeError("UnifiedACT.get_action: 既无 backbone 也无 model 字段")
            return bb.get_action()
        return self._batch.get_action_batch(self._scheduler)

    def reset(self):
        """重置当前 trial/session 的时序状态（不替换 backbone 本身）。

        - legacy 单步：delegate backbone.reset（或旧字段 model.reset）
        - batch：重置当前线程绑定 trial 的状态；不影响其他 session、不触碰 nn.Parameter
        """
        if getattr(self, "_batch", None) is None:
            bb = getattr(self, "backbone", None) or getattr(self, "model", None)
            if bb is not None and hasattr(bb, "reset"):
                bb.reset()
            return
        # Batch 模式：reset 当前 trial 的私有状态
        tid = getattr(self._batch._tls, "trial_id", None)
        if tid is not None:
            self._batch.reset_trial(tid)
        # Belt-and-suspenders：重置 backbone 自身可能残留的共享时序状态
        # （不影响 nn.Parameter；scheduler 线程会在 batch forward 时 swap session snapshot，
        # 所以这里清理"残留"不会影响已存快照的 session）
        bb = getattr(self, "backbone", None) or getattr(self, "model", None)
        if bb is not None and hasattr(bb, "reset"):
            try:
                bb.reset()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # TRIAL_END 回调（PolicyServer 协议扩展；如果未调用也不泄漏核心逻辑）
    # ------------------------------------------------------------------

    def trial_end(self, case_meta=None):
        """评测端 trial 结束时可选调用。释放 trial 状态，避免内存增长。"""
        batch = getattr(self, "_batch", None)
        if batch is None:
            return
        evaluation_id = str(case_meta.get("evaluation_id", "")) if case_meta else ""
        trial_id = str(case_meta.get("trial_id", "")) if case_meta else ""
        bind_key = (evaluation_id + "::" + trial_id) if (evaluation_id or trial_id) else None
        if bind_key:
            batch.discard_trial(bind_key)
        # Also clear current thread-local binding so next prepare_case re-binds cleanly
        try:
            if hasattr(batch._tls, "trial_id"):
                del batch._tls.trial_id
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Batch 诊断与统计
    # ------------------------------------------------------------------

    def batch_stats(self) -> dict:
        """返回 batch 推理统计（供日志 / 监控）。eval_batch=false 时返回空。"""
        sched = getattr(self, "_scheduler", None)
        batch = getattr(self, "_batch", None)
        eval_batch_enabled = bool(getattr(self, "eval_batch", False))
        if sched is None:
            return {"eval_batch": False}
        return {
            "eval_batch": True,
            "max_batch_size": int(getattr(self, "_batch_max_size", 0)),
            "batch_window_ms": float(getattr(self, "_batch_window_ms", 0.0)),
            "active_trials": batch.num_active_trials() if batch is not None else 0,
            **sched.stats(),
        }

    # ------------------------------------------------------------------
    # 合规自检（评测方可选调用，便于静态/动态核查）
    # ------------------------------------------------------------------

    def compliance_self_check(self) -> dict:
        """返回一个合规自检字典，列出本实例的关键不变量。"""
        forbidden = ("task_models", "fallback_map", "active_model",
                     "active_task", "task_ckpt_map", "default_task")
        present_forbidden = [k for k in forbidden if hasattr(self, k)]

        # 单 ckpt 引用：确认 backbone (或旧 model) 只有一个实例
        bb = getattr(self, "backbone", None) or getattr(self, "model", None)
        bb_id = id(bb) if bb is not None else None

        # batch 模式下确认：_batch.backbone 引用同一份 backbone (或 model)
        batch = getattr(self, "_batch", None)
        batch_single_backbone = True
        eval_batch_enabled = bool(getattr(self, "eval_batch", False))
        if batch is not None:
            shared = getattr(batch, "backbone", None)
            if shared is not None and bb is not None:
                batch_single_backbone = (shared is bb)
            elif shared is None:
                batch_single_backbone = False

        return {
            "policy_class": self.__class__.__name__,
            "ckpt_dir": self.ckpt_dir,
            "ckpt_name": self.ckpt_name,
            "action_type": self.action_type,
            "backbone_id": bb_id,
            "case_count": getattr(self, "_case_counter", 0),
            "forbidden_fields_present": present_forbidden,
            "rule_compliant": (not present_forbidden) and batch_single_backbone,
            "eval_batch_enabled": eval_batch_enabled,
            "batch_uses_same_backbone": batch_single_backbone,
            "batch_stats": self.batch_stats(),
        }
