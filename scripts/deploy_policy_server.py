#!/usr/bin/env python3
# ====================================================================
# GOAI 具身未来赛道 — Policy Server 部署入口
# --------------------------------------------------------------------
# 本脚本是 XPolicyLab setup_policy_server.py 的便捷包装:
#   1. 加载 deploy.yml 配置 + 命令行 overrides
#   2. 委托 XPolicyLab.PolicyServer(二进制帧 WebSocket 协议)
#   3. 可选:启动 cloudflared 隧道暴露公网 wss:// 端点
#
# 真实协议(非 JSON):XPolicyLab client_server.ws.protocol
#   消息类型: HELLO / PREPARE_CASE / RESET / INFER / TRIAL_END / HEARTBEAT / CLOSE
#   Model 接口: prepare_case / update_obs / get_action / reset
#
# 用法:
#   python scripts/deploy_policy_server.py \
#       --config-path policy/UnifiedACT/deploy.yml \
#       --overrides host=0.0.0.0 port=19002
#
# 公网暴露(DSW 上):
#   cloudflared tunnel --url http://localhost:19002 &
#   # 隧道 URL 写入 CURRENT_ENDPOINT.txt,填入 X-Eval 提交表单
#
# 合规说明: UnifiedACT 仅包装单一 ACT ckpt, 评测期不切换模型/动作类型/协议。
# ====================================================================
import argparse
import ast
import os
import sys

import yaml


def _parse_val(s: str):
    """安全解析 override 值(支持数字/bool/None/list/dict)。"""
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def load_config(config_path: str, overrides: list[str] | None = None,
                host: str | None = None, port: int | None = None) -> dict:
    """加载 YAML 配置并应用命令行 overrides。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        # 支持 key=value key=value ... 或 --key value --key value ...
        if all("=" in t and not t.startswith("-") for t in overrides):
            for t in overrides:
                k, v = t.split("=", 1)
                cfg[k] = _parse_val(v)
        else:
            it = iter(overrides)
            for key in it:
                val = next(it)
                cfg[key.lstrip("-")] = _parse_val(val)

    if host is not None:
        cfg["host"] = host
    if port is not None:
        cfg["port"] = port

    for required in ("host", "port", "policy_name"):
        if not cfg.get(required):
            raise ValueError(f"{required} 必须在配置或 overrides 中指定")

    cfg.setdefault("protocol", "ws")
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="GOAI Policy Server 部署入口(委托 XPolicyLab PolicyServer)"
    )
    parser.add_argument("--config-path", "--config_path", dest="config_path",
                        required=True, help="deploy.yml 配置路径")
    parser.add_argument("--host", default=None, help="监听地址(覆盖配置)")
    parser.add_argument("--port", type=int, default=None, help="监听端口(覆盖配置)")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, default=None,
                        help="覆盖配置值,格式: key=value key=value ...")
    args = parser.parse_args()

    cfg = load_config(args.config_path, args.overrides, args.host, args.port)

    print(f"[deploy] policy={cfg['policy_name']} protocol={cfg['protocol']} "
          f"host={cfg['host']} port={cfg['port']}", flush=True)
    print(f"[deploy] config keys: {sorted(cfg.keys())}", flush=True)

    # 委托 XPolicyLab setup_policy_server.main(deploy_cfg)
    # 它会: importlib 加载 policy model -> 实例化 PolicyServer -> asyncio.run(serve_forever)
    from setup_policy_server import main as xpl_main
    xpl_main(cfg)


if __name__ == "__main__":
    main()
