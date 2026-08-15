# UnifiedACT policy package
#
# 合规声明：
#   本包仅包装 XPolicyLab ACT 单一 checkpoint，所有 24 个评测配置共用同一组参数。
#   不存在任务切换、checkpoint 切换、fallback 映射等多模型行为。
#   符合赛事规则："审核和正式评测期间，请保持同一服务在线，
#                  不要更换模型、动作类型或协议行为。"
from .model import Model

__all__ = ["Model"]
