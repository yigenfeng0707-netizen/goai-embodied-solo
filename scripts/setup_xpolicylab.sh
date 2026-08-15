#!/usr/bin/env bash
# ====================================================================
# 克隆 XPolicyLab policy 框架（30+ baseline）
# XPolicyLab 提供了 ACT、Diffusion Policy、RDT 等 30+ baseline，
# 参赛者可在此基础上接入或适配自有 policy。
# 参考：https://github.com/XPolicyLab/XPolicyLab
# ====================================================================
set -e

# 克隆 XPolicyLab 主仓库（不含子模块，避免拉取过多无关依赖）
git clone https://github.com/XPolicyLab/XPolicyLab.git
cd XPolicyLab

echo "可用 baseline 列表："
ls policy/

echo "-----------------------------------------------------------------"
echo "请参照以下 README 完成 policy 适配："
echo "  https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/ACT/README.md"
echo "-----------------------------------------------------------------"
echo "适配完成后，将自有 policy 放入本项目 policy/YOUR_POLICY/ 目录。"
