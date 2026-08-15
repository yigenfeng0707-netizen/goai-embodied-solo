#!/usr/bin/env bash
# ====================================================================
# GOAI 具身未来赛道环境搭建脚本
# 基于 https://xsparkai.com/goai-2026/ 提交指南
# 严格遵循 X-Eval 平台官方命令，不使用任何未公开接口
# ====================================================================
set -e

echo "[1/4] 克隆 RoboDojo（含子模块）"
# 开启 git lfs（RoboDojo 部分资产走 LFS）
git lfs install
# --recurse-submodules 一次性拉取所有子模块
git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git
cd RoboDojo
# 备注：若仓库已存在，改用下面命令补全子模块：
# git submodule update --init --recursive

echo "[2/4] 安装 RoboDojo"
# 官方安装脚本：-i 表示交互式安装并创建 conda 环境 RoboDojo
bash scripts/install.sh -i
# 激活刚刚创建的 conda 环境（后续所有命令均在该环境中执行）
conda activate RoboDojo

echo "[3/4] 下载 GOAI 专用资源（跳过 init_assets.sh）"
# 先升级 huggingface_hub，确保支持 hf CLI
python -m pip install -U huggingface_hub
# 从 HuggingFace 拉取 RoboDojo 仓库的 Assets 目录（仿真资产）
# 官方提示：可跳过 RoboDojo 自带的 init_assets.sh，改用 hf 直接下载
hf download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset \
  --include "Assets/**" \
  --local-dir .
# 修正 embodiment 配置中的资产路径
python utils/update_embodiment_config_path.py

echo "[4/4] 下载 GOAI 专用数据"
# 拉取 GOAI-2026 数据集的 hdf5 训练 / 评测数据
hf download RoboDojo-Benchmark/GOAI-2026 \
  --repo-type dataset \
  --include "data/hdf5/**" \
  --local-dir .

echo "[done] 环境就绪"
echo "后续步骤："
echo "  1. bash scripts/setup_xpolicylab.sh  # 克隆 XPolicyLab policy 框架"
echo "  2. 在 policy/YOUR_POLICY/ 适配自有 policy"
echo "  3. bash scripts/smoke_test.sh         # 冒烟测试链路"
