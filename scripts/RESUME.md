# RESUME — 重新开机后继续（合规整改版）

> 状态：**COMPLIANT REWRITE**（2026-08-12 整改完成）。
> 旧 MultiACT 多 ckpt 切换方案已被赛事方拒绝，已删除并替换为 UnifiedACT（单一 ckpt）。
> NAS 数据在 `/mnt/workspace`；关掉 DSW/PAI **不会**丢 RoboDojo / cotrain ckpt / `goai_recover.sh`。

## 一句话

同一 NAS → 开新 Lab → 改 `DSW_URL` → `goai_recover.sh` → 拿 **新** Host → 冒烟 → **人工**决定 #3。

## 合规整改要点（必读）

1. 旧 `policy/MultiACT/` 目录已删除（按任务切换 ckpt 被官方认定为违规）
2. 新 `policy/UnifiedACT/` 仅包装 **唯一一个** cotrain ckpt
3. 必须 `bash scripts/train_unified_act.sh 0` 训练出 **单一** 多任务 ckpt
4. Policy Server 字段中 Policy 名称改为 `UnifiedACT`，动作类型保持 `joint`

## 开机清单

1. [魔搭 Notebook](https://www.modelscope.cn/my/mynotebook) 启动免费实例，**或** PAI `qinghuabisai`（workspace `maas_1708132065159793` / `dsw-2s287h5165zvapxmly`）。
2. 确认挂载同一 NAS：`df -h /mnt/workspace`；`test -d /mnt/workspace/RoboDojo`。
3. 更新本地 `scripts/common_config.py`：`DSW_URL` = 新 Lab URL。
4. **训练合规单 ckpt**（若 `/mnt/workspace/RoboDojo/ckpt/ACT/act-RoboDojo-cotrain/arx_x5-100-joint` 不存在）：
   ```bash
   bash /mnt/workspace/goai-embodied-solo/scripts/train_unified_act.sh 0
   ```
5. 启动合规服务：`bash /mnt/workspace/goai_recover.sh` → UnifiedACT `:19002` + cloudflared。
   - 若 `goai_recover.sh` 还是旧 MultiACT 版本，请重新生成（详见下方"recover 脚本更新"）
6. `cat /mnt/workspace/CURRENT_ENDPOINT.txt` → **新 Host**（旧 `blend-taxi-…` / #1/#2 Host 均作废）。
7. Windows：Chrome CDP `9222` + `python scripts/pai_lab_keepalive.py`（或 `modelscope_lab_keepalive.py`）。
8. Smoke：connect / prepare_case / get_action（不应出现 `switch` 字样，应看到 `SAME model`）。
9. **#3**：仅用户明确要求时提交；字段 Host=新隧道 Port=443 Policy=UnifiedACT Action=joint。

## #3 字段模板（Host 留空待填）

```
Host: <NEW after reopen>
Port: 443
Policy: UnifiedACT
Action: joint
```

## recover 脚本更新（若仍为旧 MultiACT 版本）

新 `goai_recover.sh` 启动命令应改为：

```bash
python XPolicyLab/setup_policy_server.py \
    --config_path policy/UnifiedACT/deploy.yml \
    --overrides host=0.0.0.0 port=19002 \
    > /mnt/workspace/unifiedact_server.log 2>&1 &
```

日志文件改名 `unifiedact_server.log`；监听端口仍为 19002 不变。

## 勿做

- 勿重新部署旧 MultiACT（违规方案）
- 勿假设关闭前 quick-tunnel Host 仍可用
- 勿在未确认 NAS 的情况下重下数据 / 重训
- 勿自动提交 #3

详见：`/mnt/workspace/goai_shutdown_2026_08_02.md`、仓库 `docs/pc-shutdown-runbook.md`。
