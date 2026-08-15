# Policy Server 部署文档 - GOAI Embodied Solo

> 参赛团队：GOAI Solo Builder（个人参赛）｜ 赛题一：RoboDojo / X-Eval 平台
> 部署目标：公网可访问的 Policy Server，提供 wss:// 端点供 X-Eval 评测端调用

本文档说明 Policy Server 的部署要求、选项对比、TLS 配置、启动命令、稳定性保障与监控、以及 X-Eval 提交表单字段。

---

## 0. 当前部署方案（阿里云 ECS GPU + EIP）

**更新时间**：2026-07-23

### 部署架构

```
X-Eval 评测端 → wss://<ECS公网IP>:8443 → UnifiedACT Policy Server (GPU)
```

> 合规整改后：单一 cotrain ckpt + 单一模型实例，评测期不切换模型/动作类型/协议。

### ECS 实例要求

| 项目 | 要求 |
|------|------|
| GPU | A10 24G 显存（或 V100） |
| OS | Ubuntu 22.04 LTS |
| Python | 3.11（Miniconda 创建） |
| CUDA | 12.1+（适配 PyTorch 2.7.0+cu128） |
| 系统盘 | 100G+ |
| 数据盘 | 50G+（存放 RoboDojo） |
| 公网 IP | 弹性公网 IP（EIP） |
| 安全组 | 开放 TCP 8443 入方向 |

### 部署清单

#### 需同步的文件（实际 21G，已打包为 ecs_deploy.tar.gz，2026-07-23 09:17 完成）

| 目录 | 大小 | 用途 | 是否必需 |
|------|------|------|----------|
| RoboDojo/XPolicyLab/ | 50M | UnifiedACT 代码 + 1 cotrain checkpoint | ✅ 必需 |
| RoboDojo/ckpt/ | 5.1G | ACT 预训练 checkpoint | ✅ 必需 |
| RoboDojo/Assets/ | 14G | MuJoCo 渲染资源 | ✅ 必需 |
| RoboDojo/data/ | 4.8G | 训练数据 | ⚠️ 仅评测需要 |
| RoboDojo/env, env_cfg, ... | ~200M | 环境配置 | ✅ 必需 |

#### 已排除的文件（节省 48G）

| 目录 | 大小 | 排除原因 |
|------|------|----------|
| RoboDojo/XPolicyLab/policy/ACT/processed_data/ | 48G | ACT 预处理数据，仅训练用，推理不需要 |

### 部署步骤

1. **ECS 准备**：开通实例 + EIP + 安全组
2. **同步代码**：上传 `ecs_deploy.tar.gz` 到 ECS 并解压
3. **环境安装**：执行 `deploy_ecs.sh step1`
4. **启动服务**：执行 `deploy_ecs.sh step3`
5. **验证**：执行 `deploy_ecs.sh step4`

### 部署脚本

- **[deploy_ecs.sh](../scripts/deploy_ecs.sh)**：4 步部署脚本（step1 环境 / step2 同步说明 / step3 启动 / step4 验证）
- **[stage4_ecs_package.py](../scripts/stage4_ecs_package.py)**：DSW 端打包脚本

### 与 DSW 方案的切换

- DSW 是训练环境，ECS 是部署环境
- 训练完成后，重新打包 `ecs_deploy.tar.gz` 上传到 ECS
- ECS 上 `bash deploy_ecs.sh step3` 重启服务即可切换版本

---

## 1. 部署要求

- **网络**：公网固定 IP 或稳定域名，X-Eval 评测端可从公网访问
- **协议**：WebSocket Secure（wss://），启用 TLS 加密
- **端点数量**：1-8 个 wss:// 端点
- **端口**：建议 8443 / 443 等标准 HTTPS 端口，防火墙放行
- **算力**：能加载 Policy checkpoint 并在合理延迟内返回 action（CPU 可运行，GPU 更佳）
- **稳定性**：审核期与正式评测期间持续在线

### 禁止事项

- 禁止提交 `localhost` / `127.0.0.1`
- 禁止提交局域网 IP（如 `192.168.x.x` / `10.x.x.x` / `172.16-31.x.x`）
- 禁止审核与正式评测期间更换模型、动作类型、协议行为

---

## 2. 部署选项对比

| 选项 | 公网 IP | 稳定性 | 成本 | 适用场景 |
|------|--------|--------|------|----------|
| 云服务器 | 固定公网 IP | 高 | 中（按月/按量） | **推荐**，长期稳定，易配 TLS |
| 花生壳内网穿透 | 无需固定 IP（需稳定域名） | 中 | 低 | 无公网 IP 时，依赖第三方穿透服务 |
| 反向代理（nginx） | 取决于前置主机 | 高 | 低 | 在云服务器或本机前置 nginx，TLS 终止 + WebSocket 代理 |

### 2.1 云服务器（推荐）

- 优势：固定公网 IP，带宽与稳定性可控，TLS 证书申请简单（Let's Encrypt）
- 部署：在云服务器上运行 Policy Server，开放 wss:// 端口
- 适用：审核期长时在线、正式评测高并发场景

### 2.2 花生壳内网穿透

- 适用：本地有 GPU 但无公网 IP 的场景
- 要求：需稳定域名 + TLS，避免穿透链路抖动影响评测
- 风险：第三方服务稳定性与带宽限制

### 2.3 反向代理（nginx）

- 用途：在 Policy Server 前置 nginx，由 nginx 完成 TLS 终止并代理 WebSocket
- 优势：Policy Server 本身可只跑 ws://，TLS 集中管理
- 配置要点：启用 `proxy_pass http://127.0.0.1:<backend>;` 与 WebSocket 升级头：
  ```nginx
  location / {
      proxy_pass http://127.0.0.1:8080;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
      proxy_set_header Host $host;
      proxy_read_timeout 3600s;
  }
  ```

---

## 3. TLS 证书

### 3.1 Let's Encrypt 免费证书（推荐，正式部署）

使用 certbot 申请免费证书（需域名）：

```bash
sudo apt install certbot
sudo certbot certonly --standalone -d your.domain.com
# 证书路径：
#   /etc/letsencrypt/live/your.domain.com/fullchain.pem
#   /etc/letsencrypt/live/your.domain.com/privkey.pem
```

优势：受浏览器与 wss 客户端信任，无需手动信任。

### 3.2 自签证书（仅测试）

```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=your.domain.com"
```

注意：自签证书在 X-Eval 评测端可能因证书校验失败而无法连接，仅用于本地联调，正式提交请使用受信任证书。

---

## 4. 启动命令示例

```bash
python scripts/deploy_policy_server.py \
  --host 0.0.0.0 \
  --port 8443 \
  --cert cert.pem \
  --key key.pem \
  --policy-dir XPolicyLab/policy/UnifiedACT \
  --ckpt cotrain \
  --action-type joint
```

参数说明：

| 参数 | 含义 |
|------|------|
| `--host` | 监听地址，`0.0.0.0` 表示监听所有网卡 |
| `--port` | 监听端口，需与提交表单端口一致 |
| `--cert` | TLS 证书路径（PEM） |
| `--key` | TLS 私钥路径（PEM） |
| `--policy-dir` | Policy 目录（XPolicyLab 适配结构） |
| `--ckpt` | Policy checkpoint 名称 |
| `--action-type` | 动作类型：`ee`（末端执行器）或 `joint`（关节角） |

启动后端点为：`wss://<公网IP或域名>:8443`

---

## 5. 稳定性要求

- **不更换**：审核与正式评测期间不更换模型、动作类型、协议行为
- **持续在线**：审核期可能跨数小时至数天，需保障服务持续可用
- **不回退**：每次提交对应一个稳定版本，提交后不再改动该版本

### 守护进程（推荐）

使用 systemd 或 supervisor 守护 Policy Server 进程，崩溃后自动重启：

- **systemd**：编写 `.service` 单元，`Restart=always`
- **supervisor**：配置 `autorestart=true`，`startsecs=10`

---

## 6. 监控

### 6.1 健康检查端点

建议 Policy Server 提供 HTTP 健康检查端点（如 `GET /health`），返回 200 OK，供监控探活。

### 6.2 日志

- 记录每次 wss 连接建立 / 断开
- 记录每次推理请求耗时与异常
- 日志按天滚动，保留至少 7 天

### 6.3 自动重启

- systemd：`Restart=always` + `RestartSec=5`
- supervisor：`autorestart=true` + `startretries=99`
- 可选：外部探活脚本 + 告警

---

## 7. X-Eval 提交表单字段

在 https://xsparkai.com/goai-2026/apply 填写以下字段：

| 字段 | 说明 |
|------|------|
| 队伍名称 | 参赛队伍名称 |
| 联系人 | 联系人姓名 |
| 手机号 | 联系电话 |
| 邮箱 | 联系邮箱 |
| Policy Server 主机 | 公网 IP 或域名（1-8 个端点） |
| Policy Server 端口 | 每个主机对应的端口（1-8 个） |
| 策略名称 | 仅字母、数字、下划线 |
| 动作类型 | `ee` 或 `joint` |

注意：

- 主机与端口数量须一致，最多 8 个端点
- 主机不得为 `localhost` / `127.0.0.1` / 局域网 IP
- 动作类型一经提交不得在审核期更改

---

## 8. 部署验证清单

- [ ] 公网固定 IP / 稳定域名就绪
- [ ] TLS 证书已申请并配置（推荐 Let's Encrypt）
- [ ] Policy Server 启动成功，`wss://host:port` 可访问
- [ ] 公网客户端连接 wss:// 响应正常，返回正确动作类型
- [ ] 长时间稳定性测试通过，无掉线
- [ ] 守护进程（systemd/supervisor）配置完成，崩溃自动重启
- [ ] 健康检查端点与日志就绪
- [ ] X-Eval 提交表单字段填写完整且合规
- [ ] 审核期与评测期不更换模型 / 动作类型 / 协议行为
