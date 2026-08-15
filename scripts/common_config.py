"""Common config for all stage scripts - DSW URL and paths."""
import os

# DSW instance URL (update when instance changes)
DSW_URL = os.environ.get("DSW_URL", "https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-2079390/lab")
PAI_DSW_URL = os.environ.get("PAI_DSW_URL", "https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-2045166/lab")
LAB_KEEPALIVE_URLS = ["https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-2078541/lab", "https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-2045166/lab"]
CPU_DSW_URL = os.environ.get("CPU_DSW_URL", "https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-2070748/lab")

# Local scripts directory
LOCAL_SCRIPTS_DIR = r"d:\APPs\GOAI_OpenSource\goai-embodied-solo\scripts"

# DSW remote paths
ROBODOJO_ROOT = "/mnt/workspace/RoboDojo"
XPL_ROOT = "/mnt/workspace/RoboDojo/XPolicyLab"
POLICY_SERVER_SCRIPT = f"{XPL_ROOT}/setup_policy_server.py"
DEMO_POLICY_DIR = f"{XPL_ROOT}/policy/demo_policy"
ACT_POLICY_DIR = f"{XPL_ROOT}/policy/ACT"

# Policy Server default port
POLICY_SERVER_PORT = 19000

# Skill scripts (dsw_remote.py location)
SKILL_SCRIPTS = r"c:\Users\52637\.trae-cn\skills\dsw-remote-execution\scripts"
