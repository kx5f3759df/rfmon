#!/usr/bin/env bash
# run_rfmon.sh
# 需求：
# 1) tmux 窗口 rfmon：source venv/bin/activate 后运行 rfmon_v2.py，若退出则重试
# 2) tmux 窗口 rfmon_web：source venv/bin/activate 后运行 http.server，若退出则重试
# 3) tmux 窗口 rfmon_report：source venv/bin/activate 后运行 "python3 aggr.py && python3 report.py"
#    第二个命令完成后，间隔 300 秒再次运行
#
# 统一设计：在开始定义四个 list，循环创建 tmux 会话并在会话里执行
# 注：默认重试/重跑前等待 DELAYS[i] 秒

set -u

# ---------- 配置区：四个列表 ----------
# tmux 会话名
NAMES=(
  "rfmon"
  "rfmon_web"
  "rfmon_report"
)

# 第一个命令（每轮都会先执行一次，比如激活虚拟环境）
CMD1=(
  "source venv/bin/activate"
  "source venv/bin/activate"
  "source venv/bin/activate"
)

# 第二个命令（核心循环命令；退出/完成后按 DELAYS 重试或重跑）
CMD2=(
  "python3 rfmon.py --f-range 400,520 --samp 2 --dwell 0.5 --gain 20 --overlap 10 --auto-threshold 16 --dc-khz 2"
  "python3 -m http.server 8000 --directory report"
  "nice -n 10 python3 aggr.py --dir ./ --out report/aggregated_signals.csv --link_time_max_min 30 --link_freq_base_mhz 0.0012 --link_freq_per_sec_mhz 0.0000002 --max_gap_min 10 --snap_to_canonical 1 --snap_tol_mhz 0.0015 --canon_use_first_n 5 --fuse_tol_mhz 0.0008 --round_freq 0.001 --stats_out stats.json"
)

# 第二个命令的重试/重跑延迟（秒）
# 注：前两个是“退出则重试”，第三个是“完成后每隔 300 秒再次运行”，统一用这个延迟处理
DELAYS=(
  5
  5
  300
)

# ---------- 函数区 ----------
session_exists() {
  local s="$1"
  tmux has-session -t "$s" 2>/dev/null
}

start_session() {
  local s="$1"
  local first="$2"
  local second="$3"
  local delay="$4"

  if session_exists "$s"; then
    echo "[INFO] tmux 会话已存在：$s（略过创建）"
    return 0
  fi

  # 创建一个以 bash 登录 shell 启动的 tmux 会话，避免 /bin/sh 下 source 失效
  tmux new-session -d -s "$s" "bash -l"
  echo "[OK] 已创建 tmux 会话：$s"

  # 在会话中发送循环脚本：
  # 逻辑：每轮
  #  1) 执行 CMD1（例如 source venv/bin/activate）
  #  2) 执行 CMD2
  #  3) 记录返回码，打印时间戳
  #  4) sleep 指定秒数
  #  5) 循环
  #
  # 对于 rfmon / rfmon_web：CMD2 一旦退出即重试（sleep 后进入下一轮）
  # 对于 rfmon_report：CMD2 是 "aggr && report"；完成即认为一轮结束，sleep 后再来一轮
  tmux send-keys -t "$s" "while true; do" C-m
  tmux send-keys -t "$s" "$first" C-m
  tmux send-keys -t "$s" "$second" C-m
  tmux send-keys -t "$s" 'rc=$?' C-m
  tmux send-keys -t "$s" 'echo "[${s}] $(date "+%F %T") => command finished/exit code: ${rc}"' C-m
  tmux send-keys -t "$s" "sleep $delay" C-m
  tmux send-keys -t "$s" "done" C-m

  echo "[OK] 已在会话 $s 中启动循环。"
}

# ---------- 主流程 ----------
# 基础依赖检查
if ! command -v tmux >/dev/null 2>&1; then
  echo "[ERROR] 未找到 tmux，请先安装 tmux。"
  exit 1
fi

# 三个列表长度一致性检查
if [[ ${#NAMES[@]} -ne ${#CMD1[@]} || ${#NAMES[@]} -ne ${#CMD2[@]} || ${#NAMES[@]} -ne ${#DELAYS[@]} ]]; then
  echo "[ERROR] 配置数组长度不一致，请检查 NAMES/CMD1/CMD2/DELAYS。"
  exit 1
fi

# 逐一启动
for i in "${!NAMES[@]}"; do
  start_session "${NAMES[$i]}" "${CMD1[$i]}" "${CMD2[$i]}" "${DELAYS[$i]}"
done

echo "[DONE] 所有会话已处理完成。使用 'tmux ls' 查看，'tmux attach -t rfmon' 等进入。"
