#!/usr/bin/env bash
# TopicGuard Service Manager (start / --restart / --stop / --status)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
PID_FILE="$DATA_DIR/topic_guard.pid"
LOG_FILE="$DATA_DIR/topic_guard.log"

mkdir -p "$DATA_DIR"

is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

do_stop() {
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo "[TopicGuard] Stopping PID $pid..."
        kill "$pid" 2>/dev/null || true
        for _ in {1..20}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 0.2
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        echo "[TopicGuard] Stopped."
    else
        rm -f "$PID_FILE"
        echo "[TopicGuard] Not running."
    fi
}

do_status() {
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo "[TopicGuard] Running (PID $pid)"
        echo "[TopicGuard] Log: $LOG_FILE"
        tail -n 10 "$LOG_FILE" 2>/dev/null || true
        return 0
    else
        echo "[TopicGuard] Status: STOPPED"
        return 1
    fi
}

ACTION="${1:-start}"

case "$ACTION" in
    --stop|stop)
        do_stop
        ;;
    --status|status)
        do_status
        ;;
    --restart|restart|start)
        if is_running; then
            do_stop
        fi
        echo "[TopicGuard] Starting bot daemon..."
        nohup python3 -u "$SCRIPT_DIR/main.py" >> "$LOG_FILE" 2>&1 &
        NEW_PID=$!
        echo "$NEW_PID" > "$PID_FILE"
        sleep 1.2
        if is_running; then
            echo "[TopicGuard] Started successfully (PID $NEW_PID)."
            tail -n 8 "$LOG_FILE" 2>/dev/null || true
        else
            echo "[TopicGuard] Failed to start! Check $LOG_FILE:"
            tail -n 20 "$LOG_FILE" 2>/dev/null || true
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [start|--restart|--stop|--status]"
        exit 1
        ;;
esac
