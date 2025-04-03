#!/bin/bash

LOG_DIR="/app/logs"
mkdir -p "$LOG_DIR"

AUDIO_LOG="$LOG_DIR/audio_inference.log"
PROCESS_LOG="$LOG_DIR/daily_report.log"

python3 src/debug_audio_monitoring.py > "$AUDIO_LOG" 2>&1 &
PID1=$!

python3 src/debug_process_detections.py > "$PROCESS_LOG" 2>&1 &
PID2=$!

tail -F "$AUDIO_LOG" "$PROCESS_LOG" &
TAIL_PID=$!

wait $PID1
wait $PID2

kill $TAIL_PID
