#!/bin/bash
# Start the momentum advisor server in the background and keep it running.
# Safe to run repeatedly: if it is already up it just prints the URL.
# Note: if a launchd LaunchAgent (install_launchd.sh) manages the server,
# this script detects the open port and simply reports the URL.
#
#   ./serve.sh            start (or report already running)
#   ./stop.sh             stop it
#   ./install_launchd.sh  optional: auto-start at login + restart on crash
cd "$(dirname "$0")"

PORT=8765
PIDFILE=server.pid
LOGFILE=logs/advisor_server.log
URL="http://localhost:${PORT}/advisor.html"

mkdir -p logs

# Already running (managed by launchd or a previous background start)?
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Advisor server is already running."
  echo "Open:  $URL"
  echo "Logs:  $(pwd)/$LOGFILE"
  exit 0
fi

# Stale pid file from a dead process?
if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE")
  if ! kill -0 "$PID" 2>/dev/null; then
    rm -f "$PIDFILE"
  fi
fi

echo "Starting advisor server in the background..."
nohup python3 advisor.py --serve > "$LOGFILE" 2>&1 &
PID=$!
echo $PID > "$PIDFILE"

# Wait until it answers (up to 60s; first start may download price history)
for i in $(seq 1 60); do
  if curl -s -o /dev/null --max-time 2 "$URL"; then
    echo "Advisor server is running (pid $PID)."
    echo "Open:  $URL"
    echo "Logs:  $(pwd)/$LOGFILE"
    echo "Stop:  ./stop.sh   (or make it permanent: ./install_launchd.sh)"
    exit 0
  fi
  sleep 1
done

echo "Server did not become ready in time — check the log: $LOGFILE"
echo "Tip: it may still be downloading price history on first scan."
exit 1
