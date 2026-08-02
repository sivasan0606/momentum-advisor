#!/bin/bash
# Stop the background advisor server (pid from server.pid, plus any on the port).
cd "$(dirname "$0")"
PORT=8765
PIDFILE=server.pid

if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null && echo "Stopped advisor server (pid $PID)."
  fi
  rm -f "$PIDFILE"
else
  echo "No server.pid file."
fi

# Fallback: kill whatever is listening on the port
PIDS=$(lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$PIDS" ]; then
  kill $PIDS 2>/dev/null && echo "Stopped process(es) on port $PORT."
fi

echo "Done."
