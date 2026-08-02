#!/bin/bash
# Install a launchd LaunchAgent so the advisor server:
#   * starts automatically at login,
#   * stays running (KeepAlive restarts the server process directly if it dies),
#   * keeps your holdings/cash in the browser across reboots.
#
# The LaunchAgent runs `python3 advisor.py --serve` DIRECTLY as the job (not a
# launcher script), so KeepAlive restarts the real server without looping.
#
#   ./install_launchd.sh            install + start now
#   ./install_launchd.sh --remove   uninstall
cd "$(dirname "$0")"
PROJECT="$(pwd)"
PORT=8765
PLIST="$HOME/Library/LaunchAgents/com.momentum.advisor.plist"
LABEL="com.momentum.advisor"

mkdir -p "$HOME/Library/LaunchAgents" logs

if [ "${1:-}" = "--remove" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "LaunchAgent removed. Run ./stop.sh if a background server is still up."
  exit 0
fi

# A background (nohup) server would fight the LaunchAgent for the port.
./stop.sh >/dev/null 2>&1

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PROJECT/.venv/bin/python</string>
    <string>$PROJECT/advisor.py</string>
    <string>--serve</string>
    <string>--port</string>
    <string>$PORT</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$PROJECT/logs/launchd.out.log</string>
  <key>StandardErrorPath</key><string>$PROJECT/logs/launchd.err.log</string>
</dict>
</plist>
EOF

# Use the same interpreter serve.sh uses (fall back to plain python3).
PYBIN=$(command -v python3)
sed -i '' "s|$PROJECT/.venv/bin/python|$PYBIN|" "$PLIST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

sleep 4
if curl -s -o /dev/null --max-time 3 "http://localhost:$PORT/advisor.html"; then
  echo "LaunchAgent installed and server is up:"
  echo "  http://localhost:$PORT/advisor.html"
  echo "It now starts at login and restarts automatically if it stops."
else
  echo "LaunchAgent installed. If the server is not up yet, check logs/launchd.err.log"
  echo "(it may still be downloading prices on first scan)."
fi
