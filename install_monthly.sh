#!/bin/bash
# Install a launchd LaunchAgent that runs the momentum scan + emails the report
# automatically on the 1st of every month at 7:30 PM (local time).
#
#   ./install_monthly.sh             install + (optionally) test email
#   ./install_monthly.sh --remove    uninstall
#   ./install_monthly.sh --test-only skip install, just send a test email
cd "$(dirname "$0")"
PROJECT="$(pwd)"
LABEL="com.momentum.advisor.monthly"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
HOUR=19
MINUTE=30

mkdir -p "$HOME/Library/LaunchAgents"

if [ "${1:-}" = "--remove" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Monthly LaunchAgent removed."
  exit 0
fi

# Send a test email first so failures surface before scheduling.
"$PROJECT/monthly_report.sh" --test-email
if [ $? -ne 0 ]; then
  echo "Test email FAILED — fix mail_config.json before installing the schedule."
  exit 1
fi
echo "Test email OK."

if [ "${1:-}" = "--test-only" ]; then
  echo "Test email sent; schedule not installed."
  exit 0
fi

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$PROJECT/monthly_report.sh</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Day</key><integer>1</integer>
    <key>Hour</key><integer>$HOUR</integer>
    <key>Minute</key><integer>$MINUTE</integer>
  </dict>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$PROJECT/logs/monthly_launchd.out.log</string>
  <key>StandardErrorPath</key><string>$PROJECT/logs/monthly_launchd.err.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "Monthly LaunchAgent installed: runs the scan + email on the 1st of each month at 19:30."
echo "Logs: $PROJECT/logs/monthly_report.log"
