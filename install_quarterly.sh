#!/bin/bash
# Install a launchd LaunchAgent that emails the quarterly scan reminder on the
# 4th of Feb / May / Aug / Nov at 21:00 (local time). The agent only sends the
# reminder email - the scan itself is manual.
#
#   ./install_quarterly.sh             install (after a test email)
#   ./install_quarterly.sh --remove    uninstall
#   ./install_quarterly.sh --test-only skip install, just send a test email
cd "$(dirname "$0")"
PROJECT="$(pwd)"
LABEL="com.momentum.advisor.quarterly"
OLD_LABEL="com.momentum.advisor.monthly"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
OLD_PLIST="$HOME/Library/LaunchAgents/$OLD_LABEL.plist"

mkdir -p "$HOME/Library/LaunchAgents"

if [ "${1:-}" = "--remove" ]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Quarterly reminder LaunchAgent removed."
  exit 0
fi

# Clean up the old monthly agent (no longer used).
if [ -f "$OLD_PLIST" ] || launchctl list 2>/dev/null | grep -q "$OLD_LABEL"; then
  launchctl bootout "gui/$(id -u)/$OLD_LABEL" 2>/dev/null || true
  rm -f "$OLD_PLIST"
  echo "Removed old monthly LaunchAgent ($OLD_LABEL)."
fi

# Send a test email first so failures surface before scheduling.
"$PROJECT/quarterly_report.sh" --test-email
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
    <string>$PROJECT/quarterly_reminder.sh</string>
  </array>
  <key>WorkingDirectory</key><string>$PROJECT</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Month</key>
    <array>
      <integer>2</integer>
      <integer>5</integer>
      <integer>8</integer>
      <integer>11</integer>
    </array>
    <key>Day</key><integer>4</integer>
    <key>Hour</key><integer>21</integer>
    <key>Minute</key><integer>0</integer>
  </dict>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$PROJECT/logs/quarterly_launchd.out.log</string>
  <key>StandardErrorPath</key><string>$PROJECT/logs/quarterly_launchd.err.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"

echo "Quarterly reminder LaunchAgent installed: emails the scan reminder on the"
echo "4th of Feb / May / Aug / Nov at 21:00 (first one: Aug 4, 2026)."
echo "The scan itself is manual - see the reminder email for instructions."
echo "Logs: $PROJECT/logs/quarterly_reminder.log"
