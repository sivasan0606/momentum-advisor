#!/bin/bash
# Quarterly reminder email: sends step-by-step instructions to run the scan
# yourself. Does NOT run the scan. Called by the
# com.momentum.advisor.quarterly LaunchAgent on the 4th of Feb/May/Aug/Nov.
#
#   ./quarterly_reminder.sh         send the reminder email now
cd "$(dirname "$0")"
PROJECT="$(pwd)"
PYBIN=$(command -v python3)
LOGFILE="$PROJECT/logs/quarterly_reminder.log"

mkdir -p logs

echo "=== Quarterly reminder: $(date) ===" >> "$LOGFILE"
"$PYBIN" "$PROJECT/advisor.py" --reminder \
  --email-config "$PROJECT/mail_config.json" \
  >> "$LOGFILE" 2>&1
rc=$?

echo "Exit code: $rc" >> "$LOGFILE"
echo "---" >> "$LOGFILE"
exit $rc
