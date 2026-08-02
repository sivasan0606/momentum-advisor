#!/bin/bash
# Monthly momentum report: run the scan and email the report to the recipient
# in mail_config.json. Called by the com.momentum.advisor.monthly LaunchAgent
# on the 1st of each month at 7:30 PM (local time).
#
#   ./monthly_report.sh                 run the scan + send email now
#   ./monthly_report.sh --test-email    send a test email (no scan)
cd "$(dirname "$0")"
PROJECT="$(pwd)"
PYBIN=$(command -v python3)
LOGFILE="$PROJECT/logs/monthly_report.log"

mkdir -p logs

echo "=== Monthly momentum report: $(date) ===" >> "$LOGFILE"

if [ "${1:-}" = "--test-email" ]; then
  "$PYBIN" "$PROJECT/advisor.py" --test-email --email-config "$PROJECT/mail_config.json" \
    >> "$LOGFILE" 2>&1
  rc=$?
else
  "$PYBIN" "$PROJECT/advisor.py" --email --cash 0 \
    --email-config "$PROJECT/mail_config.json" \
    --out "$PROJECT/advisor.html" >> "$LOGFILE" 2>&1
  rc=$?
fi

echo "Exit code: $rc" >> "$LOGFILE"
echo "---" >> "$LOGFILE"
exit $rc
