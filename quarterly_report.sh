#!/bin/bash
# Quarterly momentum report: run the scan manually and email the report to the
# recipient in mail_config.json. Use this when you do your once-every-3-months
# scan; a separate reminder (quarterly_reminder.sh) just tells you when.
#
#   ./quarterly_report.sh               run the scan + send email now
#   ./quarterly_report.sh --test-email  send a test email (no scan)
cd "$(dirname "$0")"
PROJECT="$(pwd)"
PYBIN=$(command -v python3)
LOGFILE="$PROJECT/logs/quarterly_report.log"

mkdir -p logs

echo "=== Quarterly momentum report: $(date) ===" >> "$LOGFILE"

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
