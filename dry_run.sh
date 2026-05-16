#!/usr/bin/env bash
# dry_run.sh — clean state and run the demo end-to-end.
#
# Usage:
#   ./dry_run.sh           # one run with a fresh database
#   ./dry_run.sh keep      # keep DB; just run demo
set -e

if [ "$1" != "keep" ]; then
  echo "  Clearing previous run state..."
  rm -f watchdog_memory.db
  rm -f audit.log
fi

echo ""
echo "  Starting demo run..."
echo ""

python -m demo.demo_task

echo ""
echo "  Demo complete. Inspect the audit trail with:"
echo "    python timeline_ui/server.py    # then open http://localhost:8000"
echo ""