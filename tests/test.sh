#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt

cd /tests
status=0
python3 -m pytest -x \
  --ctrf /logs/verifier/ctrf.json \
  /tests/test_outputs.py -rA || status=$?

if [ "$status" -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
    exit 0
fi

echo 0 > /logs/verifier/reward.txt
exit "$status"
