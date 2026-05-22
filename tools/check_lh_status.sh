#!/bin/bash
echo "=== Lighthouses libsurvive recognized in last run ==="
grep -E 'Adding lighthouse|Got OOTX|Using LH' /root/survive_dual.txt | sort -u
echo ""
echo "=== Cached libsurvive config ==="
cat /root/.config/libsurvive/config.json | python3 -c "
import sys, json
data = sys.stdin.read()
# config.json is missing some commas/braces; try to be lenient
print(data)
"
