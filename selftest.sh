#!/bin/sh
# End-to-end check on fake data: planted effect must PASS, pure noise must FAIL.  About 15 minutes.
set -e
T=$(mktemp -d)
for kind in planted noise; do
  python simulate.py $kind $T/rec_$kind 54
  RECORDER_DATA=$T/rec_$kind MICRO_DATA=$T/out_$kind python extract.py > /dev/null
  cp study.py PREREGISTRATION.md $T/
  # fake data starts 2026-07-01; put the lock 12 days in, so 42 days (1,008 hours) are confirmation hours
  (cd $T && MICRO_DATA=$T/out_$kind MICRO_RUNS=$T/runs_$kind MICRO_LOCK="2026-07-13 00:00" python study.py freeze > /dev/null &&
            MICRO_DATA=$T/out_$kind MICRO_RUNS=$T/runs_$kind MICRO_LOCK="2026-07-13 00:00" python study.py confirm) | sed "s/^/[$kind] /"
done
