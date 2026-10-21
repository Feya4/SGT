#!/usr/bin/env bash
# Start the control experiments in the background; they wait until a GPU has enough free memory.
# Usage:   bash start_controls.sh <prefix of the Table 7 runs> [free MB needed, default 4000]
# Example: bash start_controls.sh sgt200_bracs
# Finished runs (also earlier runs named <prefix>_morphshuf / _posshuf) are skipped, so this starts only
# what is missing. Set SGT_ROOT first (folder that contains outputs/ and data/).
# Follow:  tail -f controls_<prefix>.log        Stop: kill $(cat controls_<prefix>.pid)
set -e
REF=${1:?give the prefix of the reference runs, e.g. sgt200_bracs}
NEED=${2:-4000}
cd "$(dirname "$0")"
if [ -z "$SGT_ROOT" ]; then echo "set SGT_ROOT first, e.g. export SGT_ROOT=/path/with/outputs"; exit 1; fi
python run_controls.py --ref "$REF" --dry_run
nohup python run_controls.py --ref "$REF" --wait_mb "$NEED" > "controls_${REF}.log" 2>&1 &
echo $! > "controls_${REF}.pid"
echo "started (pid $(cat controls_${REF}.pid)); log: controls_${REF}.log"
