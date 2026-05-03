#!/usr/bin/env bash
# Sequential lpl_v8 parametric study — asymmetric i0_lpl/i0_lst, checkup every 50 cycles
# Run from batFEM root:
#   conda run -n fenics-env bash run_lpl_v8.sh
# or:
#   bash run_lpl_v8.sh   (if fenics-env is already active)

set -euo pipefail
LOGDIR="results/lpl_v8_logs"
mkdir -p "$LOGDIR"

CELL="json_battery/cells/cell_Ecker2015.json"
OPTS="json_options/p2d_sei_lpl_robust.json"

run_case() {
    local cname=$1 tname=$2
    local plan="json_testplan/Cycles_500_${cname}_${tname}_CU50.json"
    local out="lpl_v8_${cname}_${tname}"
    local log="$LOGDIR/${out}.log"

    echo "======================================================"
    echo "  Starting: $out"
    echo "  $(date)"
    echo "======================================================"

    # Check for existing checkpoint (resume if blown-up previously)
    local ckpt_dir="results/${out}/checkpoints"
    local restart_flag=""
    if [ -d "$ckpt_dir" ]; then
        local last_ckpt
        last_ckpt=$(ls "$ckpt_dir"/*.npz 2>/dev/null | sort | tail -1 || true)
        if [ -n "$last_ckpt" ]; then
            echo "  Resuming from checkpoint: $last_ckpt"
            restart_flag="-restart $last_ckpt"
        fi
    fi

    python3 test_examples/test_sim.py \
        "$CELL" "$plan" "$OPTS" \
        -output "$out" \
        $restart_flag \
        2>&1 | tee "$log"

    echo "  Finished: $out  ($(date))"
}

run_case 05C 10deg
run_case 05C 25deg
run_case 05C 40deg
run_case 10C 10deg
run_case 10C 25deg
run_case 10C 40deg
run_case 20C 10deg
run_case 20C 25deg
run_case 20C 40deg

echo ""
echo "All lpl_v8 cases complete."
