#!/usr/bin/env bash
# Parallel campaign launcher. Splits the independent training runs across
# single-threaded background workers (balanced so the slow off-policy families
# don't dominate wall-clock), then runs analysis + figures + paper once all
# workers finish.
#
#   bash scripts/run_campaign.sh [SEEDS] [TIMESTEPS] [SENS_TS]
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # 1 thread/process; parallel processes
# Activate the project virtualenv if present (so `python` has the deps).
[ -f .venv/bin/activate ] && source .venv/bin/activate

SEEDS="${1:-5}"
TS="${2:-120000}"
SENS_TS="${3:-80000}"
XSEEDS="${4:-3}"
LOG=scratchpad/campaign_logs
mkdir -p "$LOG"
bench() { python -m experiments.benchmark_algorithms "$@"; }

# seed lists: split 0..SEEDS-1 into a lower and upper half for load balancing
HALF=$(( (SEEDS + 1) / 2 ))
LO=$(seq 0 $((HALF-1)) | tr '\n' ' ')
HI=$(seq "$HALF" $((SEEDS-1)) | tr '\n' ' ')

echo "Campaign: $SEEDS seeds, $TS steps (sensitivity $SENS_TS). Logs in $LOG/"

# W1: SAC lower-half seeds (slowest family, split across two workers)
bench --algos SAC --rewards hackable mitigated --seed-list $LO --timesteps "$TS" \
    > "$LOG/w1_sac_lo.log" 2>&1 &
W1=$!
# W2: SAC upper-half seeds + TD3 lower-half seeds
{ bench --algos SAC --rewards hackable mitigated --seed-list $HI --timesteps "$TS"
  bench --algos TD3 --rewards hackable mitigated --seed-list $LO --timesteps "$TS"
} > "$LOG/w2_sac_td3.log" 2>&1 &
W2=$!
# W3: TD3 upper-half seeds + reward-coefficient sensitivity study
{ bench --algos TD3 --rewards hackable mitigated --seed-list $HI --timesteps "$TS"
  python -m experiments.sensitivity_study --timesteps "$SENS_TS" --seeds 1 --init-dist 100
} > "$LOG/w3_td3_sens.log" 2>&1 &
W3=$!
# W4: PPO core + full mitigation ablation + cross-domain (fast family)
{ bench --algos PPO --rewards hackable mitigated --seeds "$SEEDS" --timesteps "$TS"
  bench --algos PPO --rewards quota_hackable potential constraint_gated hierarchical \
        --seeds "$SEEDS" --timesteps "$TS"
  python -m experiments.cross_domain --seeds "$XSEEDS" --timesteps "$TS"
} > "$LOG/w4_ppo_xdomain.log" 2>&1 &
W4=$!

echo "Workers: W1(SAC-lo)=$W1 W2(SAC-hi+TD3-lo)=$W2 W3(TD3-hi+sens)=$W3 W4(PPO+xdomain)=$W4"
for w in $W1 $W2 $W3 $W4; do wait "$w"; done
echo "All workers finished."

echo "=== Analysis + figures + paper ==="
python -m experiments.statistical_analysis
python -m experiments.visualize
python -m experiments.generate_paper
( cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
   && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
   && echo "paper/main.pdf built" ) || echo "pdflatex step skipped/failed"
echo "=== CAMPAIGN COMPLETE ==="
