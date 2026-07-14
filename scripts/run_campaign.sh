#!/usr/bin/env bash
# Parallel campaign launcher, balanced for wall-clock. The on-policy family (PPO)
# is fast, so it carries the full ablation + sensitivity + cross-domain; the
# slow off-policy families (SAC, TD3) run fewer seeds at a reduced budget (they
# exhibit the exploit robustly regardless, and off-policy is sample-efficient).
# Runs analysis + figures + paper once all workers finish.
#
#   bash scripts/run_campaign.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[ -f .venv/bin/activate ] && source .venv/bin/activate

PPO_SEEDS="${1:-5}";  PPO_TS="${2:-120000}"
OFF_SEEDS="${3:-3}";  SAC_TS="${4:-60000}";  TD3_TS="${5:-70000}"
SENS_TS="${6:-70000}"; XSEEDS="${7:-3}"
LOG=scratchpad/campaign_logs
mkdir -p "$LOG"
bench() { python -m experiments.benchmark_algorithms "$@"; }

echo "Campaign: PPO ${PPO_SEEDS}x${PPO_TS}; SAC/TD3 ${OFF_SEEDS} seeds @ ${SAC_TS}/${TD3_TS}."

# W1: SAC (slowest per step) -- core exploit vs mitigation
bench --algos SAC --rewards hackable mitigated --seeds "$OFF_SEEDS" --timesteps "$SAC_TS" \
    > "$LOG/w1_sac.log" 2>&1 &
W1=$!
# W2: TD3 core + cross-domain transfer
{ bench --algos TD3 --rewards hackable mitigated --seeds "$OFF_SEEDS" --timesteps "$TD3_TS"
  python -m experiments.cross_domain --seeds "$XSEEDS" --timesteps 100000
} > "$LOG/w2_td3_xdomain.log" 2>&1 &
W2=$!
# W3: PPO core (all three families' headline) + sensitivity study
{ bench --algos PPO --rewards hackable mitigated --seeds "$PPO_SEEDS" --timesteps "$PPO_TS"
  python -m experiments.sensitivity_study --timesteps "$SENS_TS" --seeds 1 --init-dist 100
} > "$LOG/w3_ppo_sens.log" 2>&1 &
W3=$!
# W4: PPO mitigation ablation (whole taxonomy)
bench --algos PPO --rewards quota_hackable potential constraint_gated hierarchical \
      --seeds "$PPO_SEEDS" --timesteps "$PPO_TS" > "$LOG/w4_ppo_ablation.log" 2>&1 &
W4=$!

echo "Workers: W1(SAC)=$W1 W2(TD3+xdomain)=$W2 W3(PPO+sens)=$W3 W4(PPO-ablation)=$W4"
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
