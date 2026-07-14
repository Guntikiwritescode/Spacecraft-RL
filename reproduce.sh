#!/usr/bin/env bash
# Reproduce the full Goodhart's Taxonomy campaign end to end.
#
#   bash reproduce.sh [quick|full]
#
# quick (default): a fast smoke that still produces every figure/table/PDF.
# full:            the thesis-scale campaign (5 seeds, 150k steps).
set -euo pipefail

PRESET="${1:-quick}"
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"

if [ "$PRESET" = "full" ]; then
    SEEDS=5; TS=150000; SENS_SEEDS=3; XSEEDS=5
else
    SEEDS=3; TS=60000; SENS_SEEDS=1; XSEEDS=3
fi

echo "=== [1/6] Core cross-family benchmark (PPO/SAC/TD3, hackable+mitigated) ==="
python -m experiments.benchmark_algorithms \
    --algos PPO SAC TD3 --rewards hackable mitigated \
    --seeds "$SEEDS" --timesteps "$TS"

echo "=== [2/6] PPO mitigation ablation (whole taxonomy) ==="
python -m experiments.benchmark_algorithms \
    --algos PPO \
    --rewards quota_hackable potential constraint_gated hierarchical \
    --seeds "$SEEDS" --timesteps "$TS"

echo "=== [3/6] Reward-coefficient sensitivity study ==="
python -m experiments.sensitivity_study \
    --timesteps "$TS" --seeds "$SENS_SEEDS" --init-dist 150

echo "=== [4/6] Cross-domain transfer (autonomous lane-keeping) ==="
python -m experiments.cross_domain --seeds "$XSEEDS" --timesteps "$TS"

echo "=== [5/6] Statistical analysis + figures ==="
python -m experiments.statistical_analysis
python -m experiments.visualize

echo "=== [6/6] Paper: inject numbers and compile PDF ==="
python -m experiments.generate_paper
( cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
    && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
    && echo "paper/main.pdf built" ) || echo "pdflatex not available; paper source is in paper/"

echo "=== DONE. See figures/, results/, and paper/main.pdf ==="
