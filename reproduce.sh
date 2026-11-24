#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# The scripts append to the records, so a rerun has to start from empty files.
# Move the old records and trajectories aside first, so that a failed rerun
# loses nothing.
shopt -s nullglob
old=(results/*.jsonl)
if [ "${#old[@]}" -gt 0 ] || [ -d results/trajectories ]; then
  prev="results/previous-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$prev"
  [ "${#old[@]}" -gt 0 ] && mv "${old[@]}" "$prev"/
  [ -d results/trajectories ] && mv results/trajectories "$prev"/
  echo "Moved the old records to $prev"
fi

bench() { python -m experiments.benchmark_algorithms "$@"; }

bench --algos SAC --rewards hackable mitigated --seeds 3 --timesteps 60000 &
pids=($!)
{ bench --algos TD3 --rewards hackable mitigated --seeds 3 --timesteps 70000
  python -m experiments.cross_domain --seeds 3 --timesteps 100000
} &
pids+=($!)
{ bench --algos PPO --rewards hackable mitigated --seeds 5 --timesteps 120000
  python -m experiments.sensitivity_study --seeds 1 --timesteps 70000 --init-dist 100
} &
pids+=($!)
bench --algos PPO --rewards quota_hackable potential constraint_gated hierarchical \
    --seeds 5 --timesteps 120000 &
pids+=($!)
for pid in "${pids[@]}"; do wait "$pid"; done

python -m experiments.statistical_analysis
python -m experiments.visualize
