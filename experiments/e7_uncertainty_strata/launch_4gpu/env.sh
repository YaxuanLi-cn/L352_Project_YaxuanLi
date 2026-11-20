ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT" || exit 1
source /venv/rafel/bin/activate
E4=code/experiments/e4_self_overrating/rerun
E7=code/experiments/e7_uncertainty_strata
RERUN=data/experiments/e4_self_overrating/rerun
SEEDS=data/experiments/e7_uncertainty_strata/ckpt_seeds
PILOT=records/e7_uncertainty_strata/pilot
LOGS=records/e7_uncertainty_strata/logs
export ROOT E4 E7 RERUN SEEDS PILOT LOGS
