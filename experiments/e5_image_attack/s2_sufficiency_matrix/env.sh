cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
ROOT="$(cd ../../../.. && pwd)"
RUNS=$ROOT/data/experiments/e5_image_attack/s2_sufficiency_matrix/runs
REC=$ROOT/records/e5_image_attack/s2_sufficiency_matrix
STATE=$RUNS/.queue
export ROOT RUNS REC STATE
mkdir -p "$RUNS" "$REC/logs" "$STATE/locks" "$STATE/gate"
