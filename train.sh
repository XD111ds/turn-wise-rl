set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${VERL_PATH:?Set VERL_PATH}"
: "${SGLANG_PATH:?Set SGLANG_PATH}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${TRAIN_PARQUET:?Set TRAIN_PARQUET}"
: "${RL_USER_TURNS_JSON:?Set RL_USER_TURNS_JSON}"
: "${FRAME_ROOT:?Set FRAME_ROOT}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${LLM_EVALUATOR_URL:?Set LLM_EVALUATOR_URL}"
: "${LLM_EVALUATOR_MODEL:?Set LLM_EVALUATOR_MODEL}"

export MODEL_PATH TRAIN_PARQUET RL_USER_TURNS_JSON OUTPUT_DIR LLM_EVALUATOR_URL LLM_EVALUATOR_MODEL
export PYTHONPATH="$ROOT:$VERL_PATH:$SGLANG_PATH/python${PYTHONPATH:+:$PYTHONPATH}"
export TURNWISE_TOOL_CONFIG="$ROOT/configs/video_tool.yaml"
export TURNWISE_REWARD_FUNCTION="$ROOT/recipe/proactive/reward_function.py"
export MAX_PIXELS="${MAX_PIXELS:-100352}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$MAX_PIXELS}"
export FPS_MAX_FRAMES=90
export VERL_AUTO_PADDING=TRUE
export LLM_EVALUATOR_NUM_PROCESSES="${LLM_EVALUATOR_NUM_PROCESSES:-20}"
export LLM_EVALUATOR_MAX_SCORES=5
export LLM_EVALUATOR_PAUC_ADD_PREV_TURNS=0
export LLM_EVALUATOR_REPETITION_IN_ALL_SPANS=1

python - "$MODEL_PATH" "$TRAIN_PARQUET" "$RL_USER_TURNS_JSON" "$FRAME_ROOT" "$OUTPUT_DIR" <<'PY'
from pathlib import Path
import sys

for value in sys.argv[1:]:
    if '<' in value or '>' in value:
        raise SystemExit('Replace path placeholders before training')
for value in sys.argv[1:5]:
    if not Path(value).is_absolute() or not Path(value).exists():
        raise SystemExit(f'Expected an existing absolute path: {value}')
if not Path(sys.argv[5]).is_absolute():
    raise SystemExit('OUTPUT_DIR must be an absolute path')
PY

mkdir -p "$OUTPUT_DIR/runtime/data"
if [[ -e "$OUTPUT_DIR/runtime/data/datasets" || -L "$OUTPUT_DIR/runtime/data/datasets" ]]; then
    [[ "$(readlink -f "$OUTPUT_DIR/runtime/data/datasets")" == "$(readlink -f "$FRAME_ROOT")" ]] || exit 1
else
    ln -s "$FRAME_ROOT" "$OUTPUT_DIR/runtime/data/datasets"
fi
cd "$OUTPUT_DIR/runtime"
python -u -m verl.trainer.main_ppo \
    --config-path "$ROOT/configs" \
    --config-name train \
    algorithm.credit_assignment.alpha_max="${ALPHA:-1.0}" \
    "$@"
