#!/usr/bin/env bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 1
cd "$ROOT" || exit 1

MODEL="${MODEL:-ddn}"
SCALE="${SCALE:-130M}"
NGPU="${NGPU:-8}"
GLOBAL_BATCH_SEQUENCES="${GLOBAL_BATCH_SEQUENCES:-256}"

case "$SCALE" in
    130M)
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"; STEPS="${STEPS:-9500}"
        LR="${LR:-0.0006}"; WARMUP_STEPS="${WARMUP_STEPS:-512}"; TOKEN_LABEL=5B ;;
    340M)
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"; STEPS="${STEPS:-30000}"
        LR="${LR:-0.0004}"; WARMUP_STEPS="${WARMUP_STEPS:-1024}"; TOKEN_LABEL=15B ;;
    1B)
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"; STEPS="${STEPS:-95000}"
        LR="${LR:-0.0004}"; WARMUP_STEPS="${WARMUP_STEPS:-1024}"; TOKEN_LABEL=50B ;;
    *) printf 'Unsupported scale: %s\n' "$SCALE" >&2; exit 2 ;;
esac

[ "$((GLOBAL_BATCH_SEQUENCES % (NGPU * MICRO_BATCH_SIZE)))" -eq 0 ] || {
    printf 'Global batch must be divisible by GPU count times micro batch size.\n' >&2
    exit 2
}
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SEQUENCES / NGPU / MICRO_BATCH_SIZE))
OUTPUT="${OUTPUT:-${ROOT}/exp/${MODEL}_${SCALE}/fineweb_edu_${TOKEN_LABEL}}"

exec torchrun \
    --standalone \
    --nproc_per_node="$NGPU" \
    -m flame.train \
    --job.dump_folder "$OUTPUT" \
    --job.description "${MODEL}_${SCALE}_fineweb_edu_${TOKEN_LABEL}" \
    --job.print_args \
    --model.config "${ROOT}/configs/${MODEL}_${SCALE}.json" \
    --model.tokenizer_path "${TOKENIZER:-fla-hub/transformer-1.3B-100B}" \
    --optimizer.lr "$LR" \
    --optimizer.weight_decay 0.01 \
    --lr_scheduler.decay_type cosine \
    --lr_scheduler.warmup_steps "$WARMUP_STEPS" \
    --lr_scheduler.lr_min 0.1 \
    --training.batch_size "$MICRO_BATCH_SIZE" \
    --training.seq_len 2048 \
    --training.gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --training.steps "$STEPS" \
    --training.skip_nan_inf \
    --training.data_parallel_replicate_degree "$NGPU" \
    --training.data_parallel_shard_degree 1 \
    --training.dataset "${DATASET:-HuggingFaceFW/fineweb-edu}" \
    --training.dataset_name "${DATASET_NAME:-sample-100BT}" \
    --training.dataset_split train \
    --training.streaming \
    --training.num_workers "${NUM_WORKERS:-4}" \
    --training.compile \
    --activation_checkpoint.mode none \
    --checkpoint.enable_checkpoint \
    --checkpoint.interval "${CHECKPOINT_INTERVAL:-500}" \
    --checkpoint.load_step "${RESUME_STEP:--1}" \
    --checkpoint.keep_latest_k 2 \
    --metrics.enable_tensorboard \
    "$@"
