#!/bin/bash
#export NCCL_IB_TC=136
#export NCCL_IB_SL=5
#export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=WARN
export HF_HOME=/home/didimai/hjpark/llm_study/.hf_cache

# ---- Environment setup -----------------------------------------------------
REPO_ROOT="/home/didimai/hjpark/llm_study"
VENV_PATH="${REPO_ROOT}/venv_sft"

if [ -f "${VENV_PATH}/bin/python" ]; then
    PY="${VENV_PATH}/bin/python"
    export PATH="${VENV_PATH}/bin:${PATH}"
    echo "Using virtual environment: ${VENV_PATH}"
else
    echo "CRITICAL: Virtual environment not found at ${VENV_PATH}. Run setup first." >&2
    exit 1
fi

# ---- paths ------------------------------------------------------------------
REPO_ROOT="/home/didimai/hjpark/llm_study"
SFT_ROOT="${REPO_ROOT}/Qwen3-Coder/finetuning/sft"

RAW_DATA=${1:-"${REPO_ROOT}/data/domainknowledge_dataset.jsonl"}
PRETRAINED_MODEL=${2:-"Qwen/Qwen2.5-Coder-1.5B"}
OUTPUT_DIR=${3:-"${REPO_ROOT}/checkpoints/qwen25_coder_1_5b_domainknowledge"}

PROCESSED_DIR="${REPO_ROOT}/data/processed"
PROCESSED_BASE="${PROCESSED_DIR}/domainknowledge_dataset"
PROCESSED_DATA="${PROCESSED_BASE}.npy"   # binarize_data.py appends .npy

MAX_LENGTH=512

# ---- 1) binarize (chatml -> tokenized .npy) ---------------------------------
mkdir -p "${PROCESSED_DIR}"
if [ ! -f "${PROCESSED_DATA}" ]; then
    echo "Binarizing ${RAW_DATA} -> ${PROCESSED_DATA}"
    cd "${SFT_ROOT}"
    $PY binarize_data.py \
        --input_path "${RAW_DATA}" \
        --output_path "${PROCESSED_BASE}" \
        --tokenizer_path "${PRETRAINED_MODEL}" \
        --max_len ${MAX_LENGTH} \
        --workers 4 \
        --save_format ".npy"
else
    echo "Skip binarize, already exists: ${PROCESSED_DATA}"
fi

# ---- 2) distributed setup ---------------------------------------------------
# Robust GPU detection
GPUS_PER_NODE=$($PY -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
if [ -z "${GPUS_PER_NODE}" ]; then
    GPUS_PER_NODE=0
fi

if [ "${GPUS_PER_NODE}" -eq 0 ]; then
    echo "WARNING: No GPU detected. Running without DeepSpeed on CPU."
    USE_DEEPSPEED=0
    GPUS_PER_NODE=1
else
    USE_DEEPSPEED=1
fi

MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
MASTER_PORT=${MASTER_PORT:-6105}
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
DEEPSPEED_CONFIG="${SFT_ROOT}/configs/default_offload_opt_param.json"
PEFT_CONFIG_FOLDER="${SFT_ROOT}/configs/lora"

# small dataset -> smaller global batch / warmup
BATCH_SIZE=32
MICRO_BATCH_SIZE=4
GRAD_ACCU=$(($BATCH_SIZE / $WORLD_SIZE / $MICRO_BATCH_SIZE))
if [ "${GRAD_ACCU}" -lt 1 ]; then GRAD_ACCU=1; fi

LR=2e-4
MIN_LR=5e-6
WARMUP_STEPS=10
WEIGHT_DECAY=0.0

echo "OUTPUT_DIR ${OUTPUT_DIR}"
echo "Pretrained Model ${PRETRAINED_MODEL}"
echo "WORLD_SIZE $WORLD_SIZE MICRO_BATCH_SIZE $MICRO_BATCH_SIZE GRAD_ACCU $GRAD_ACCU"
echo "${DISTRIBUTED_ARGS}"

# ---- 3) train ---------------------------------------------------------------
cd "${SFT_ROOT}"

if [ "${USE_DEEPSPEED}" -eq 1 ]; then
    torchrun ${DISTRIBUTED_ARGS} train.py \
        --model_name_or_path  ${PRETRAINED_MODEL} \
        --data_path ${PROCESSED_DATA} \
        --model_max_length ${MAX_LENGTH} \
        --output_dir ${OUTPUT_DIR} \
        --num_train_epochs 1 \
        --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCU} \
        --per_device_eval_batch_size 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 5 \
        --learning_rate ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_steps ${WARMUP_STEPS} \
        --lr_scheduler_type "cosine" \
        --logging_strategy "steps" \
        --logging_steps 1 \
        --deepspeed ${DEEPSPEED_CONFIG} \
        --report_to "tensorboard" \
        --bf16 True \
        --tf32 True \
        --truncate_source False \
        --use_peft True \
        --peft_config_path ${PEFT_CONFIG_FOLDER}
else
    $PY train.py \
        --model_name_or_path  ${PRETRAINED_MODEL} \
        --data_path ${PROCESSED_DATA} \
        --model_max_length ${MAX_LENGTH} \
        --output_dir ${OUTPUT_DIR} \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --per_device_eval_batch_size 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 2 \
        --learning_rate ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --warmup_steps ${WARMUP_STEPS} \
        --lr_scheduler_type "cosine" \
        --logging_strategy "steps" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --truncate_source False \
        --use_peft True \
        --peft_config_path ${PEFT_CONFIG_FOLDER}
fi