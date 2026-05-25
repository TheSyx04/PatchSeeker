#!/bin/bash
# ============================================================
# pipeline/step2_encode.sh
# ============================================================
# Bước 2 — Chạy trên KAGGLE T4 (mỗi session encode 1 shard)
#
# Cách dùng trong Kaggle notebook (%%bash cell):
#
#   export PROJECT=openssl      # hoặc linux
#   export SHARD_INDEX=0        # shard này: 0, 1, ..., (TOTAL_SHARDS-1)
#   export TOTAL_SHARDS=2       # openssl=2, linux=8
#   export ENCODE_QUERIES=true  # chỉ set true cho SHARD_INDEX=0
#   bash /kaggle/working/PatchSeeker/pipeline/step2_encode.sh 2>&1
#
# Ghi chú: luôn thêm "2>&1" cuối lệnh bash để xem đủ output/error
# ============================================================

# KHÔNG dùng "set -e" để tránh ẩn thông báo lỗi

# ── Config từ env vars ────────────────────────────────────────
PROJECT="${PROJECT:-openssl}"
SHARD_INDEX="${SHARD_INDEX:-0}"
TOTAL_SHARDS="${TOTAL_SHARDS:-2}"
ENCODE_QUERIES="${ENCODE_QUERIES:-false}"

# Paths
REPO_DIR="/kaggle/working/PatchSeeker"
OUTPUT_DIR="/kaggle/working/output"
EVAL_SRC="${REPO_DIR}/eval/tevatron/src"
ENCODE_SCRIPT="${EVAL_SRC}/repllama/encode_qwen.py"

HF_REPO="Ngoc9392/Patchseeker_Checkpoint"
MODEL_FOLDER="qwen3_8b"
BASE_MODEL="Qwen/Qwen3-8B"
CKPT_DIR="${REPO_DIR}/checkpoints/${MODEL_FOLDER}"

SHARD_PAD=$(printf "%02d" "${SHARD_INDEX}")

# ── Auto-detect DATA_DIR ──────────────────────────────────────
# Tìm corpus.json trong /kaggle/input/ (bất kể user đặt tên dataset gì)
if [ -n "${DATA_DIR}" ] && [ -f "${DATA_DIR}/corpus.json" ]; then
    echo "  DATA_DIR từ env: ${DATA_DIR}"
else
    CORPUS_FILE=$(find /kaggle/input -name "corpus.json" 2>/dev/null | head -1)
    if [ -z "${CORPUS_FILE}" ]; then
        echo "❌ Không tìm thấy corpus.json trong /kaggle/input/"
        echo ""
        echo "Các dataset đang mount:"
        ls /kaggle/input/ 2>/dev/null || echo "  (không có dataset nào)"
        echo ""
        echo "Giải pháp:"
        echo "  1. Đảm bảo đã add dataset 'patchseeker-openssl-data' vào notebook"
        echo "  2. Hoặc set thủ công: export DATA_DIR=/kaggle/input/<tên-dataset>"
        exit 1
    fi
    DATA_DIR=$(dirname "${CORPUS_FILE}")
    echo "  ✅ Auto-detect DATA_DIR: ${DATA_DIR}"
fi

mkdir -p "${OUTPUT_DIR}/corpus_emb"
mkdir -p "${OUTPUT_DIR}/queries_emb"
mkdir -p "${CKPT_DIR}"

echo "============================================================"
echo "STEP 2: Encode corpus shard ${SHARD_INDEX}/${TOTAL_SHARDS}"
echo "  Project:      ${PROJECT}"
echo "  Shard:        ${SHARD_INDEX} / ${TOTAL_SHARDS}"
echo "  Data:         ${DATA_DIR}"
echo "  Output:       ${OUTPUT_DIR}/corpus_emb/corpus_${SHARD_PAD}.pkl"
echo "  REPO_DIR:     ${REPO_DIR}"
echo "============================================================"

# Kiểm tra repo
if [ ! -f "${ENCODE_SCRIPT}" ]; then
    echo "❌ Không tìm thấy encode script: ${ENCODE_SCRIPT}"
    echo "  → Chạy: git clone https://github.com/<user>/PatchSeeker /kaggle/working/PatchSeeker"
    exit 1
fi

# ── [0] Cài dependencies ─────────────────────────────────────
echo ""
echo "[0] Cài dependencies..."
pip install -q -r "${REPO_DIR}/pipeline_test/requirements.txt"
if [ $? -ne 0 ]; then
    echo "❌ pip install thất bại — thử cài từng package:"
    pip install -q torch transformers accelerate peft bitsandbytes faiss-gpu-cu12 datasets huggingface-hub tqdm
fi

# Cài tevatron từ source
if ! python3 -c "import tevatron" 2>/dev/null; then
    echo "  → Cài tevatron từ source..."
    pip install -q -e "${EVAL_SRC}"
fi
export PYTHONPATH="${EVAL_SRC}:${PYTHONPATH}"

python3 - <<PYEOF
import os
from huggingface_hub import hf_hub_download

REPO         = "${HF_REPO}"
MODEL_FOLDER = "${MODEL_FOLDER}"
CKPT_DIR     = "${CKPT_DIR}"
TOKEN        = os.environ.get("HF_TOKEN", None)

INFERENCE_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]

downloaded, skipped = 0, 0
for fname in INFERENCE_FILES:
    local_path = os.path.join(CKPT_DIR, fname)
    if os.path.exists(local_path):
        skipped += 1
        continue
    try:
        remote_path = f"{MODEL_FOLDER}/{fname}"
        hf_hub_download(
            repo_id=REPO, filename=remote_path,
            local_dir=CKPT_DIR, local_dir_use_symlinks=False, token=TOKEN,
        )
        src = os.path.join(CKPT_DIR, MODEL_FOLDER, fname)
        if os.path.exists(src):
            import shutil
            os.makedirs(CKPT_DIR, exist_ok=True)
            os.replace(src, local_path)
        downloaded += 1
        print(f"  [ok]  {fname}")
    except Exception as e:
        print(f"  [skip] {fname}: {e}")
        skipped += 1

print(f"Downloaded {downloaded}, skipped {skipped}")
PYEOF

echo "  ✅ Checkpoint ready: ${CKPT_DIR}"

# ── [2] Encode corpus shard ───────────────────────────────────
echo ""
echo "[2] Encode corpus shard ${SHARD_INDEX}/${TOTAL_SHARDS}..."
T_START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python3 "${ENCODE_SCRIPT}" \
    --output_dir=temp \
    --model_name_or_path "${CKPT_DIR}" \
    --tokenizer_name "${BASE_MODEL}" \
    --fp16 \
    --dataset_proc_num 4 \
    --per_device_eval_batch_size 16 \
    --p_max_len 512 \
    --dataset_name json \
    --train_dir "${DATA_DIR}/corpus.json" \
    --raw_file_path "${DATA_DIR}/corpus.json" \
    --encoded_save_path "${OUTPUT_DIR}/corpus_emb/corpus_${SHARD_PAD}.pkl" \
    --encode_num_shard "${TOTAL_SHARDS}" \
    --encode_shard_index "${SHARD_INDEX}"

T_CORPUS=$(($(date +%s) - T_START))
echo "✅ Encode corpus shard ${SHARD_INDEX} xong: ${T_CORPUS}s ($(echo "scale=1; $T_CORPUS/60" | bc) phút)"

# ── [3] Encode queries (chỉ chạy 1 lần, không cần shard) ─────
if [ "${ENCODE_QUERIES}" = "true" ]; then
    echo ""
    echo "[3] Encode queries (CVE descriptions)..."
    T_START=$(date +%s)

    CUDA_VISIBLE_DEVICES=0 python3 "${ENCODE_SCRIPT}" \
        --output_dir=temp \
        --model_name_or_path "${CKPT_DIR}" \
        --tokenizer_name "${BASE_MODEL}" \
        --fp16 \
        --dataset_proc_num 4 \
        --per_device_eval_batch_size 16 \
        --q_max_len 512 \
        --dataset_name json \
        --train_dir "${DATA_DIR}/queries.json" \
        --raw_file_path "${DATA_DIR}/queries.json" \
        --encoded_save_path "${OUTPUT_DIR}/queries_emb/queries.pkl" \
        --encode_is_qry

    T_QUERIES=$(($(date +%s) - T_START))
    echo "✅ Encode queries xong: ${T_QUERIES}s"
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "OUTPUT FILES:"
ls -lh "${OUTPUT_DIR}/corpus_emb/"
[ -f "${OUTPUT_DIR}/queries_emb/queries.pkl" ] && ls -lh "${OUTPUT_DIR}/queries_emb/"
echo ""
echo "TIẾP THEO:"
echo "  1. Download output files (corpus_${SHARD_PAD}.pkl)"
echo "  2. Upload lên Kaggle Dataset 'patchseeker-${PROJECT}-emb'"
echo "  3. Sau khi ĐỦ ${TOTAL_SHARDS} shards → chạy step3_retrieve.sh"
echo "============================================================"
