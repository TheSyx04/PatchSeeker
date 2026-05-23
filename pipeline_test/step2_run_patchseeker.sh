#!/bin/bash
# ============================================================
# BƯỚC 2: Chạy PatchSeeker encode + rank (TEST với 5 CVEs)
# ============================================================
# Tự động pull checkpoint từ HuggingFace Hub.
# Không cần truyền path thủ công.
#
# Usage (từ thư mục PatchSeeker/):
#   bash pipeline_test/step2_run_patchseeker.sh
#
# Env vars (optional override):
#   HF_REPO   — HuggingFace model repo  (default: Ngoc9392/Patchseeker_Checkpoint)
#   DATA_DIR  — thư mục data từ step 1  (default: ./pipeline_test/data)
#   HF_TOKEN  — HF token nếu repo private
# ============================================================

set -e   # Dừng nếu có lỗi
set -o pipefail

# ── Config ───────────────────────────────────────────────────
HF_REPO="${HF_REPO:-Ngoc9392/Patchseeker_Checkpoint}"
DATA_DIR="${DATA_DIR:-./pipeline_test/data}"
OUTPUT_DIR="./pipeline_test/output"
TOKENIZER="meta-llama/Llama-2-7b-hf"
EVAL_SRC="./eval/tevatron/src"

mkdir -p "${OUTPUT_DIR}/corpus_emb"
mkdir -p "${OUTPUT_DIR}/queries_emb"

echo "============================================================"
echo "BƯỚC 2: PatchSeeker Inference (TEST 5 CVEs)"
echo "  HF Repo:  ${HF_REPO}"
echo "  Data dir: ${DATA_DIR}"
echo "============================================================"

# ── Validate data từ step 1 ──────────────────────────────────
echo ""
echo "[0/4] Kiểm tra data từ Step 1..."
for f in corpus.json queries.json; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        echo "❌ Thiếu file: ${DATA_DIR}/${f}"
        echo "   → Hãy chạy step 1 trước: python pipeline_test/step1_prepare_test_data.py"
        exit 1
    fi
done
N_CORPUS=$(python -c "import json; d=json.load(open('${DATA_DIR}/corpus.json')); print(len(d))")
N_QUERIES=$(python -c "import json; d=json.load(open('${DATA_DIR}/queries.json')); print(len(d))")
echo "  ✅ corpus.json:  ${N_CORPUS} commits"
echo "  ✅ queries.json: ${N_QUERIES} CVEs"

# ── [0] Pull checkpoint từ HuggingFace Hub ───────────────────
echo ""
echo "[1/4] Pull checkpoint từ HuggingFace Hub..."
echo "  Repo: ${HF_REPO}"
T_START=$(date +%s)

CKPT=$(python - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download

repo  = os.environ.get("HF_REPO", "Ngoc9392/Patchseeker_Checkpoint")
token = os.environ.get("HF_TOKEN", None)

try:
    path = snapshot_download(
        repo_id   = repo,
        repo_type = "model",
        token     = token,
        local_dir = f"./checkpoints/{repo.split('/')[-1]}",
    )
    print(path, end="")
except Exception as e:
    print(f"❌ Không thể download checkpoint: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
)

T_DL=$(($(date +%s) - T_START))
echo "  ✅ Checkpoint tại: ${CKPT}  (${T_DL}s)"

# ── [1] Encode corpus (commits) ──────────────────────────────
echo ""
echo "[2/4] Encode corpus (commits)..."
T_START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python ${EVAL_SRC}/repllama/encode_cve.py \
    --output_dir=temp \
    --model_name_or_path "${CKPT}" \
    --tokenizer_name "${TOKENIZER}" \
    --fp16 \
    --dataset_proc_num 4 \
    --per_device_eval_batch_size 32 \
    --p_max_len 512 \
    --dataset_name json \
    --train_dir "${DATA_DIR}/corpus.json" \
    --raw_file_path "${DATA_DIR}/corpus.json" \
    --encoded_save_path "${OUTPUT_DIR}/corpus_emb/corpus.pkl" \
    --encode_num_shard 1 \
    --encode_shard_index 0

T_CORPUS=$(($(date +%s) - T_START))
echo "✅ Encode corpus xong: ${T_CORPUS}s ($(echo "scale=1; $T_CORPUS/60" | bc) phút)"

# ── [2] Encode queries (CVE descriptions) ────────────────────
echo ""
echo "[3/4] Encode queries (CVE descriptions)..."
T_START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python ${EVAL_SRC}/repllama/encode_cve.py \
    --output_dir=temp \
    --model_name_or_path "${CKPT}" \
    --tokenizer_name "${TOKENIZER}" \
    --fp16 \
    --dataset_proc_num 4 \
    --per_device_eval_batch_size 32 \
    --q_max_len 512 \
    --dataset_name json \
    --train_dir "${DATA_DIR}/queries.json" \
    --raw_file_path "${DATA_DIR}/queries.json" \
    --encoded_save_path "${OUTPUT_DIR}/queries_emb/queries.pkl" \
    --encode_is_qry

T_QUERIES=$(($(date +%s) - T_START))
echo "✅ Encode queries xong: ${T_QUERIES}s"

# ── [3] FAISS retrieval ──────────────────────────────────────
echo ""
echo "[4/4] FAISS retrieval (ranking)..."
T_START=$(date +%s)

python -m tevatron.faiss_retriever \
    --query_reps "${OUTPUT_DIR}/queries_emb/queries.pkl" \
    --passage_reps "${OUTPUT_DIR}/corpus_emb/corpus.pkl" \
    --depth 10 \
    --batch_size 32 \
    --save_text \
    --save_ranking_to "${OUTPUT_DIR}/ranking.txt"

T_RANK=$(($(date +%s) - T_START))
echo "✅ Ranking xong: ${T_RANK}s"

# ── Tổng kết ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "BENCHMARK KẾT QUẢ (${N_QUERIES} CVEs, ${N_CORPUS} commits)"
echo "  Download checkpoint: ${T_DL}s"
echo "  Encode corpus:       ${T_CORPUS}s"
echo "  Encode queries:      ${T_QUERIES}s"
echo "  FAISS rank:          ${T_RANK}s"
T_TOTAL=$((T_CORPUS + T_QUERIES + T_RANK))
echo "  TỔNG (không tính DL): ${T_TOTAL}s ($(echo "scale=1; $T_TOTAL/60" | bc) phút)"
echo "============================================================"
echo ""
echo "Output: ${OUTPUT_DIR}/ranking.txt"
echo ""
echo "Top kết quả:"
head -20 "${OUTPUT_DIR}/ranking.txt"

# Lưu benchmark
python - <<PYEOF
import json, os
from datetime import datetime
log = {
    "timestamp":            datetime.now().isoformat(),
    "hf_repo":              "${HF_REPO}",
    "checkpoint_path":      "${CKPT}",
    "download_sec":         ${T_DL},
    "encode_corpus_sec":    ${T_CORPUS},
    "encode_queries_sec":   ${T_QUERIES},
    "faiss_rank_sec":       ${T_RANK},
    "total_inference_sec":  ${T_TOTAL},
    "n_commits":            ${N_CORPUS},
    "n_cves":               ${N_QUERIES},
}
os.makedirs("./pipeline_test/output", exist_ok=True)
with open("./pipeline_test/output/benchmark_step2.json", "w") as f:
    json.dump(log, f, indent=2)
print("Benchmark saved → ./pipeline_test/output/benchmark_step2.json")
PYEOF

echo ""
echo "Tiếp theo: chạy step3_parse_results.py"
