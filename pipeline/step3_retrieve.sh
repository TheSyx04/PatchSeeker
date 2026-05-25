#!/bin/bash
# ============================================================
# pipeline/step3_retrieve.sh
# ============================================================
# Bước 3 — Chạy trên KAGGLE (1 session cuối, sau khi có đủ shards)
#
# Cách dùng trong Kaggle notebook (%%bash cell):
#
#   export PROJECT=openssl      # hoặc linux
#   export TOTAL_SHARDS=2       # phải khớp với số shards đã encode
#   bash /kaggle/working/PatchSeeker/pipeline/step3_retrieve.sh
#
# Input (Kaggle Dataset):
#   - "patchseeker-<project>-data":  queries.json (để lấy thông tin)
#   - "patchseeker-<project>-emb":   corpus_00.pkl, corpus_01.pkl, ..., queries.pkl
#
# Output:
#   /kaggle/working/output/ranking.txt
# ============================================================

set -e
set -o pipefail

# ── Config ────────────────────────────────────────────────────
PROJECT="${PROJECT:-openssl}"
TOTAL_SHARDS="${TOTAL_SHARDS:-2}"
DEPTH="${DEPTH:-10}"     # top-K commits per CVE

# Paths
REPO_DIR="/kaggle/working/PatchSeeker"
EMB_DIR="/kaggle/input/patchseeker-${PROJECT}-emb"    # Kaggle Dataset với embeddings
OUTPUT_DIR="/kaggle/working/output"
EVAL_SRC="${REPO_DIR}/eval/tevatron/src"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "STEP 3: FAISS Retrieval"
echo "  Project:      ${PROJECT}"
echo "  Total shards: ${TOTAL_SHARDS}"
echo "  Depth (top-K): ${DEPTH}"
echo "  Embeddings:   ${EMB_DIR}"
echo "============================================================"

# ── [0] Cài dependencies ─────────────────────────────────────
echo ""
echo "[0] Cài dependencies..."
pip install -q -r "${REPO_DIR}/pipeline_test/requirements.txt"

if ! python3 -c "import tevatron" 2>/dev/null; then
    pip install -q -e "${EVAL_SRC}"
fi
export PYTHONPATH="${EVAL_SRC}:${PYTHONPATH}"

# ── [1] Kiểm tra embedding files ─────────────────────────────
echo ""
echo "[1] Kiểm tra embedding files..."

QUERIES_PKL="${EMB_DIR}/queries_emb/queries.pkl"
if [ ! -f "${QUERIES_PKL}" ]; then
    echo "❌ Thiếu queries.pkl: ${QUERIES_PKL}"
    echo "   → Chạy step2 với ENCODE_QUERIES=true trước"
    exit 1
fi

# Kiểm tra tất cả corpus shards
MISSING=0
for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
    SHARD_FILE="${EMB_DIR}/corpus_emb/corpus_$(printf '%02d' $i).pkl"
    if [ ! -f "${SHARD_FILE}" ]; then
        echo "  ❌ Missing: ${SHARD_FILE}"
        MISSING=$((MISSING + 1))
    else
        SIZE=$(du -sh "${SHARD_FILE}" | cut -f1)
        echo "  ✅ corpus_$(printf '%02d' $i).pkl (${SIZE})"
    fi
done

if [ "${MISSING}" -gt 0 ]; then
    echo ""
    echo "❌ ${MISSING} shard(s) bị thiếu. Cần chạy đủ ${TOTAL_SHARDS} sessions."
    exit 1
fi

echo "  ✅ queries.pkl"
echo "  Tất cả ${TOTAL_SHARDS} shards OK"

# ── [2] FAISS retrieval ───────────────────────────────────────
echo ""
echo "[2] FAISS retrieval (merge ${TOTAL_SHARDS} shards + search)..."
T_START=$(date +%s)

python3 -m tevatron.faiss_retriever \
    --query_reps "${QUERIES_PKL}" \
    --passage_reps "${EMB_DIR}/corpus_emb/corpus_*.pkl" \
    --depth "${DEPTH}" \
    --batch_size 64 \
    --save_text \
    --save_ranking_to "${OUTPUT_DIR}/ranking.txt"

T_RANK=$(($(date +%s) - T_START))
echo "✅ FAISS retrieval xong: ${T_RANK}s"

# ── [3] Quick stats ───────────────────────────────────────────
echo ""
echo "[3] Quick stats..."
N_LINES=$(wc -l < "${OUTPUT_DIR}/ranking.txt")
N_QUERIES=$(python3 -c "
lines = set()
with open('${OUTPUT_DIR}/ranking.txt') as f:
    for line in f:
        lines.add(line.split()[0])
print(len(lines))
")
echo "  Ranking file: ${N_LINES} lines, ${N_QUERIES} unique CVEs"
echo "  Sample (first 10 lines):"
head -10 "${OUTPUT_DIR}/ranking.txt"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "HOÀN THÀNH STEP 3!"
echo "  Output: ${OUTPUT_DIR}/ranking.txt"
echo ""
echo "TIẾP THEO (chạy LOCAL):"
echo "  python pipeline/step4_evaluate.py --project ${PROJECT} \\"
echo "      --ranking-file /path/to/ranking.txt"
echo "============================================================"
