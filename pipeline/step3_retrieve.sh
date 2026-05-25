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
#   bash /kaggle/working/PatchSeeker/pipeline/step3_retrieve.sh 2>&1
#
# KHÔNG dùng "set -e" để tránh ẩn thông báo lỗi
# ============================================================

# ── Config ────────────────────────────────────────────────────
PROJECT="${PROJECT:-openssl}"
TOTAL_SHARDS="${TOTAL_SHARDS:-2}"
DEPTH="${DEPTH:-10}"

REPO_DIR="/kaggle/working/PatchSeeker"
OUTPUT_DIR="/kaggle/working/output"
EVAL_SRC="${REPO_DIR}/eval/tevatron/src"

mkdir -p "${OUTPUT_DIR}"

# ── Auto-detect embedding directory ──────────────────────────
# Tìm queries.pkl ở bất kỳ đâu trong /kaggle/input
echo "============================================================"
echo "STEP 3: FAISS Retrieval"
echo "  Project:       ${PROJECT}"
echo "  Total shards:  ${TOTAL_SHARDS}"
echo "  Depth (top-K): ${DEPTH}"
echo "============================================================"

echo ""
echo "[auto-detect] Tìm embedding files trong /kaggle/input..."
echo ""
echo "Tất cả .pkl files:"
find /kaggle/input -name "*.pkl" 2>/dev/null | sort
echo ""

# Tìm queries.pkl
QUERIES_PKL=$(find /kaggle/input -name "queries.pkl" 2>/dev/null | head -1)
if [ -z "${QUERIES_PKL}" ]; then
    echo "❌ Không tìm thấy queries.pkl trong /kaggle/input/"
    echo ""
    echo "Datasets đang mount:"
    ls /kaggle/input/
    echo ""
    echo "Giải pháp:"
    echo "  1. Add dataset 'patchseeker-openssl-emb' vào notebook"
    echo "  2. Đảm bảo upload đúng cấu trúc: queries_emb/queries.pkl"
    echo "  3. Hoặc set thủ công: export QUERIES_PKL=/path/to/queries.pkl"
    exit 1
fi
echo "  ✅ queries.pkl: ${QUERIES_PKL}"

# Tìm thư mục chứa corpus pkl files
CORPUS_DIR=$(find /kaggle/input -name "corpus_*.pkl" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
if [ -z "${CORPUS_DIR}" ]; then
    echo "❌ Không tìm thấy corpus_*.pkl trong /kaggle/input/"
    exit 1
fi
echo "  ✅ corpus dir: ${CORPUS_DIR}"

# Kiểm tra đủ shards
echo ""
echo "[1] Kiểm tra embedding files..."
MISSING=0
for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
    SHARD_FILE="${CORPUS_DIR}/corpus_$(printf '%02d' $i).pkl"
    if [ ! -f "${SHARD_FILE}" ]; then
        echo "  ❌ Missing: corpus_$(printf '%02d' $i).pkl (tìm trong ${CORPUS_DIR})"
        MISSING=$((MISSING + 1))
    else
        SIZE=$(du -sh "${SHARD_FILE}" | cut -f1)
        echo "  ✅ corpus_$(printf '%02d' $i).pkl (${SIZE})"
    fi
done
echo "  ✅ queries.pkl ($(du -sh ${QUERIES_PKL} | cut -f1))"

if [ "${MISSING}" -gt 0 ]; then
    echo ""
    echo "❌ ${MISSING} shard(s) bị thiếu."
    echo "   Các corpus files tìm thấy:"
    find /kaggle/input -name "corpus_*.pkl" | sort
    exit 1
fi
echo ""
echo "  Tất cả ${TOTAL_SHARDS} shards OK → bắt đầu FAISS retrieval"

# ── [0] Cài dependencies ─────────────────────────────────────
echo ""
echo "[0] Cài dependencies..."
pip install -q -r "${REPO_DIR}/pipeline_test/requirements.txt"
if ! python3 -c "import tevatron" 2>/dev/null; then
    echo "  → Cài tevatron từ source..."
    pip install -q -e "${EVAL_SRC}"
fi
export PYTHONPATH="${EVAL_SRC}:${PYTHONPATH}"

# ── [2] FAISS retrieval ───────────────────────────────────────
echo ""
echo "[2] FAISS retrieval (merge ${TOTAL_SHARDS} shards + search)..."
T_START=$(date +%s)

python3 -m tevatron.faiss_retriever \
    --query_reps "${QUERIES_PKL}" \
    --passage_reps "${CORPUS_DIR}/corpus_*.pkl" \
    --depth "${DEPTH}" \
    --batch_size 64 \
    --save_text \
    --save_ranking_to "${OUTPUT_DIR}/ranking.txt"

FAISS_EXIT=$?
T_RANK=$(($(date +%s) - T_START))

if [ ${FAISS_EXIT} -ne 0 ]; then
    echo "❌ FAISS retrieval thất bại (exit code ${FAISS_EXIT})"
    exit 1
fi
echo "✅ FAISS retrieval xong: ${T_RANK}s"

# ── [3] Quick stats ───────────────────────────────────────────
echo ""
echo "[3] Quick stats..."
N_LINES=$(wc -l < "${OUTPUT_DIR}/ranking.txt")
N_QUERIES=$(python3 -c "
lines = set()
with open('${OUTPUT_DIR}/ranking.txt') as f:
    for line in f:
        parts = line.strip().split()
        if parts: lines.add(parts[0])
print(len(lines))
")
echo "  Ranking file: ${N_LINES} lines, ${N_QUERIES} unique CVEs"
echo ""
echo "  Sample (first 15 lines):"
head -15 "${OUTPUT_DIR}/ranking.txt"

# ── Summary ───────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "HOÀN THÀNH STEP 3!"
echo "  Output: ${OUTPUT_DIR}/ranking.txt"
echo ""
echo "TIẾP THEO:"
echo "  1. Download ranking.txt từ Kaggle Output"
echo "  2. Chạy local: python pipeline/step4_evaluate.py \\"
echo "       --project ${PROJECT} --ranking-file ranking.txt"
echo "============================================================"
