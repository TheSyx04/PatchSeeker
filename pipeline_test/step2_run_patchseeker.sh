#!/bin/bash
# ============================================================
# BƯỚC 2: Chạy PatchSeeker encode + rank (TEST với 5 CVEs)
# ============================================================
# Model: qwen3_8b (LoRA adapter trên Qwen/Qwen3-8B)
# Checkpoint: Ngoc9392/Patchseeker_Checkpoint / qwen3_8b/
#
# Chỉ download inference files (~50MB adapter),
# KHÔNG download optimizer states (~15GB).
#
# Usage (từ thư mục PatchSeeker/):
#   bash pipeline_test/step2_run_patchseeker.sh
#
# Env vars (optional override):
#   DATA_DIR  — thư mục data từ step 1  (default: ./pipeline_test/data)
#   HF_TOKEN  — HF token nếu cần
# ============================================================

set -e
set -o pipefail

# ── Config ───────────────────────────────────────────────────
HF_REPO="Ngoc9392/Patchseeker_Checkpoint"
MODEL_FOLDER="qwen3_8b"                        # subfolder trong HF repo
BASE_MODEL="Qwen/Qwen3-8B"                     # base model để load weights
CKPT_DIR="./checkpoints/${MODEL_FOLDER}"       # local path lưu adapter

DATA_DIR="${DATA_DIR:-./pipeline_test/data}"
OUTPUT_DIR="./pipeline_test/output"
EVAL_SRC="./eval/tevatron/src"

mkdir -p "${OUTPUT_DIR}/corpus_emb"
mkdir -p "${OUTPUT_DIR}/queries_emb"
mkdir -p "${CKPT_DIR}"

echo "============================================================"
echo "BƯỚC 2: PatchSeeker Inference (TEST 5 CVEs)"
echo "  Model:    ${MODEL_FOLDER} (LoRA on ${BASE_MODEL})"
echo "  HF Repo:  ${HF_REPO}/${MODEL_FOLDER}"
echo "  Data dir: ${DATA_DIR}"
echo "============================================================"

# ── [0] Validate data từ step 1 ──────────────────────────────
echo ""
echo "[0/4] Kiểm tra data từ Step 1..."
for f in corpus.json queries.json; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        echo "❌ Thiếu file: ${DATA_DIR}/${f}"
        echo "   → Hãy chạy step 1 trước: python pipeline_test/step1_prepare_test_data.py"
        exit 1
    fi
done
N_CORPUS=$(python3 -c "import json; print(len(json.load(open('${DATA_DIR}/corpus.json'))))")
N_QUERIES=$(python3 -c "import json; print(len(json.load(open('${DATA_DIR}/queries.json'))))")
echo "  ✅ corpus.json:  ${N_CORPUS} commits"
echo "  ✅ queries.json: ${N_QUERIES} CVEs"

# ── [1] Download chỉ inference files (~50MB, không download optimizer) ──
echo ""
echo "[1/4] Download inference files từ HuggingFace..."
echo "  Repo:   ${HF_REPO}"
echo "  Folder: ${MODEL_FOLDER}/"
echo "  (Bỏ qua optimizer states, rng states — chỉ lấy adapter + tokenizer)"
T_START=$(date +%s)

python3 - <<PYEOF
import os, sys
from huggingface_hub import hf_hub_download

REPO         = "${HF_REPO}"
MODEL_FOLDER = "${MODEL_FOLDER}"
CKPT_DIR     = "${CKPT_DIR}"
TOKEN        = os.environ.get("HF_TOKEN", None)

# Chỉ download những file cần thiết cho inference:
# - adapter_config.json + adapter_model.safetensors (LoRA weights)
# - tokenizer files
# KHÔNG download: global_step*/, rng_state*.pth, training_args.bin, zero_to_fp32.py
INFERENCE_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",           # BPE tokenizer (Qwen/GPT style)
]

downloaded = 0
skipped    = 0
for fname in INFERENCE_FILES:
    remote_path = f"{MODEL_FOLDER}/{fname}"
    local_path  = os.path.join(CKPT_DIR, fname)

    # Skip nếu đã có
    if os.path.exists(local_path):
        print(f"  [cache] {fname}", flush=True)
        downloaded += 1
        continue

    try:
        hf_hub_download(
            repo_id    = REPO,
            filename   = remote_path,
            local_dir  = CKPT_DIR,
            local_dir_use_symlinks = False,
            token      = TOKEN,
        )
        # hf_hub_download lưu vào subdir, move ra ngoài
        src = os.path.join(CKPT_DIR, MODEL_FOLDER, fname)
        if os.path.exists(src):
            os.makedirs(CKPT_DIR, exist_ok=True)
            os.replace(src, local_path)
        print(f"  [ok]    {fname}", flush=True)
        downloaded += 1
    except Exception as e:
        # Một số file không tồn tại tùy model (vd: merges.txt với SentencePiece)
        print(f"  [skip]  {fname}  ({e})", flush=True)
        skipped += 1

# Dọn thư mục con nếu hf_hub_download tạo ra
import shutil
sub = os.path.join(CKPT_DIR, MODEL_FOLDER)
if os.path.isdir(sub) and not os.listdir(sub):
    shutil.rmtree(sub)

print(f"\n✅ Downloaded {downloaded} files, skipped {skipped}", flush=True)
print(f"✅ Checkpoint tại: {CKPT_DIR}", flush=True)
PYEOF

T_DL=$(($(date +%s) - T_START))
echo "  Download xong: ${T_DL}s"
echo "  Files trong ${CKPT_DIR}:"
ls -lh "${CKPT_DIR}/"

# ── [2] Encode corpus (commits) ──────────────────────────────
echo ""
echo "[2/4] Encode corpus (commits)..."
T_START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python3 ${EVAL_SRC}/repllama/encode_cve.py \
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
    --encoded_save_path "${OUTPUT_DIR}/corpus_emb/corpus.pkl" \
    --encode_num_shard 1 \
    --encode_shard_index 0

T_CORPUS=$(($(date +%s) - T_START))
echo "✅ Encode corpus xong: ${T_CORPUS}s ($(echo "scale=1; $T_CORPUS/60" | bc) phút)"

# ── [3] Encode queries (CVE descriptions) ────────────────────
echo ""
echo "[3/4] Encode queries (CVE descriptions)..."
T_START=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python3 ${EVAL_SRC}/repllama/encode_cve.py \
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

# ── [4] FAISS retrieval ──────────────────────────────────────
echo ""
echo "[4/4] FAISS retrieval (ranking)..."
T_START=$(date +%s)

python3 -m tevatron.faiss_retriever \
    --query_reps "${OUTPUT_DIR}/queries_emb/queries.pkl" \
    --passage_reps "${OUTPUT_DIR}/corpus_emb/corpus.pkl" \
    --depth 10 \
    --batch_size 32 \
    --save_text \
    --save_ranking_to "${OUTPUT_DIR}/ranking.txt"

T_RANK=$(($(date +%s) - T_START))
echo "✅ Ranking xong: ${T_RANK}s"

# ── Tổng kết ─────────────────────────────────────────────────
T_TOTAL=$((T_CORPUS + T_QUERIES + T_RANK))
echo ""
echo "============================================================"
echo "BENCHMARK KẾT QUẢ"
echo "  Model:           ${MODEL_FOLDER} (${BASE_MODEL})"
echo "  Corpus:          ${N_CORPUS} commits"
echo "  Queries:         ${N_QUERIES} CVEs"
echo "  Download ckpt:   ${T_DL}s"
echo "  Encode corpus:   ${T_CORPUS}s"
echo "  Encode queries:  ${T_QUERIES}s"
echo "  FAISS rank:      ${T_RANK}s"
echo "  TỔNG inference:  ${T_TOTAL}s ($(echo "scale=1; $T_TOTAL/60" | bc) phút)"
echo "============================================================"
echo ""
echo "Top kết quả:"
head -20 "${OUTPUT_DIR}/ranking.txt"

# Lưu benchmark
python3 - <<PYEOF
import json, os
from datetime import datetime
log = {
    "timestamp":           datetime.now().isoformat(),
    "model_folder":        "${MODEL_FOLDER}",
    "base_model":          "${BASE_MODEL}",
    "hf_repo":             "${HF_REPO}",
    "download_sec":        ${T_DL},
    "encode_corpus_sec":   ${T_CORPUS},
    "encode_queries_sec":  ${T_QUERIES},
    "faiss_rank_sec":      ${T_RANK},
    "total_inference_sec": ${T_TOTAL},
    "n_commits":           ${N_CORPUS},
    "n_cves":              ${N_QUERIES},
}
os.makedirs("./pipeline_test/output", exist_ok=True)
with open("./pipeline_test/output/benchmark_step2.json", "w") as f:
    json.dump(log, f, indent=2)
print("Benchmark saved → ./pipeline_test/output/benchmark_step2.json")
PYEOF

echo ""
echo "Tiếp theo: chạy step3_parse_results.py"
