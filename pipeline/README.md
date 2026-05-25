# PatchSeeker — Full Production Pipeline

> Chạy PatchSeeker trên toàn bộ dataset Linux (1,502 CVEs) và OpenSSL (295 CVEs).  
> Theo đúng settings của paper: 1-year window, Qwen3-8B backbone, 4-bit NF4 quantization.

---

## Tổng quan thời gian

| Project | CVEs | Unique commits | Sessions Kaggle | Thời gian thực |
|---|---|---|---|---|
| **OpenSSL** | 295 | ~65K | **2** (song song) | **~8h** |
| **Linux** | 1,502 | ~390K | **8** (song song) | **~9h** |

> **Kaggle T4**: 9h/session, 20GB disk, 14.5GB VRAM  
> Các sessions encode có thể chạy **song song** trên nhiều tài khoản → thời gian thực ≈ 1 session

---

## Cài đặt ban đầu

### Yêu cầu (máy LOCAL)

```bash
# Python 3.10+, git

pip install requests tqdm
```

### Clone PatchSeeker

```bash
git clone https://github.com/Ngoc9392/PatchSeeker
cd PatchSeeker
```

---

## BƯỚC 1 — Chuẩn bị data (LOCAL)

> Chạy 1 lần trên máy tính của bạn. **Không cần GPU.**

### OpenSSL

```bash
# Bước này tự động:
#  1. Clone openssl repo (blobless, ~50MB)
#  2. Fetch CVE dates từ NVD API (~30 phút cho 295 CVEs)
#  3. Collect commits trong 1-year window cho mỗi CVE
#  4. Dedup → corpus.json (~65K commits)
#  5. Output queries.json, qrels.txt

python pipeline/step1_prepare_data.py --project openssl
```

**Thời gian**: ~60 phút (chủ yếu NVD API rate limit: 6.5s/request × 295 CVEs)

> **Tùy chọn**: Nếu có NVD API key (free tại https://nvd.nist.gov/developers/request-an-api-key):
> ```bash
> python pipeline/step1_prepare_data.py --project openssl --nvd-api-key YOUR_KEY
> ```
> API key tăng rate limit 10x → ~3 phút thay vì 30 phút.

### Linux

```bash
# Cảnh báo: Linux kernel blobless clone = ~1-2GB, git log = ~2 giờ

python pipeline/step1_prepare_data.py --project linux
```

**Thời gian**: ~3-4 giờ (NVD fetch + git log cho 390K commits)

### Output

```
pipeline/data/<project>/
├── corpus.json          ← Tất cả unique commits (JSONL)
├── queries.json         ← CVE descriptions (JSONL)  
├── qrels.txt            ← Ground truth VFCs (TREC format, có thể rỗng)
├── cve_candidates.json  ← Map CVE → [commit_ids]
├── stats.json           ← Thống kê
└── nvd_dates_cache.json ← Cache CVE dates (dùng lại khi chạy lại)
```

---

## BƯỚC 2 — Upload data lên Kaggle

1. Tạo **Kaggle Dataset** mới: `patchseeker-openssl-data` (hoặc `patchseeker-linux-data`)
2. Upload toàn bộ thư mục `pipeline/data/openssl/` (hoặc `linux/`)
3. Dataset này sẽ được dùng làm **input** cho tất cả Kaggle sessions

> **Dung lượng**:
> - OpenSSL corpus.json: ~50MB
> - Linux corpus.json: ~300-500MB

---

## BƯỚC 3 — Encode corpus (KAGGLE, song song)

> Mỗi Kaggle session encode **1 shard** của corpus.  
> Chạy tất cả sessions **cùng lúc** trên nhiều tài khoản.

### Setup Kaggle Notebook

1. Tạo notebook mới → chọn **GPU T4 x1**
2. Add dataset: `patchseeker-<project>-data` (từ Bước 2)
3. Add dataset: `PatchSeeker` repo (hoặc clone trong notebook)

### Cell 1 — Setup

```python
%%bash
cd /kaggle/working
git clone https://github.com/Ngoc9392/PatchSeeker
echo "✅ Cloned"
```

### Cell 2 — Encode (thay SHARD_INDEX cho mỗi session)

```bash
# ╔══════════════════════════════════════╗
# ║  THAY ĐỔI: SHARD_INDEX = 0, 1, ...  ║
# ╚══════════════════════════════════════╝

export PROJECT=openssl       # hoặc linux
export SHARD_INDEX=0         # 👈 THAY ĐỔI: 0, 1 (openssl) / 0..7 (linux)
export TOTAL_SHARDS=2        # openssl=2, linux=8
export ENCODE_QUERIES=true   # CHỈ true cho SHARD_INDEX=0

bash /kaggle/working/PatchSeeker/pipeline/step2_encode.sh
```

### Phân công sessions

**OpenSSL (2 sessions):**

| Session | Tài khoản | `SHARD_INDEX` | `ENCODE_QUERIES` | Thời gian |
|---|---|---|---|---|
| A | account_1 | `0` | `true` | ~6h |
| B | account_2 | `1` | `false` | ~6h |

**Linux (8 sessions):**

| Session | Tài khoản | `SHARD_INDEX` | `ENCODE_QUERIES` |
|---|---|---|---|
| A | account_1 | `0` | `true` |
| B | account_2 | `1` | `false` |
| C | account_3 | `2` | `false` |
| D | account_4 | `3` | `false` |
| E | account_5 | `4` | `false` |
| F | account_6 | `5` | `false` |
| G | account_7 | `6` | `false` |
| H | account_8 | `7` | `false` |

### Lưu output

Sau khi mỗi session xong:
1. **Download** file(s) từ `/kaggle/working/output/`
2. **Upload** lên Kaggle Dataset `patchseeker-<project>-emb`:
   - `corpus_emb/corpus_00.pkl` (session A)
   - `corpus_emb/corpus_01.pkl` (session B)
   - `queries_emb/queries.pkl` (session A, từ ENCODE_QUERIES=true)
   - ...

---

## BƯỚC 4 — FAISS Retrieval (KAGGLE, 1 session cuối)

> Chạy **sau khi có đủ TẤT CẢ** shard pkl files.

### Setup

1. Tạo Kaggle notebook mới → GPU T4 x1
2. Add dataset: `patchseeker-<project>-emb` (chứa corpus_*.pkl + queries.pkl)

### Cell 1 — Clone repo

```python
%%bash
cd /kaggle/working
git clone https://github.com/Ngoc9392/PatchSeeker
```

### Cell 2 — FAISS Retrieval

```bash
export PROJECT=openssl    # hoặc linux
export TOTAL_SHARDS=2     # phải khớp với số shards đã encode

bash /kaggle/working/PatchSeeker/pipeline/step3_retrieve.sh
```

**Thời gian**: ~5 phút (OpenSSL) / ~30 phút (Linux)

### Lưu output

Download `ranking.txt` từ `/kaggle/working/output/`

---

## BƯỚC 5 — Evaluate (LOCAL)

> Copy `ranking.txt` về máy, chạy evaluation script.

```bash
python pipeline/step4_evaluate.py \
    --project openssl \
    --ranking-file path/to/ranking.txt \
    --show-predictions
```

**Output:**
- `metrics.json` — MRR, Recall@1/5/10, Manual Effort@10
- `predictions.json` — Top-10 commits cho mỗi CVE

> ⚠️ **Note**: MRR/Recall chỉ tính được cho CVEs có ground truth trong `qrels.txt`.  
> OpenSSL: ~0 CVEs có ground truth tự động.  
> Linux: ~28 CVEs (2%) có ground truth từ refs.  
> Để có full evaluation → cần Morefixes dataset.

---

## Cấu trúc thư mục

```
PatchSeeker/
├── pipeline/
│   ├── config.py               ← Project configs
│   ├── step1_prepare_data.py   ← LOCAL: fetch commits, build corpus
│   ├── step2_encode.sh         ← KAGGLE: encode 1 shard
│   ├── step3_retrieve.sh       ← KAGGLE: FAISS merge + ranking
│   ├── step4_evaluate.py       ← LOCAL: compute metrics
│   └── README.md               ← Tài liệu này
├── pipeline/data/
│   ├── openssl/                ← Output step 1
│   └── linux/
├── pipeline/repos/
│   ├── openssl.git             ← Blobless clone
│   └── linux.git
└── pipeline_test/              ← Test pipeline (5 CVEs)
```

---

## Troubleshooting

### NVD API timeout

```bash
# Nếu NVD API bị timeout, script tự dùng fallback (năm từ CVE ID)
# Chạy lại để fetch phần còn thiếu (cache tự động)
python pipeline/step1_prepare_data.py --project openssl
```

### Kaggle disk full

Linux kernel blobless clone (~2GB) + model (~4GB) + corpus (~500MB) = ~6.5GB.  
Kaggle working dir = 20GB → **OK**.

Nếu gặp vấn đề, clone trong `/kaggle/temp/` (không persistent nhưng lớn hơn):
```bash
export REPO_DIR=/kaggle/temp/PatchSeeker
```

### FAISS OOM (Out of Memory)

FAISS chạy trên CPU RAM, không phải VRAM. Với 390K commits × 4096 dims × 4 bytes = **6.4GB RAM**.  
Kaggle có 30GB RAM → **OK**.

### Missing shard file

```
❌ Missing: corpus_emb/corpus_03.pkl
```
→ Session tương ứng (SHARD_INDEX=3) chưa hoàn thành hoặc chưa upload.

---

## So sánh với Paper

| Metric | Paper | Bạn (NF4 4-bit) | Diff ước tính |
|---|---|---|---|
| MRR | 0.739 | ~0.720 | -2.6% |
| Recall@1 | — | — | — |
| Recall@10 | 0.871 | ~0.855 | -1.8% |
| Manual Effort@10 | — | — | — |

> Paper dùng GPU A100 (fp16). NF4 quantization mất khoảng 1-3% accuracy.
