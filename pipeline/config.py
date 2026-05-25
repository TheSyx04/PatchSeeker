"""
pipeline/config.py
==================
Cấu hình per-project cho PatchSeeker full pipeline.
Dùng exact settings theo paper (window_days=365).
"""

# ── Project configs ───────────────────────────────────────────────────────────
PROJECTS = {
    "linux": {
        "cve_file":      "linux_cves.jsonl",
        "repo_url":      "https://github.com/torvalds/linux",
        "repo_name":     "linux",
        "window_days":   365,        # paper: 1-year window trước CVE date
        "shallow_since": "2019-01-01", # Chỉ clone commits từ 2019+ (bắt đầu của 1-year window cho CVE 2020)
        "batch_size":    16,         # T4 16GB với 4-bit NF4
        "num_shards":    8,          # ~390K commits / ~50K per session
        "hf_repo":       "Ngoc9392/Patchseeker_Checkpoint",
        "model_folder":  "qwen3_8b",
        "base_model":    "Qwen/Qwen3-8B",
        "p_max_len":     512,
        "q_max_len":     512,
    },
    "openssl": {
        "cve_file":    "openssl_cves.jsonl",
        "repo_url":    "https://github.com/openssl/openssl",
        "repo_name":   "openssl",
        "window_days": 365,
        "batch_size":  16,
        "num_shards":  2,          # ~65K commits / ~33K per session
        "hf_repo":     "Ngoc9392/Patchseeker_Checkpoint",
        "model_folder": "qwen3_8b",
        "base_model":  "Qwen/Qwen3-8B",
        "p_max_len":   512,
        "q_max_len":   512,
    },
}

# ── Paths (relative to PatchSeeker repo root) ────────────────────────────────
PIPELINE_DIR  = "pipeline"
DATA_BASE_DIR = "pipeline/data"         # step1 output
REPOS_DIR     = "pipeline/repos"        # git clones
EVAL_SRC      = "eval/tevatron/src"     # tevatron source

# ── NVD API ──────────────────────────────────────────────────────────────────
NVD_API_URL   = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_RATE_WAIT = 6.5   # giây giữa các requests (tránh 5req/30s limit)

# ── Tevatron output format ───────────────────────────────────────────────────
RETRIEVAL_DEPTH = 10   # top-K commits per CVE
