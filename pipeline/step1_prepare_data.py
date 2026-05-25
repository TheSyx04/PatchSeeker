"""
pipeline/step1_prepare_data.py
================================
Bước 1 — Chạy LOCAL (không cần GPU).

Workflow:
  1. Clone repo (blobless bare clone — chỉ commit history, không download code)
  2. Fetch CVE publication dates từ NVD API (cache vào JSON)
  3. Với mỗi CVE: git log lấy tất cả commits trong 1-year window
  4. Dedup commits trùng nhau giữa các CVEs
  5. Extract ground truth từ refs (nếu có commit URL)
  6. Output corpus.json, queries.json, qrels.txt

Usage:
  python pipeline/step1_prepare_data.py --project openssl
  python pipeline/step1_prepare_data.py --project linux
  python pipeline/step1_prepare_data.py --project linux --skip-clone
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Fix UnicodeEncodeError trên Windows (cp1252 → utf-8)
sys.stdout.reconfigure(encoding="utf-8")

import requests
from tqdm import tqdm

# ── Thêm root vào PYTHONPATH để import config ─────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.config import PROJECTS, DATA_BASE_DIR, REPOS_DIR, NVD_API_URL, NVD_RATE_WAIT

# ── Regex để extract commit hash từ refs ─────────────────────────────────────
COMMIT_RE = re.compile(
    r"github\.com/[^/]+/[^/]+/commit/([0-9a-f]{7,40})"
    r"|git\.kernel\.org.*[?&]id=([0-9a-f]{7,40})"
    r"|openssl\.org.*[?&]id=([0-9a-f]{7,40})"
    r"|git\.openssl\.org.*commit/([0-9a-f]{7,40})"
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Git operations
# ═══════════════════════════════════════════════════════════════════════════

def clone_repo(repo_url: str, repo_path: Path) -> None:
    """Blobless bare clone — chỉ tải commit history, không tải file contents.
    
    Linux kernel: ~1-2GB  (so với ~5GB full clone)
    OpenSSL:      ~50MB
    """
    if repo_path.exists():
        print(f"  [skip] Repo đã tồn tại: {repo_path}")
        return

    print(f"  Cloning {repo_url}")
    print(f"  → {repo_path}")
    print(f"  Dùng --filter=blob:none (blobless) để giảm dung lượng...")

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git", "clone",
        "--filter=blob:none",   # không tải file blobs
        "--bare",               # không cần working directory
        repo_url,
        str(repo_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"  ✅ Clone xong: {repo_path}")


def get_commits_in_window(
    repo_path: Path,
    after_dt: datetime,
    before_dt: datetime,
) -> list[dict]:
    """Lấy tất cả commits trong [after_dt, before_dt] từ git log.
    
    Chạy 1 lệnh git log duy nhất → rất nhanh dù có 65K commits.
    Returns: [{"commit_id": str, "date": str, "msg": str}, ...]
    """
    # Dùng tab làm separator — safe trên Windows (không dùng \x00 vì Windows
    # không cho phép null bytes trong CreateProcess command-line arguments)
    # Format: "<hash>\t<date>\t<subject>"  — 1 dòng mỗi commit
    fmt = "%H\t%ai\t%s"

    cmd = [
        "git", f"--git-dir={repo_path}",
        "log",
        f"--format={fmt}",
        f"--after={after_dt.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"--before={before_dt.strftime('%Y-%m-%dT%H:%M:%S')}",
        "--no-merges",    # bỏ merge commits (ít informative)
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        timeout=600, errors="replace"
    )

    if result.returncode != 0:
        return []

    commits = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "<hash>\t<date ISO>\t<subject>"
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue

        commit_id = parts[0].strip()
        date_str  = parts[1].strip()[:10]   # YYYY-MM-DD
        subject   = parts[2].strip()

        if commit_id and len(commit_id) == 40:
            commits.append({
                "commit_id": commit_id,
                "date":      date_str,
                "msg":       subject,
            })

    return commits


# ═══════════════════════════════════════════════════════════════════════════
# 2. NVD API — lấy ngày publish của CVE
# ═══════════════════════════════════════════════════════════════════════════

def load_date_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    return {}


def save_date_cache(cache_path: Path, cache: dict) -> None:
    cache_path.write_text(json.dumps(cache, indent=2))


def fetch_cve_date_from_nvd(cve_id: str, api_key: str | None = None) -> str | None:
    """Lấy ngày publish CVE từ NVD API. Trả về 'YYYY-MM-DD' hoặc None."""
    params = {"cveId": cve_id}
    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = requests.get(NVD_API_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return None
        published = vulns[0]["cve"].get("published", "")
        return published[:10] if published else None
    except Exception:
        return None


def get_cve_dates(
    cves: list[dict],
    cache_path: Path,
    api_key: str | None = None,
) -> dict[str, str]:
    """Lấy publication dates cho tất cả CVEs. Cache kết quả vào file JSON.
    
    Returns: {cve_id: "YYYY-MM-DD"}
    Fallback nếu API fail: dùng năm từ CVE ID (e.g. CVE-2020-... → 2020-07-01)
    """
    cache = load_date_cache(cache_path)
    dates = {}

    to_fetch = [c for c in cves if c["id"] not in cache]

    if to_fetch:
        print(f"\n  Fetching {len(to_fetch)} CVE dates từ NVD API...")
        print(f"  (rate limit: {NVD_RATE_WAIT}s/request — est. {len(to_fetch)*NVD_RATE_WAIT/60:.0f} phút)")

        for cve in tqdm(to_fetch, desc="  NVD API"):
            cve_id = cve["id"]
            date = fetch_cve_date_from_nvd(cve_id, api_key)

            if date is None:
                # Fallback: lấy năm từ CVE ID
                m = re.search(r"CVE-(\d{4})-", cve_id)
                year = int(m.group(1)) if m else 2020
                date = f"{year}-07-01"   # giữa năm

            cache[cve_id] = date
            time.sleep(NVD_RATE_WAIT)   # rate limit

        save_date_cache(cache_path, cache)
        print(f"  ✅ Cached {len(cache)} CVE dates → {cache_path}")

    for cve in cves:
        dates[cve["id"]] = cache.get(cve["id"], f"2020-07-01")

    return dates


# ═══════════════════════════════════════════════════════════════════════════
# 3. Ground truth extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_vfc_from_refs(refs: list[str]) -> list[str]:
    """Extract commit hashes từ refs nếu là GitHub/git commit URLs."""
    hashes = []
    for ref in refs:
        m = COMMIT_RE.search(ref)
        if m:
            h = next(g for g in m.groups() if g)
            hashes.append(h.lower())
    return list(set(hashes))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PatchSeeker Step 1 — Prepare data from git repo"
    )
    parser.add_argument(
        "--project", required=True,
        choices=list(PROJECTS.keys()),
        help="Project name (linux hoặc openssl)"
    )
    parser.add_argument(
        "--skip-clone", action="store_true",
        help="Bỏ qua bước clone repo (nếu đã clone)"
    )
    parser.add_argument(
        "--nvd-api-key", default=None,
        help="NVD API key để tăng rate limit (optional)"
    )
    parser.add_argument(
        "--cve-file", default=None,
        help="Ghi đè đường dẫn file CVEs (mặc định từ config)"
    )
    args = parser.parse_args()

    cfg = PROJECTS[args.project]
    cve_file = Path(args.cve_file or cfg["cve_file"])
    repo_path = Path(REPOS_DIR) / f"{cfg['repo_name']}.git"
    out_dir   = Path(DATA_BASE_DIR) / args.project

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"STEP 1: Chuẩn bị data — {args.project.upper()}")
    print(f"  CVE file:    {cve_file}")
    print(f"  Repo:        {cfg['repo_url']}")
    print(f"  Window:      {cfg['window_days']} ngày")
    print(f"  Output dir:  {out_dir}")
    print("=" * 65)

    # ── 1. Load CVEs ─────────────────────────────────────────────────────
    print("\n[1/5] Load CVEs...")
    cves = []
    with open(cve_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cves.append(json.loads(line))
    print(f"  ✅ {len(cves)} CVEs loaded")

    # ── 2. Clone repo ─────────────────────────────────────────────────────
    print("\n[2/5] Clone repo (blobless bare)...")
    if not args.skip_clone:
        clone_repo(cfg["repo_url"], repo_path)
    else:
        print(f"  [skip] --skip-clone")

    if not repo_path.exists():
        print(f"  ❌ Repo không tồn tại: {repo_path}")
        sys.exit(1)

    # ── 3. Fetch CVE dates ────────────────────────────────────────────────
    print("\n[3/5] Fetch CVE publication dates từ NVD...")
    cache_path = out_dir / "nvd_dates_cache.json"
    cve_dates = get_cve_dates(cves, cache_path, args.nvd_api_key)

    # ── 4. Collect commits per CVE ────────────────────────────────────────
    print(f"\n[4/5] Collect commits (window={cfg['window_days']} ngày)...")
    t_start = time.time()

    all_commits: dict[str, dict] = {}        # commit_id → commit info
    cve_to_commits: dict[str, list[str]] = {}  # cve_id → [commit_ids]
    qrels: dict[str, list[str]] = {}          # cve_id → [vfc_commit_ids]

    for cve in tqdm(cves, desc="  CVEs"):
        cve_id = cve["id"]
        pub_date_str = cve_dates.get(cve_id, "2020-07-01")

        try:
            pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
        except ValueError:
            pub_date = datetime(2020, 7, 1)

        after_dt  = pub_date - timedelta(days=cfg["window_days"])
        before_dt = pub_date + timedelta(days=1)   # inclusive của pub_date

        commits = get_commits_in_window(repo_path, after_dt, before_dt)

        cve_to_commits[cve_id] = [c["commit_id"] for c in commits]
        for c in commits:
            if c["commit_id"] not in all_commits:
                all_commits[c["commit_id"]] = {
                    "commit_id": c["commit_id"],
                    "date":      c["date"],
                    "msg":       c["msg"],
                    "repo":      cfg["repo_name"],
                }

        # Ground truth: extract VFC từ refs
        vfcs = extract_vfc_from_refs(cve.get("refs", []))
        if vfcs:
            qrels[cve_id] = vfcs

    elapsed = time.time() - t_start
    print(f"\n  ✅ {len(cves)} CVEs xử lý xong trong {elapsed:.0f}s ({elapsed/60:.1f} phút)")
    print(f"  📦 Unique commits: {len(all_commits):,}")
    print(f"  🏷️  CVEs có ground truth: {len(qrels)}/{len(cves)}")

    # ── 5. Save output files ──────────────────────────────────────────────
    print(f"\n[5/5] Lưu output → {out_dir}")

    # corpus.json (JSONL)
    corpus_path = out_dir / "corpus.json"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for commit in all_commits.values():
            f.write(json.dumps(commit, ensure_ascii=False) + "\n")
    print(f"  ✅ corpus.json: {len(all_commits):,} commits ({corpus_path.stat().st_size/1e6:.1f} MB)")

    # queries.json (JSONL)
    queries_path = out_dir / "queries.json"
    with open(queries_path, "w", encoding="utf-8") as f:
        for cve in cves:
            obj = {
                "id":          cve["id"],
                "description": cve.get("description", cve.get("title", "")),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  ✅ queries.json: {len(cves)} CVEs")

    # qrels.txt (TREC format: query_id 0 doc_id relevance)
    qrels_path = out_dir / "qrels.txt"
    with open(qrels_path, "w", encoding="utf-8") as f:
        for cve_id, vfc_hashes in qrels.items():
            for vh in vfc_hashes:
                f.write(f"{cve_id} 0 {vh} 1\n")
    print(f"  ✅ qrels.txt: {sum(len(v) for v in qrels.values())} entries "
          f"({len(qrels)} CVEs có ground truth)")

    # cve_candidates.json — map CVE → candidate commit IDs (để evaluate chính xác)
    candidates_path = out_dir / "cve_candidates.json"
    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(cve_to_commits, f)
    print(f"  ✅ cve_candidates.json: saved")

    # stats.json — metadata cho reporting
    stats = {
        "project":         args.project,
        "n_cves":          len(cves),
        "n_unique_commits": len(all_commits),
        "n_with_qrels":    len(qrels),
        "window_days":     cfg["window_days"],
        "num_shards":      cfg["num_shards"],
        "generated_at":    datetime.now().isoformat(),
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 65)
    print("HOÀN THÀNH Step 1!")
    print(f"  Output: {out_dir}/")
    print(f"  Bước tiếp: Upload {out_dir}/ lên Kaggle Dataset")
    print(f"  Sau đó chạy step2_encode.sh với num_shards={cfg['num_shards']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
