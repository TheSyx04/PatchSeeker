"""
BƯỚC 1: Chuẩn bị dữ liệu test với 5 CVE Linux kernel
========================================================
Giống hệt pipeline thật, chỉ giới hạn số commits để đo thời gian.

Pipeline thật dùng:
- PyDriller để extract commits (giống bước này)
- corpus.json + queries.json format giống bước này
- PatchSeeker checkpoint từ HuggingFace (bước 2)

Optimizations (so với baseline):
1. git log → hashes (~0.5s) thay vì Repository.traverse_commits() (~10+ phút block)
2. ThreadPoolExecutor: mỗi worker dùng PyDriller single=hash → parallel diff extract
3. Pre-built corpus cache: extract 1 lần, lần sau load tức thì

Bước này chạy trên LOCAL MACHINE (CPU), không cần GPU.
Output: data/ folder → upload lên Kaggle Dataset để chạy bước 2.

Chạy: python pipeline_test/step1_prepare_test_data.py
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from pydriller import Repository

# ─── CONFIG ────────────────────────────────────────────────────────────────────
CVE_FILE    = r"C:\Users\minhq\Downloads\đatn\datasets_vulguard\PatchSeeker\linux_cves.jsonl"
REPO_PATH   = r"C:\Users\minhq\Documents\GitHub\PatchSeeker\pipeline_test\linux_repo"
OUTPUT_DIR  = r"C:\Users\minhq\Documents\GitHub\PatchSeeker\pipeline_test\data"
N_TEST_CVES = 5
MAX_COMMITS = 500          # Giới hạn cho test (pipeline thật: None = không giới hạn)
SINCE       = datetime(2020, 1, 1)
TO          = datetime(2025, 12, 31)

# Parallel workers: None = auto (cpu_count - 1)
NUM_WORKERS = None

# Cache: extract 1 lần, lưu lại, lần sau load tức thì
CACHE_FILE  = r"C:\Users\minhq\Documents\GitHub\PatchSeeker\pipeline_test\data\raw_commits_cache.jsonl"
# ────────────────────────────────────────────────────────────────────────────────


def log(msg="", **kwargs):
    """Print với flush để không bị buffer khi pipe qua Tee-Object."""
    print(msg, flush=True, **kwargs)


# ─── CVE selection ─────────────────────────────────────────────────────────────

def select_test_cves(cve_file, n=5):
    """Lấy N CVEs Linux kernel đầu tiên từ file."""
    selected = []
    with open(cve_file, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            desc = obj["description"].lower()
            if "linux kernel" in desc and len(obj["description"]) > 100:
                selected.append(obj)
            if len(selected) == n:
                break
    return selected


# ─── Clone ─────────────────────────────────────────────────────────────────────

def clone_linux(repo_path):
    """
    Clone Linux kernel — FULL clone, không shallow.
    Kiểm tra .git/shallow để phát hiện shallow clone (instant, không block).
    """
    if os.path.exists(repo_path):
        shallow_file = os.path.join(repo_path, ".git", "shallow")
        if os.path.exists(shallow_file):
            log(f"⚠️  Repo là SHALLOW CLONE (.git/shallow tồn tại).")
            log("    Đang unshallow (git fetch --unshallow)...")
            t0 = time.time()
            subprocess.run(
                ["git", "-C", repo_path, "fetch", "--unshallow"],
                check=True
            )
            log(f"    Unshallow xong! {(time.time()-t0)/60:.1f} phút")
        else:
            log(f"Repo đã tồn tại (full clone): {repo_path}")
        return

    log("Cloning Linux kernel (FULL clone, không shallow)...")
    log("⚠️  Sẽ tốn 1–3 giờ và ~6–10GB disk")
    t0 = time.time()
    subprocess.run([
        "git", "clone",
        "--single-branch", "--branch", "master",
        "https://github.com/torvalds/linux",
        repo_path
    ], check=True)
    log(f"Clone xong! {(time.time()-t0)/60:.1f} phút")


# ─── Pre-built corpus cache ────────────────────────────────────────────────────

def _cache_meta(since, to, repo_path):
    return {
        "since": since.isoformat(),
        "to":    to.isoformat(),
        "repo":  os.path.abspath(repo_path),
    }


def load_corpus_cache(cache_file, since, to, repo_path, max_commits=None):
    """Load cache nếu tồn tại và hợp lệ. Trả về list hoặc None."""
    meta_file = cache_file + ".meta.json"
    if not os.path.exists(cache_file) or not os.path.exists(meta_file):
        return None

    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("params") != _cache_meta(since, to, repo_path):
        log("  Cache metadata không khớp → bỏ cache cũ.")
        return None

    commits = []
    with open(cache_file, encoding="utf-8") as f:
        for line in f:
            commits.append(json.loads(line))

    if max_commits and len(commits) < max_commits:
        log(f"  Cache có {len(commits)} commits < {max_commits} cần → re-extract.")
        return None

    return commits[:max_commits] if max_commits else commits


def save_corpus_cache(commits, cache_file, since, to, repo_path):
    """Lưu commits vào cache."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        for c in commits:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    meta = {
        "params":    _cache_meta(since, to, repo_path),
        "n_commits": len(commits),
        "cached_at": datetime.now().isoformat(),
    }
    with open(cache_file + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log(f"  Cache lưu: {len(commits)} commits → {cache_file}")


# ─── Per-hash worker ───────────────────────────────────────────────────────────

def _extract_hash(args):
    """
    Worker: extract diff + metadata từ 1 commit hash.
    Dùng PyDriller single=hash — chỉ load đúng commit đó, không scan toàn repo.
    Thread-safe (mỗi thread có Repository object riêng).
    """
    repo_path, commit_hash = args
    try:
        for commit in Repository(repo_path, single=commit_hash).traverse_commits():
            diff_text = ""
            modified_files = []
            for mod in commit.modified_files:
                if mod.diff:
                    diff_text += mod.diff[:3000]      # 3KB/file
                modified_files.append(mod.filename)
            return {
                "commit_id": commit.hash,
                "owner":     "torvalds",
                "repo":      "linux",
                "date":      commit.author_date.isoformat(),
                "msg":       commit.msg[:1000],
                "diff":      diff_text[:5000],         # 5KB total
                "files":     modified_files[:20],
            }
    except Exception:
        return None


# ─── Main extract function ─────────────────────────────────────────────────────

def extract_commits_pydriller(repo_path, since, to, max_commits=None,
                               num_workers=None, cache_file=None):
    """
    Extract commits — GIỐNG HỆT pipeline thật + optimized.

    Flow:
      Cache hit?  → load file (< 1s)
      Cache miss? → git log (0.5s) → ThreadPool per-hash → save cache
    """
    log()
    log("Extracting commits với PyDriller (git log + parallel)...")
    log(f"  Repo:  {repo_path}")
    log(f"  Since: {since.date()}  →  To: {to.date()}")

    n_workers = num_workers or max(1, (os.cpu_count() or 4) - 1)
    log(f"  Workers: {n_workers} threads")
    if max_commits:
        log(f"  Giới hạn: {max_commits} commits (test mode)")
    log()

    # ── 1. Cache check ───────────────────────────────────────────
    if cache_file:
        cached = load_corpus_cache(cache_file, since, to, repo_path, max_commits)
        if cached is not None:
            log(f"  ✅ CACHE HIT — {len(cached)} commits (< 1s, không cần extract)")
            log(f"     {cache_file}")
            return cached, 0.001

        log("  Cache miss → extract từ repo...")
        log()

    # ── 2. git log lấy hashes (~0.5s cho mọi repo size) ─────────
    log("  [2a] git log lấy commit hashes...", end="")
    t_log = time.time()

    result = subprocess.run([
        "git", "-C", repo_path, "log",
        "--format=%H",
        f"--after={since.strftime('%Y-%m-%d')}",
        f"--before={to.strftime('%Y-%m-%d')}",
        "--first-parent",    # main branch only
    ], capture_output=True, text=True, check=True)

    all_hashes = [h.strip() for h in result.stdout.splitlines() if h.strip()]
    all_hashes.reverse()   # git log newest-first → oldest-first

    if max_commits:
        all_hashes = all_hashes[:max_commits]

    log(f" {len(all_hashes):,} hashes ({time.time()-t_log:.2f}s)")

    # ── 3. Parallel extract per commit hash ─────────────────────
    log(f"  [2b] Extract diff song song ({n_workers} workers)...")
    t0 = time.time()
    commits = []
    done = 0
    total = len(all_hashes)

    args_list = [(repo_path, h) for h in all_hashes]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_extract_hash, a): a for a in args_list}

        for future in as_completed(futures):
            result = future.result()
            if result:
                commits.append(result)
            done += 1

            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                last = commits[-1] if commits else {}
                log(f"  [{done:>5}/{total}] "
                    f"{last.get('commit_id','--------')[:8]} | "
                    f"{last.get('msg', '')[:45]:<45} | "
                    f"{rate:.1f} commits/s")

    elapsed = time.time() - t0

    # Sắp xếp theo date (as_completed không đảm bảo thứ tự)
    commits.sort(key=lambda c: c["date"])

    log()
    log(f"Extracted {len(commits)} commits trong {elapsed:.1f}s ({elapsed/60:.1f} phút)")
    log(f"Tốc độ: {len(commits)/elapsed:.2f} commits/giây")

    # ── 4. Lưu cache ─────────────────────────────────────────────
    if cache_file:
        save_corpus_cache(commits, cache_file, since, to, repo_path)

    return commits, elapsed


# ─── Format output ─────────────────────────────────────────────────────────────

def build_patchseeker_input(commits, cves, output_dir):
    """
    Format corpus.json và queries.json cho PatchSeeker.
    Giống hệt format mà PatchSeeker eval scripts mong đợi.
    """
    os.makedirs(output_dir, exist_ok=True)

    # corpus.json
    corpus = [{
        "docid":    c["commit_id"],
        "text":     (c["msg"] + " " + c.get("diff", ""))[:2000],
        "query_id": "",
        "query":    ""
    } for c in commits]

    corpus_path = os.path.join(output_dir, "corpus.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    log(f"corpus.json:       {len(corpus):>6} commits → {corpus_path}")

    # queries.json
    queries = [{"query_id": cve["id"], "query": cve["description"]} for cve in cves]
    queries_path = os.path.join(output_dir, "queries.json")
    with open(queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    log(f"queries.json:      {len(queries):>6} CVEs    → {queries_path}")

    # raw_commits.jsonl
    commits_path = os.path.join(output_dir, "raw_commits.jsonl")
    with open(commits_path, "w", encoding="utf-8") as f:
        for c in commits:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    log(f"raw_commits.jsonl: {len(commits):>6} commits → {commits_path}")

    return corpus_path, queries_path


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Force unbuffered output (không bị buffer khi pipe)
    sys.stdout.reconfigure(line_buffering=True)

    t_total_start = time.time()

    log("=" * 65)
    log("BƯỚC 1: Chuẩn bị dữ liệu test (5 CVE Linux kernel)")
    log("Pipeline: giống thật + git log + parallel + corpus cache")
    log("=" * 65)

    # ── 1. Chọn CVEs ─────────────────────────────────────────────
    log(f"\n[1/4] Chọn {N_TEST_CVES} CVEs Linux kernel từ file...")
    cves = select_test_cves(CVE_FILE, N_TEST_CVES)
    log(f"\nĐã chọn {len(cves)} CVEs:")
    for cve in cves:
        log(f"  ✓ {cve['id']}: {cve['description'][:70]}...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "test_cves.jsonl"), "w", encoding="utf-8") as f:
        for cve in cves:
            f.write(json.dumps(cve, ensure_ascii=False) + "\n")
    log(f"\n  Lưu tại: {OUTPUT_DIR}/test_cves.jsonl")

    # ── 2. Clone ─────────────────────────────────────────────────
    log(f"\n[2/4] Clone Linux kernel repo (nếu chưa có)...")
    clone_linux(REPO_PATH)

    # ── 3. Extract commits ───────────────────────────────────────
    label = f"{MAX_COMMITS} commits test" if MAX_COMMITS else "FULL (pipeline thật)"
    log(f"\n[3/4] Extract commits bằng PyDriller ({label})...")
    commits, extract_time = extract_commits_pydriller(
        repo_path=REPO_PATH,
        since=SINCE,
        to=TO,
        max_commits=MAX_COMMITS,
        num_workers=NUM_WORKERS,
        cache_file=CACHE_FILE,
    )

    # ── 4. Format cho PatchSeeker ────────────────────────────────
    log(f"\n[4/4] Format corpus.json + queries.json cho PatchSeeker...")
    build_patchseeker_input(commits, cves, OUTPUT_DIR)

    # ── Summary ──────────────────────────────────────────────────
    total_time = time.time() - t_total_start
    is_cache = extract_time < 1.0
    commits_per_sec = len(commits) / extract_time if not is_cache and extract_time > 0 else None
    full_linux_commits = 420_000
    est_hours = (full_linux_commits / commits_per_sec) / 3600 if commits_per_sec else None

    log("\n" + "=" * 65)
    log("HOÀN THÀNH Bước 1!")
    log(f"\n  Kết quả:")
    log(f"    CVEs:              {len(cves)}")
    log(f"    Commits extracted: {len(commits)}")
    if is_cache:
        log(f"    Extract time:      (từ cache — gần tức thì)")
    else:
        log(f"    Extract time:      {extract_time:.1f}s ({extract_time/60:.1f} phút)")
        log(f"    Tốc độ PyDriller:  {commits_per_sec:.3f} commits/giây  [parallel]")
    log(f"    Total time:        {total_time:.1f}s ({total_time/60:.1f} phút)")

    if est_hours:
        log(f"\n  Ước tính (từ benchmark này):")
        log(f"    Full Linux (420K commits): ~{est_hours:.0f} giờ ({est_hours/24:.1f} ngày)")

    log(f"\n  Data lưu tại: {OUTPUT_DIR}")
    log("\n  Tiếp theo:")
    log("    → Upload thư mục data/ lên Kaggle Dataset")
    log("    → Chạy step2_run_patchseeker.sh trên Kaggle (cần GPU)")
    log("=" * 65)

    # Benchmark log
    log_data = {
        "timestamp":       datetime.now().isoformat(),
        "mode":            "cache_hit" if is_cache else "parallel_git_log",
        "num_workers":     NUM_WORKERS or max(1, (os.cpu_count() or 4) - 1),
        "n_cves":          len(cves),
        "n_commits":       len(commits),
        "since":           SINCE.isoformat(),
        "to":              TO.isoformat(),
        "extract_time_sec": round(extract_time, 2),
        "total_time_sec":  round(total_time, 2),
        "commits_per_sec": round(commits_per_sec, 4) if commits_per_sec else None,
        "est_full_linux_hours": round(est_hours, 1) if est_hours else None,
        "cve_ids":         [c["id"] for c in cves],
        "tool":            "PyDriller+git-log+ThreadPoolExecutor",
    }
    log_path = os.path.join(OUTPUT_DIR, "benchmark_step1.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    log(f"\n  Benchmark lưu tại: {log_path}")


if __name__ == "__main__":
    main()
