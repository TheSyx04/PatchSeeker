"""
BƯỚC 3: Parse ranking output → xem kết quả VFC predictions
============================================================
Đọc ranking.txt từ PatchSeeker, hiển thị top commits cho mỗi CVE,
và tính toán ước tính thời gian cho full dataset.

Chạy: python pipeline_test/step3_parse_results.py
"""

import json
import os
from datetime import datetime

OUTPUT_DIR  = r"C:\Users\minhq\Documents\GitHub\PatchSeeker\pipeline_test\output"
DATA_DIR    = r"C:\Users\minhq\Documents\GitHub\PatchSeeker\pipeline_test\data"
RANKING_FILE = os.path.join(OUTPUT_DIR, "ranking.txt")
BENCHMARK_STEP1 = os.path.join(DATA_DIR, "benchmark_step1.json")
BENCHMARK_STEP2 = os.path.join(OUTPUT_DIR, "benchmark_step2.json")


def load_ranking(ranking_path, top_k=5):
    """Đọc file ranking, trả về dict {cve_id: [(commit_id, rank), ...]}."""
    rankings = {}
    with open(ranking_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cve_id, commit_id, rank = parts[0], parts[1], int(parts[2])
            if rank <= top_k:
                rankings.setdefault(cve_id, []).append((commit_id, rank))
    return rankings


def load_cves(cve_path):
    cves = {}
    with open(cve_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cves[obj["id"]] = obj
    return cves


def load_commits_index(commits_path):
    """Build index commit_id → commit info."""
    index = {}
    with open(commits_path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            index[c["commit_id"]] = c
    return index


def extrapolate_time(benchmark_step1, benchmark_step2,
                      full_commits_linux=252000,
                      full_commits_openssl=25000,
                      full_cves_linux=336,
                      full_cves_openssl=500):
    """
    Từ benchmark của 500 commits + 5 CVEs,
    ước tính thời gian cho full dataset.
    """
    n_commits_test = benchmark_step1["n_commits"]
    encode_corpus_sec = benchmark_step2["encode_corpus_sec"]
    encode_queries_sec = benchmark_step2["encode_queries_sec"]
    faiss_sec = benchmark_step2["faiss_rank_sec"]
    n_cves_test = benchmark_step2["n_cves"]

    # Encode corpus: tuyến tính theo số commits
    encode_per_commit = encode_corpus_sec / n_commits_test

    # Encode queries: tuyến tính theo số CVEs
    encode_per_cve = encode_queries_sec / n_cves_test

    # FAISS: ~logarithmic nhưng tính tuyến tính cho an toàn
    faiss_per_commit = faiss_sec / n_commits_test

    results = {}
    for name, n_c, n_q in [
        ("Linux (2020-2023, 252K commits)", full_commits_linux, full_cves_linux),
        ("Linux (2020-2021, 168K commits)", 168000, full_cves_linux),
        ("OpenSSL (~25K commits)", full_commits_openssl, full_cves_openssl),
    ]:
        corpus_est  = encode_per_commit * n_c
        queries_est = encode_per_cve    * n_q
        faiss_est   = faiss_per_commit  * n_c
        total_est   = corpus_est + queries_est + faiss_est

        results[name] = {
            "n_commits":         n_c,
            "n_cves":            n_q,
            "encode_corpus_h":   round(corpus_est / 3600, 1),
            "encode_queries_h":  round(queries_est / 3600, 2),
            "faiss_h":           round(faiss_est / 3600, 2),
            "total_h":           round(total_est / 3600, 1),
            "kaggle_weeks":      round(total_est / 3600 / 30, 1),  # 30h/tuần
        }
    return results


def main():
    print("=" * 65)
    print("BƯỚC 3: Kết quả PatchSeeker Test (5 CVEs Linux kernel)")
    print("=" * 65)

    # Load data
    rankings = load_ranking(RANKING_FILE, top_k=5)
    cves = load_cves(os.path.join(DATA_DIR, "test_cves.jsonl"))

    # Load commit index nếu có
    commits_path = os.path.join(DATA_DIR, "raw_commits.jsonl")
    commit_index = load_commits_index(commits_path) if os.path.exists(commits_path) else {}

    # Hiển thị kết quả ranking
    print(f"\n{'─'*65}")
    print(f"TOP-5 COMMITS CHO MỖI CVE")
    print(f"{'─'*65}")

    vfc_predictions = {}
    for cve_id, cve_data in cves.items():
        print(f"\n📌 {cve_id}")
        print(f"   Desc: {cve_data['description'][:100]}...")

        if cve_id not in rankings:
            print("   ⚠️  Không có ranking (CVE không được tìm thấy)")
            continue

        print(f"   {'Rank':<6} {'Commit Hash':<45} {'Message':<40}")
        print(f"   {'─'*4:<6} {'─'*10:<45} {'─'*20:<40}")

        for commit_id, rank in sorted(rankings[cve_id], key=lambda x: x[1]):
            commit_info = commit_index.get(commit_id, {})
            msg = commit_info.get("msg", "N/A")[:40]
            date = commit_info.get("date", "N/A")[:10]
            print(f"   #{rank:<5} {commit_id[:12]}... ({date})  {msg}")

        # Top-1 là predicted VFC
        top1_commit = rankings[cve_id][0][0]
        vfc_predictions[cve_id] = top1_commit
        print(f"   → Predicted VFC: {top1_commit[:12]}...")

    # Lưu VFC predictions
    vfc_path = os.path.join(OUTPUT_DIR, "vfc_predictions.json")
    with open(vfc_path, "w") as f:
        json.dump(vfc_predictions, f, indent=2)
    print(f"\n✅ VFC predictions lưu tại: {vfc_path}")

    # Ước tính thời gian full dataset
    print(f"\n{'─'*65}")
    print("ƯỚC TÍNH THỜI GIAN CHO FULL DATASET (từ benchmark)")
    print(f"{'─'*65}")

    if os.path.exists(BENCHMARK_STEP1) and os.path.exists(BENCHMARK_STEP2):
        with open(BENCHMARK_STEP1) as f:
            b1 = json.load(f)
        with open(BENCHMARK_STEP2) as f:
            b2 = json.load(f)

        print(f"\nBenchmark thực đo (test run):")
        print(f"  Commits test:          {b1['n_commits']}")
        print(f"  CVEs test:             {b2['n_cves']}")
        print(f"  Encode corpus time:    {b2['encode_corpus_sec']}s ({b2['encode_corpus_sec']/60:.1f} phút)")
        print(f"  Encode queries time:   {b2['encode_queries_sec']}s")
        print(f"  FAISS rank time:       {b2['faiss_rank_sec']}s")

        estimates = extrapolate_time(b1, b2)
        print(f"\nƯớc tính cho full dataset:")
        print(f"{'Dataset':<40} {'Commits':>8} {'CVEs':>6} {'Total(h)':>10} {'Kaggle(tuần)':>14}")
        print(f"{'─'*40} {'─'*8} {'─'*6} {'─'*10} {'─'*14}")

        for name, est in estimates.items():
            print(f"{name:<40} {est['n_commits']:>8,} {est['n_cves']:>6} "
                  f"{est['total_h']:>10.1f}h {est['kaggle_weeks']:>14.1f}")

        # Lưu ước tính
        est_path = os.path.join(OUTPUT_DIR, "time_estimates.json")
        with open(est_path, "w") as f:
            json.dump({
                "benchmark": {"step1": b1, "step2": b2},
                "estimates": estimates,
                "generated_at": datetime.now().isoformat()
            }, f, indent=2)
        print(f"\n📊 Ước tính đầy đủ lưu tại: {est_path}")

    else:
        print("⚠️  Chưa có file benchmark. Chạy step1 và step2 trước.")

    print(f"\n{'═'*65}")
    print("HOÀN THÀNH! Pipeline test đã chạy xong.")
    print("Kết quả:")
    print(f"  - VFC predictions: {vfc_path}")
    print(f"  - Ranking chi tiết: {RANKING_FILE}")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
