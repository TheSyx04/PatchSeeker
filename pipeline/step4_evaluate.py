"""
pipeline/step4_evaluate.py
===========================
Bước 4 — Chạy LOCAL (không cần GPU).

Đọc ranking.txt từ FAISS retrieval và tính các metrics:
  - MRR (Mean Reciprocal Rank)
  - Recall@1, Recall@5, Recall@10
  - Manual Effort@10
  - Precision@10

Nếu không có qrels.txt, chỉ output top-10 predictions cho mỗi CVE.

Usage:
  python pipeline/step4_evaluate.py --project openssl --ranking-file path/to/ranking.txt
  python pipeline/step4_evaluate.py --project linux   --ranking-file path/to/ranking.txt --top-k 10
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Fix Windows Unicode
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.config import PROJECTS, DATA_BASE_DIR


# ═══════════════════════════════════════════════════════════════════════════
# Load functions
# ═══════════════════════════════════════════════════════════════════════════

def load_ranking(ranking_path: Path, top_k: int = 10) -> dict[str, list[tuple]]:
    """Load ranking.txt → {cve_id: [(commit_id, rank, score), ...]}"""
    raw: dict[str, list] = defaultdict(list)

    with open(ranking_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            cve_id, commit_id, score = parts[0], parts[1], float(parts[2])
            raw[cve_id].append((commit_id, score))

    rankings = {}
    for cve_id, entries in raw.items():
        entries.sort(key=lambda x: x[1], reverse=True)
        rankings[cve_id] = [
            (commit_id, rank + 1, score)
            for rank, (commit_id, score) in enumerate(entries[:top_k])
        ]
    return rankings


def load_qrels(qrels_path: Path) -> dict[str, set[str]]:
    """Load qrels.txt (TREC format) → {cve_id: {commit_id, ...}}"""
    qrels: dict[str, set] = defaultdict(set)

    if not qrels_path.exists():
        return qrels

    with open(qrels_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                cve_id, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
                if rel > 0:
                    # Cả hash đầy đủ và prefix 7/12 ký tự để match
                    qrels[cve_id].add(doc_id.lower())
                    qrels[cve_id].add(doc_id[:12].lower())
                    qrels[cve_id].add(doc_id[:7].lower())

    return dict(qrels)


def load_queries(queries_path: Path) -> dict[str, dict]:
    """Load queries.json → {cve_id: cve_obj}"""
    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["id"]] = obj
    return queries


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    rankings: dict[str, list],
    qrels: dict[str, set],
    top_k: int = 10,
) -> dict:
    """Tính MRR, Recall@K, Manual Effort@K theo paper."""
    if not qrels:
        return {}

    mrr_sum       = 0.0
    recall_at     = {1: 0, 5: 0, 10: 0}
    effort_sum    = 0
    n_evaluated   = 0

    for cve_id, vfc_set in qrels.items():
        if cve_id not in rankings:
            continue

        n_evaluated += 1
        ranked_commits = [commit_id for commit_id, rank, score in rankings[cve_id]]

        # MRR: 1/rank của VFC đầu tiên xuất hiện trong top-K
        rr = 0.0
        first_hit_rank = None
        for rank, commit_id in enumerate(ranked_commits[:top_k], start=1):
            if any(commit_id.lower().startswith(vh) or vh.startswith(commit_id.lower()[:7])
                   for vh in vfc_set):
                rr = 1.0 / rank
                first_hit_rank = rank
                break
        mrr_sum += rr

        # Recall@K: có VFC trong top-K không?
        for k in [1, 5, 10]:
            hits = sum(
                1 for c in ranked_commits[:k]
                if any(c.lower().startswith(vh) or vh.startswith(c.lower()[:7])
                       for vh in vfc_set)
            )
            if hits > 0:
                recall_at[k] += 1

        # Manual Effort@10: số commits cần review trước VFC đầu tiên
        if first_hit_rank is not None:
            effort_sum += first_hit_rank
        else:
            effort_sum += top_k   # phải review hết K mà không tìm được

    if n_evaluated == 0:
        return {}

    return {
        "n_evaluated":    n_evaluated,
        "mrr":            round(mrr_sum / n_evaluated, 4),
        "recall@1":       round(recall_at[1] / n_evaluated, 4),
        "recall@5":       round(recall_at[5] / n_evaluated, 4),
        "recall@10":      round(recall_at[10] / n_evaluated, 4),
        "manual_effort@10": round(effort_sum / n_evaluated, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PatchSeeker Step 4 — Evaluate ranking results"
    )
    parser.add_argument("--project",  required=True, choices=list(PROJECTS.keys()))
    parser.add_argument("--ranking-file", required=True, help="Path to ranking.txt")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--show-predictions", action="store_true",
                        help="In top-K predictions cho mỗi CVE")
    args = parser.parse_args()

    data_dir     = Path(DATA_BASE_DIR) / args.project
    ranking_path = Path(args.ranking_file)
    qrels_path   = data_dir / "qrels.txt"
    queries_path = data_dir / "queries.json"

    print("=" * 65)
    print(f"STEP 4: Evaluate — {args.project.upper()}")
    print(f"  Ranking: {ranking_path}")
    print(f"  Qrels:   {qrels_path}")
    print("=" * 65)

    # Load data
    rankings = load_ranking(ranking_path, top_k=args.top_k)
    qrels    = load_qrels(qrels_path)
    queries  = load_queries(queries_path) if queries_path.exists() else {}

    print(f"\n  Rankings loaded: {len(rankings)} CVEs")
    print(f"  Ground truth:   {len(qrels)} CVEs có VFC")

    # ── Metrics ──────────────────────────────────────────────────────────
    if qrels:
        print(f"\n{'─'*65}")
        print("EVALUATION METRICS")
        print(f"{'─'*65}")

        metrics = compute_metrics(rankings, qrels, top_k=args.top_k)
        if metrics:
            print(f"  CVEs evaluated:     {metrics['n_evaluated']}")
            print(f"  MRR:                {metrics['mrr']:.4f}")
            print(f"  Recall@1:           {metrics['recall@1']:.4f}")
            print(f"  Recall@5:           {metrics['recall@5']:.4f}")
            print(f"  Recall@10:          {metrics['recall@10']:.4f}")
            print(f"  Manual Effort@10:   {metrics['manual_effort@10']:.2f}")

            # So sánh với paper
            print(f"\n  [Paper benchmark] MRR=0.739, Recall@10=0.871")
            if metrics["mrr"] > 0:
                diff_mrr = (metrics["mrr"] - 0.739) / 0.739 * 100
                diff_r10 = (metrics["recall@10"] - 0.871) / 0.871 * 100
                print(f"  [Kết quả của bạn] "
                      f"MRR={metrics['mrr']:.3f} ({diff_mrr:+.1f}%), "
                      f"Recall@10={metrics['recall@10']:.3f} ({diff_r10:+.1f}%)")

            # Save metrics
            out_path = ranking_path.parent / "metrics.json"
            with open(out_path, "w") as f:
                json.dump({"project": args.project, **metrics}, f, indent=2)
            print(f"\n  ✅ Metrics saved → {out_path}")
    else:
        print("\n  ⚠️  Không có ground truth → bỏ qua metric calculation")

    # ── Top predictions ───────────────────────────────────────────────────
    if args.show_predictions or not qrels:
        print(f"\n{'─'*65}")
        print(f"TOP-{args.top_k} PREDICTIONS")
        print(f"{'─'*65}")

        for cve_id, entries in list(rankings.items())[:20]:   # giới hạn 20 CVEs
            cve_desc = queries.get(cve_id, {}).get("description", "")[:80]
            print(f"\n📌 {cve_id}")
            if cve_desc:
                print(f"   {cve_desc}...")
            for commit_id, rank, score in entries:
                vfc_mark = "✅ VFC" if any(
                    commit_id.lower().startswith(vh) or vh.startswith(commit_id.lower()[:7])
                    for vh in qrels.get(cve_id, set())
                ) else ""
                print(f"   #{rank:<3} {commit_id[:12]}  score={score:.4f}  {vfc_mark}")

    # ── Save predictions JSON ─────────────────────────────────────────────
    pred_path = ranking_path.parent / "predictions.json"
    predictions = {
        cve_id: [
            {"rank": rank, "commit_id": commit_id, "score": round(score, 6)}
            for commit_id, rank, score in entries
        ]
        for cve_id, entries in rankings.items()
    }
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Predictions saved → {pred_path}")

    print("\n" + "=" * 65)
    print("HOÀN THÀNH! Pipeline kết thúc.")
    print("=" * 65)


if __name__ == "__main__":
    main()
