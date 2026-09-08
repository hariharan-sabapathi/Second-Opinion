"""Cost/latency/accuracy benchmark for the retrieval-grading loop, and the
failure-analysis harness that finds cases where the self-grader disagreed
with reality.

What this answers, with real numbers instead of guesses:

  1. "What does each extra round actually cost, in latency and tokens?"
     -> per-round and cumulative figures, from rag_graph.round_metrics.
  2. "Was capping at 3 rounds the right call?"
     -> run the same eval set with the round cap forced to 1, 2 and 3 and
        compare cost growth against accuracy growth.
  3. "Where does the grader get it wrong?"
     -> cross-check every grading verdict against an objective oracle (do the
        phrases a correct answer needs actually appear in that round's
        retrieved context?), independent of what the grader believed.

Usage:
    cp eval/testset.example.json eval/testset.json   # then fill in real
                                                       # questions against a
                                                       # real PDF
    uv run scripts/benchmark.py --testset eval/testset.json

Requires GEMINI_API_KEY (and, for $ cost, GEMMA_INPUT_PRICE_PER_1M /
GEMMA_OUTPUT_PRICE_PER_1M) in .env — see .env.example. Every LLM call in this
script bills against that key; a 3-case, cap=[1,2,3] run costs roughly what a
handful of manual questions in the app would.

Writes:
    eval/results/raw_metrics.csv        one row per (case, round-cap)
    eval/results/summary.md             the round 1/2/3 cost/latency/accuracy table
    eval/results/failure_analysis.md    up to 5 false-pass + 5 false-reject cases

Paste summary.md and failure_analysis.md into README.md yourself once you've
read the failure cases and filled in the "why" for each — that write-up is
the point of the exercise, not something this script should do for you.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import rag_graph  # noqa: E402  (path must be set up first)

# Mirrors main.py's chunking settings. Duplicated rather than imported because
# importing main.py would also build and try to launch its Gradio UI at
# module load time — keep the two in sync by hand if you change one.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def index_pdf(pdf_path: Path, persist_dir: Path) -> tuple[int, int]:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    Chroma.from_documents(chunks, rag_graph.embeddings, persist_directory=str(persist_dir))
    return len(chunks), len(pages)


def load_testset(path: Path) -> dict:
    data = json.loads(path.read_text())
    cases = [c for c in data["cases"] if not str(c.get("id", "")).startswith("_")]
    if not cases:
        raise ValueError(f"{path} has no cases — copy eval/testset.example.json and fill it in first")
    return {"pdf": data["pdf"], "cases": cases}


def score_answer(answer: str, case: dict) -> bool:
    """Deterministic keyword scoring, not an LLM judge — grading the grader's
    architecture with more calls to the same model it's built from would
    fold the thing under test into the thing doing the measuring."""
    phrases = case["expected_answer_contains"]
    answer_l = (answer or "").lower()
    hits = [p.lower() in answer_l for p in phrases]
    return all(hits) if case.get("match_mode", "any") == "all" else any(hits)


def context_has_answer(context: str, case: dict) -> bool:
    """The objective oracle for 'was the needed information actually
    retrieved this round', independent of the grader's own verdict."""
    phrases = case["expected_answer_contains"]
    context_l = (context or "").lower()
    return any(p.lower() in context_l for p in phrases)


def run_case(app, case: dict) -> dict:
    started = time.perf_counter()
    result = app.invoke({
        "question": case["question"],
        "refined_query": "",
        "context": "",
        "reflection": "",
        "answer": "",
        "iterations": 0,
        "reflection_log": [],
        "context_log": [],
        "round_metrics": [],
    })
    result["_wall_s"] = time.perf_counter() - started
    return result


def parse_verdict(log_entry: str) -> str:
    for line in log_entry.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            return "YES" if "YES" in line.upper() else "NO"
    return "UNKNOWN"


def parse_reason(log_entry: str) -> str:
    for line in log_entry.splitlines():
        if line.strip().upper().startswith("REASON:"):
            return line.split(":", 1)[1].strip()
    return "(no reason parsed)"


def find_failures(case: dict, result: dict) -> list[dict]:
    """Cross-check every round's grader verdict against the objective oracle.
    Returns candidate false-pass / false-reject cases for human write-up —
    the *mechanical* disagreement is real; the two-sentence 'why' still needs
    a human (or a follow-up Claude pass) reading the actual chunk text."""
    failures = []
    reflection_log = result.get("reflection_log", [])
    context_log = result.get("context_log", [])
    for round_num, (log_entry, context) in enumerate(zip(reflection_log, context_log), start=1):
        verdict = parse_verdict(log_entry)
        oracle_good = context_has_answer(context, case)
        if verdict == "YES" and not oracle_good:
            kind = "false_pass"  # grader approved retrieval that didn't actually have the answer
        elif verdict == "NO" and oracle_good:
            kind = "false_reject"  # grader rejected retrieval that did have the answer
        else:
            continue
        failures.append({
            "case_id": case["id"],
            "question": case["question"],
            "round": round_num,
            "kind": kind,
            "verdict": verdict,
            "grader_reason": parse_reason(log_entry),
            "expected_phrases": case["expected_answer_contains"],
            "context": context,
        })
    return failures


def render_summary_table(rows: list[dict]) -> str:
    header = (
        "| Round cap | Mean cumulative latency (s) | Mean cumulative tokens | "
        "Mean cumulative cost | Accuracy | Δ accuracy |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = []
    prev_acc = None
    for row in rows:
        cost = f"${row['mean_cost']:.4f}" if row["mean_cost"] is not None else "n/a (pricing not set)"
        delta = "—" if prev_acc is None else f"{(row['accuracy'] - prev_acc) * 100:+.0f}pp"
        lines.append(
            f"| {row['cap']} | {row['mean_latency']:.1f} | {row['mean_tokens']:,.0f} | "
            f"{cost} | {row['accuracy'] * 100:.0f}% | {delta} |"
        )
        prev_acc = row["accuracy"]
    n = rows[0]["n"] if rows else 0
    footer = f"\n*n = {n} eval case(s). Generated by `scripts/benchmark.py` on {time.strftime('%Y-%m-%d')}.*\n"
    return header + "\n".join(lines) + "\n" + footer


def render_failure_section(failures: list[dict], top_n: int) -> str:
    false_pass = [f for f in failures if f["kind"] == "false_pass"][:top_n]
    false_reject = [f for f in failures if f["kind"] == "false_reject"][:top_n]

    def render_group(title, items, missing_note):
        out = [f"### {title}\n"]
        if not items:
            out.append(f"*None found in this run. {missing_note}*\n")
            return "\n".join(out)
        for i, f in enumerate(items, start=1):
            out.append(
                f"**{i}. `{f['case_id']}` (round {f['round']}) — {f['question']}**\n\n"
                f"- Grader verdict: `{f['verdict']}` — \"{f['grader_reason']}\"\n"
                f"- Oracle phrase(s) checked: {', '.join(f['expected_phrases'])}\n"
                f"- Why (2 sentences): _TODO — read the retrieved context for this round "
                f"and explain concretely why the grader's verdict didn't match reality._\n"
            )
        return "\n".join(out)

    parts = [
        render_group(
            "Grader passed bad retrieval (false pass)", false_pass,
            "Expand eval/testset.json with more edge-case questions to surface some.",
        ),
        "",
        render_group(
            "Grader rejected good retrieval (false reject)", false_reject,
            "Expand eval/testset.json with more edge-case questions to surface some.",
        ),
    ]
    parts.append(f"\n*Generated by `scripts/benchmark.py` on {time.strftime('%Y-%m-%d')}.*\n")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", type=Path, default=REPO_ROOT / "eval" / "testset.json")
    parser.add_argument("--pdf", type=Path, default=None, help="Override the PDF path from the testset file")
    parser.add_argument("--caps", type=int, nargs="+", default=[1, 2, 3], help="Round caps to compare")
    parser.add_argument("--top-n-failures", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "eval" / "results")
    args = parser.parse_args()

    if not args.testset.exists():
        parser.error(
            f"{args.testset} not found. Copy eval/testset.example.json to {args.testset.name} "
            "and fill in real questions against a real PDF first."
        )

    testset = load_testset(args.testset)
    pdf_path = args.pdf or (REPO_ROOT / testset["pdf"])
    if not pdf_path.exists():
        parser.error(f"PDF not found: {pdf_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="second-opinion-bench-") as tmp:
        persist_dir = Path(tmp) / "chroma_db"
        n_chunks, n_pages = index_pdf(pdf_path, persist_dir)
        print(f"Indexed {n_chunks} chunks from {n_pages} pages of {pdf_path.name}")

        # rag_graph.get_db() reads this module global at call time, so pointing
        # it at a scratch index here does not touch the app's real chroma_db/.
        rag_graph.CHROMA_DB_PATH = str(persist_dir)

        raw_rows = []
        all_failures = []
        max_cap = max(args.caps)

        for cap in args.caps:
            app = rag_graph.build_app(max_iterations=cap)
            for case in testset["cases"]:
                print(f"  cap={cap} case={case['id']!r} ...", end=" ", flush=True)
                result = run_case(app, case)
                rounds = rag_graph.summarize_by_round(result.get("round_metrics", []))
                final = rounds[-1] if rounds else None
                correct = score_answer(result.get("answer", ""), case)
                print(f"{'PASS' if correct else 'FAIL'} in {result.get('iterations', 0)} round(s)")

                raw_rows.append({
                    "case_id": case["id"],
                    "cap": cap,
                    "rounds_used": result.get("iterations", 0),
                    "wall_s": round(result["_wall_s"], 3),
                    "cumulative_latency_s": final["cumulative_latency_s"] if final else None,
                    "cumulative_total_tokens": final["cumulative_total_tokens"] if final else None,
                    "cumulative_cost_usd": final["cumulative_cost_usd"] if final else None,
                    "correct": correct,
                })

                if cap == max_cap:
                    all_failures.extend(find_failures(case, result))

    # ── Aggregate the round-cap comparison table ──
    summary_rows = []
    for cap in args.caps:
        cap_rows = [r for r in raw_rows if r["cap"] == cap]
        n = len(cap_rows)
        latencies = [r["cumulative_latency_s"] for r in cap_rows if r["cumulative_latency_s"] is not None]
        tokens = [r["cumulative_total_tokens"] for r in cap_rows if r["cumulative_total_tokens"] is not None]
        costs = [r["cumulative_cost_usd"] for r in cap_rows if r["cumulative_cost_usd"] is not None]
        summary_rows.append({
            "cap": cap,
            "n": n,
            "mean_latency": sum(latencies) / len(latencies) if latencies else 0.0,
            "mean_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
            "mean_cost": (sum(costs) / len(costs)) if costs and len(costs) == n else None,
            "accuracy": sum(r["correct"] for r in cap_rows) / n if n else 0.0,
        })

    summary_md = render_summary_table(summary_rows)
    failures_md = render_failure_section(all_failures, args.top_n_failures)

    # ── Write files ──
    csv_path = args.out_dir / "raw_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()) if raw_rows else [])
        writer.writeheader()
        writer.writerows(raw_rows)

    (args.out_dir / "summary.md").write_text(summary_md)
    (args.out_dir / "failure_analysis.md").write_text(failures_md)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {args.out_dir / 'summary.md'}")
    print(f"Wrote {args.out_dir / 'failure_analysis.md'}")
    print("\nRead the failure cases, write the two-sentence 'why' for each, then paste both")
    print("files' contents into README.md.")


if __name__ == "__main__":
    main()
