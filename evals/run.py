"""Run the agent on a fixed set of topics and score the reports.

    python -m evals.run --smoke --yes                       # 3 English cases
    python -m evals.run --yes --label before-change         # every case
    python -m evals.run --yes --baseline evals/results/<earlier-run>
    python -m evals.run --rescore evals/results/<run> --yes # score saved reports again, no agent run

A run calls the real models and search, so it costs money. Without --yes it only prints the plan.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from evals import judge, metrics
from src.agent import builder
from src.settings import get_settings
from src.usage import UsageTracker

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "cases.json"
RESULTS_DIR = HERE / "results"
SMOKE = ("rag-en", "fictional-en", "false-premise-en")
TOKENS_PER_ANALYST = 35_000
JUDGE_TOKENS_PER_CASE = 12_000
SCORED_KEYS = ("metrics", "checks", "claims", "rubric", "judge_usage")
MEAN_KEYS = ("words", "subheadings", "sources", "authority_share", "max_host_share", "marketing_sources")


def load_cases(ids: list[str] | None = None) -> list[dict]:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if ids is None:
        return cases
    by_id = {case["id"]: case for case in cases}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}. Known: {', '.join(by_id)}")
    return [by_id[i] for i in ids]


def make_graph():
    saver = MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[("src.utils.objects", "Analyst")]))
    return builder.compile(checkpointer=saver)


def make_judge(model: str | None):
    from src.utils.models import get_llm

    if not model:
        return get_llm()
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(model=model, timeout=settings.llm_timeout, max_retries=settings.llm_max_retries)


def score(result: dict, texts: dict[int, dict], case: dict, judge_llm, max_claims: int) -> dict:
    """Add the checks and, when a judge is given, the model-graded results to a saved run."""
    scored = {key: value for key, value in result.items() if key not in SCORED_KEYS}
    measured = metrics.evaluate_report(scored["report"], scored["sources"])
    scored.update(metrics=measured, checks=metrics.run_checks(measured))
    if judge_llm is not None and texts:
        judge_tracker = UsageTracker()
        claims = judge.extract_claims(scored["report"], max_claims)
        scored["claims"] = judge.judge_claims(judge_llm, claims, texts, [judge_tracker])
        if case.get("rubric"):
            scored["rubric"] = judge.judge_rubric(judge_llm, scored["report"], case["rubric"], [judge_tracker])
        scored["judge_usage"] = judge_tracker.snapshot()
    return scored


def run_case(case: dict, analysts: int = 2, judge_llm=None, max_claims: int = 24) -> dict:
    graph = make_graph()
    tracker = UsageTracker()
    config = {
        "configurable": {"thread_id": f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"},
        "callbacks": [tracker],
        "recursion_limit": 80,
    }
    started = time.monotonic()
    graph.invoke({"topic": case["topic"], "max_analyst": analysts, "language": case["language"]}, config)
    final = graph.invoke(Command(resume=""), config)
    elapsed = time.monotonic() - started

    sources = final["final_sources"]
    by_url = {source["url"]: source for source in final.get("sources") or []}
    texts = {
        source["n"]: {
            "title": source["title"], "url": source["url"],
            "content": str(by_url.get(source["url"], {}).get("content") or ""),
        }
        for source in sources
    }
    saved = {
        "id": case["id"], "topic": case["topic"], "language": case["language"], "category": case["category"],
        "analysts": analysts, "usage": tracker.snapshot(elapsed), "report": final["final_report"],
        "sources": [{key: source.get(key) for key in ("n", "url", "title", "kind")} for source in sources],
        "source_texts": texts, "warnings": final.get("warnings") or [],
        "review_notes": final.get("review_notes") or [],
    }
    return score(saved, texts, case, judge_llm, max_claims)


def rescore(saved: dict, judge_llm, max_claims: int = 24) -> dict:
    case = next((c for c in load_cases() if c["id"] == saved["id"]), {})
    texts = {int(n): text for n, text in (saved.get("source_texts") or {}).items()}
    return score(saved, texts, case, judge_llm, max_claims)


def slim(result: dict) -> dict:
    """The part of a result that is kept in summary.json and compared between runs."""
    if "error" in result:
        return {"error": result["error"]}
    row = {"metrics": result["metrics"], "checks": result["checks"], "usage": result["usage"]}
    if "claims" in result:
        row["claims"] = {k: v for k, v in result["claims"].items() if k != "rows"}
    for key in ("rubric", "judge_usage"):
        if key in result:
            row[key] = result[key]
    return row


def aggregate(rows: list[dict]) -> dict:
    ok = [row for row in rows if "error" not in row]
    agg: dict = {"cases": len(rows), "errors": len(rows) - len(ok)}
    for key in MEAN_KEYS:
        agg[key] = round(statistics.fmean(row["metrics"][key] for row in ok), 3) if ok else None
    checks = [passed for row in ok for passed in row["checks"].values()]
    agg["hard_checks_passed"] = round(sum(checks) / len(checks), 3) if checks else None

    judged = [row["claims"] for row in ok if row.get("claims")]
    counted = sum(c["judged"] for c in judged)
    score_sum = sum(c["supported"] + 0.5 * c["partial"] for c in judged)
    agg["faithfulness"] = round(score_sum / counted, 3) if counted else None
    agg["unsupported_claims"] = sum(c["unsupported"] for c in judged) if judged else None
    agg["contradicted_claims"] = sum(c["contradicted"] for c in judged) if judged else None

    graded = [row["rubric"]["passed"] for row in ok if row.get("rubric") and row["rubric"]["passed"] is not None]
    agg["honesty_pass_rate"] = round(sum(graded) / len(graded), 3) if graded else None
    agg["tokens"] = sum(row["usage"]["total_tokens"] + row.get("judge_usage", {}).get("total_tokens", 0) for row in ok)
    agg["seconds_mean"] = round(statistics.fmean(row["usage"]["elapsed"] for row in ok), 1) if ok else None
    return agg


def compare(old: dict, new: dict) -> list[str]:
    lines = []
    for key in new:
        before, after = old.get(key), new[key]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (before, after))
        if numeric and before != after:
            lines.append(f"  {key:22s} {before} -> {after}  ({after - before:+.3g})")
    return lines


def load_baseline(path: str) -> dict:
    target = Path(path)
    target = target / "summary.json" if target.is_dir() else target
    if not target.exists():
        raise SystemExit(f"Baseline not found: {target}")
    return json.loads(target.read_text(encoding="utf-8"))


def load_saved(folder: str, ids: list[str] | None) -> list[dict]:
    path = Path(folder)
    if not path.is_dir():
        raise SystemExit(f"Not a results folder: {path}")
    saved = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json")) if p.name != "summary.json"]
    saved = [s for s in saved if "report" in s and (ids is None or s["id"] in ids)]
    if not saved:
        raise SystemExit(f"No saved reports found in {path}")
    return saved


def print_table(results: list[dict]) -> None:
    print(f"{'case':26s} {'words':>5s} {'srcs':>4s} {'auth':>5s} {'host':>5s} {'checks':>7s} "
          f"{'faith':>6s} {'unsup':>5s} {'honest':>6s} {'tokens':>8s} {'sec':>4s}")
    for result in results:
        if "error" in result:
            print(f"{result['id']:26s} ERROR  {result['error']}")
            continue
        m, c = result["metrics"], result.get("claims") or {}
        rubric = (result.get("rubric") or {}).get("passed")
        honest = "-" if rubric is None else ("pass" if rubric else "FAIL")
        faith = "-" if c.get("faithfulness") is None else f"{c['faithfulness']:.2f}"
        tokens = result["usage"]["total_tokens"] + result.get("judge_usage", {}).get("total_tokens", 0)
        passed = sum(result["checks"].values())
        print(f"{result['id']:26s} {m['words']:5d} {m['sources']:4d} {m['authority_share']:5.0%} "
              f"{m['max_host_share']:5.0%} {passed:>3d}/{len(result['checks']):<3d} {faith:>6s} "
              f"{c.get('unsupported', '-')!s:>5s} {honest:>6s} {tokens:8,d} {result['usage']['elapsed']:4.0f}")
        failed = [name for name, ok in result["checks"].items() if not ok]
        if failed:
            print(f"    failed checks: {', '.join(failed)}")
        if rubric is False:
            print(f"    rubric: {result['rubric']['reason']}")


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--smoke", action="store_true", help=f"only {', '.join(SMOKE)}")
    parser.add_argument("--rescore", metavar="FOLDER", help="score the reports saved in an earlier run again")
    parser.add_argument("--analysts", type=int, default=2, help="analysts per research (default 2)")
    parser.add_argument("--parallel", type=int, default=3, help="researches at once (default 3)")
    parser.add_argument("--no-judge", action="store_true", help="skip the model-graded checks (cheaper)")
    parser.add_argument("--judge-model", help="model for the judge (default: the app's main model)")
    parser.add_argument("--max-claims", type=int, default=24, help="claims judged per report (default 24)")
    parser.add_argument("--label", default="", help="name added to the results folder")
    parser.add_argument("--baseline", help="an earlier results folder or summary.json to compare with")
    parser.add_argument("--yes", action="store_true", help="really run (it spends OpenAI and Tavily credits)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    ids = args.cases.split(",") if args.cases else (list(SMOKE) if args.smoke else None)
    judging = not args.no_judge
    if args.rescore:
        saved = load_saved(args.rescore, ids)
        names = [s["id"] for s in saved]
        estimate = len(saved) * JUDGE_TOKENS_PER_CASE if judging else 0
        print(f"Rescoring {len(saved)} saved report(s), judge {'on' if judging else 'off'}: {', '.join(names)}")
        needs_yes, needs_keys = judging, ("OPENAI_API_KEY",) if judging else ()
    else:
        cases = load_cases(ids)
        names = [case["id"] for case in cases]
        per_case = args.analysts * TOKENS_PER_ANALYST + (JUDGE_TOKENS_PER_CASE if judging else 0)
        estimate = len(cases) * per_case
        print(f"{len(cases)} case(s), {args.analysts} analyst(s) each, judge {'on' if judging else 'off'}: "
              f"{', '.join(names)}")
        needs_yes, needs_keys = True, ("OPENAI_API_KEY", "TAVILY_API_KEY")
    if needs_yes:
        print(f"Rough cost: ~{estimate:,} tokens" + ("" if args.rescore else " plus Tavily searches") + ".")
        if not args.yes:
            print("Nothing was run. Add --yes to start.")
            return 0
    missing = [name for name in needs_keys if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Missing in the environment: {', '.join(missing)}")

    judge_llm = make_judge(args.judge_model) if judging else None
    if args.rescore:
        jobs = {s["id"]: (lambda s=s: rescore(s, judge_llm, args.max_claims)) for s in saved}
    else:
        jobs = {c["id"]: (lambda c=c: run_case(c, args.analysts, judge_llm, args.max_claims)) for c in cases}

    label = re.sub(r"[^\w.-]", "-", args.label).strip("-")
    prefix = "rescore" if args.rescore else ""
    suffix = "-".join(part for part in (prefix, label) if part)
    out_dir = RESULTS_DIR / (datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{suffix}" if suffix else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    def safe(case_id: str):
        try:
            return jobs[case_id]()
        except Exception as exc:
            return {"id": case_id, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [pool.submit(safe, case_id) for case_id in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results[result["id"]] = result
            status = "failed" if "error" in result else f"done in {result['usage']['elapsed']:.0f}s"
            print(f"[{done}/{len(jobs)}] {result['id']} {status}", flush=True)
            (out_dir / f"{result['id']}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if "report" in result:
                (out_dir / f"{result['id']}.md").write_text(result["report"], encoding="utf-8")

    ordered = [results[case_id] for case_id in names]
    rows = {result["id"]: slim(result) for result in ordered}
    first = next((r for r in ordered if "analysts" in r), {})
    summary = {
        "label": args.label, "created": datetime.now().isoformat(timespec="seconds"),
        "analysts": first.get("analysts", args.analysts), "judge": judging,
        "cases": rows, "aggregate": aggregate(list(rows.values())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print_table(ordered)
    print("\nAggregate:")
    for key, value in summary["aggregate"].items():
        print(f"  {key:22s} {value}")

    if args.baseline:
        old = load_baseline(args.baseline)
        shared = [i for i in rows if i in old["cases"]]
        if not shared:
            print("\nNo shared cases with the baseline, nothing to compare.")
        else:
            print(f"\nChange vs baseline '{old.get('label') or args.baseline}' on {len(shared)} shared case(s):")
            lines = compare(aggregate([old["cases"][i] for i in shared]), aggregate([rows[i] for i in shared]))
            print("\n".join(lines) if lines else "  no difference")
    print(f"\nSaved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
