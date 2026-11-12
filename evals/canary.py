"""Calibrate the claim judge against answers we know.

    python -m evals.canary evals/results/<run> --yes

Anthropic's eval guidance says a model grader has to be checked against known labels before its numbers are
trusted. This takes the reports of an earlier run and builds claims whose right verdict is known without
asking any model:

- control: a cited sentence exactly as the report wrote it. The judge should not call it unsupported.
- misattributed: the same sentence, now citing a source on another topic (a source of another report in the
  folder, or the least related source of the same report when there is only one report).
- number_swapped: a figure in the sentence replaced by one that is in none of its cited sources.
- flipped: a word reversed (increase to decrease, higher to lower) or "can" to "cannot". English reports only.

Every altered claim is wrong, so the judge should not call it supported. The judge is blind to which claims
were altered. A control the judge rejects is either a judge false alarm or a claim the report really cannot
back, so read those too.
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from evals import judge
from evals.run import RESULTS_DIR, make_judge
from src.usage import UsageTracker
from src.utils import figures

KINDS = ("misattributed", "number_swapped", "flipped")
CAUGHT = ("unsupported", "contradicted", "unverified")
PLAIN = ("supported", "partial")
ABSTAINED = ("unknown", "unjudged", "not_a_claim")
TOKENS_PER_REPORT = 14_000
MAX_OVERLAP = 0.2
FOREIGN_FROM = 1000  # source numbers from other reports start here

_CONTENT_WORDS = re.compile(r"[^\W\d_]{4,}")
_PLAIN_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_GROUPED = re.compile(r"\d{1,3}(?:[.,]\d{3})+")

# Word pairs, English only, tried in this order; the first one found is applied. Only reversals that always
# change the meaning are used. A plain "is" to "is not" is left out: inside a subordinate clause it can leave
# the sentence true ("even when recall is not adequate").
_FLIPS = [
    (r"\bmore likely\b", "less likely"), (r"\bless likely\b", "more likely"),
    (r"\bincreases\b", "decreases"), (r"\bincreased\b", "decreased"), (r"\bincrease\b", "decrease"),
    (r"\bdecreases\b", "increases"), (r"\bdecreased\b", "increased"), (r"\bdecrease\b", "increase"),
    (r"\bhigher\b", "lower"), (r"\blower\b", "higher"), (r"\bimproves\b", "worsens"),
    (r"\breduces\b", "raises"), (r"\boutperforms\b", "underperforms"), (r"\bcan\b", "cannot"),
]


def _words(text: str) -> set[str]:
    return set(_CONTENT_WORDS.findall(text.lower()))


def misattribute(claim: dict, texts: dict[int, dict]) -> dict | None:
    words = _words(claim["text"])
    if not words:
        return None
    claimed = figures.all_number_forms(claim["text"])
    options = []
    for n, source in texts.items():
        content = str(source.get("content") or "")
        if n in claim["cites"] or not content:
            continue
        if claimed & figures.all_number_forms(content):
            continue
        options.append((len(words & _words(content)) / len(words), n))
    options = [option for option in options if option[1] >= FOREIGN_FROM] or options
    if not options or min(options)[0] > MAX_OVERLAP:
        return None
    return {**claim, "cites": [min(options)[1]]}


def swap_number(claim: dict, texts: dict[int, dict]) -> dict | None:
    available = set()
    for n in claim["cites"]:
        available |= figures.all_number_forms(str(texts.get(n, {}).get("content") or ""))
    for raw in figures.significant_numbers(claim["text"]):
        if not _PLAIN_NUMBER.fullmatch(raw) or _GROUPED.fullmatch(raw):
            continue
        place = re.search(r"(?<![\d.,])" + re.escape(raw) + r"(?![\d])", claim["text"])
        if not place:
            continue
        decimals = len(re.split(r"[.,]", raw)[1]) if re.search(r"[.,]", raw) else 0
        separator = "," if "," in raw else "."
        value = float(raw.replace(",", "."))
        for factor in (1.6, 0.6, 2.7):
            new = f"{value * factor:.{decimals}f}".replace(".", separator)
            if figures.number_forms(new) & available:
                continue
            return {**claim, "text": claim["text"][:place.start()] + new + claim["text"][place.end():]}
    return None


def flip(claim: dict, texts: dict[int, dict]) -> dict | None:
    for pattern, replacement in _FLIPS:
        match = re.search(pattern, claim["text"], re.I)
        if match:
            new = re.sub(pattern, replacement, match.group(0), flags=re.I)
            if match.group(0)[0].isupper():
                new = new[0].upper() + new[1:]
            text = claim["text"][:match.start()] + new + claim["text"][match.end():]
            return {**claim, "text": text}
    return None


MUTATIONS = {"misattributed": misattribute, "number_swapped": swap_number, "flipped": flip}


def build_variants(
    saved: dict, per_report: int, seed: int = 0, extra: dict[int, dict] | None = None,
) -> tuple[list[dict], dict[int, dict]]:
    """Controls and altered claims for one saved report, shuffled, with ids 1..n. `extra` holds sources of
    other reports: when there are any, a misattributed claim cites one of those (a source on another topic,
    so the claim is certainly not in it), otherwise the least related source of the same report."""
    texts = {int(n): text for n, text in (saved.get("source_texts") or {}).items()}
    texts.update(extra or {})
    english = saved.get("language") == "en"
    variants = []
    for index, claim in enumerate(judge.extract_claims(saved["report"], per_report)):
        variants.append({"kind": "control", "text": claim["text"], "cites": claim["cites"], "original": claim["text"]})
        for step in range(len(KINDS)):
            kind = KINDS[(index + step) % len(KINDS)]
            if kind == "flipped" and not english:
                continue
            altered = MUTATIONS[kind](claim, texts)
            if altered:
                variants.append({"kind": kind, "text": altered["text"], "cites": altered["cites"],
                                 "original": claim["text"]})
                break
    random.Random(seed).shuffle(variants)
    for number, variant in enumerate(variants, 1):
        variant["id"] = number
    return variants, texts


def outcome(kind: str, verdict: str) -> str:
    """What a verdict means for a claim whose right answer is known. "Unknown" and "not a claim" (a sentence about
    the report itself has no truth value, altered or not) are left out and counted as abstained."""
    if verdict in ABSTAINED:
        return "abstained"
    if kind == "control":
        return "alarm" if verdict in CAUGHT else ("ok" if verdict in PLAIN else "abstained")
    if verdict in CAUGHT:
        return "caught"
    return "soft" if verdict == "partial" else "missed"


def tally(rows: list[dict]) -> dict:
    result = {}
    for kind in ("control", *KINDS):
        mine = [row for row in rows if row["kind"] == kind]
        counts = {name: sum(1 for row in mine if row["outcome"] == name)
                  for name in ("ok", "alarm", "caught", "soft", "missed", "abstained")}
        decided = len(mine) - counts["abstained"]
        entry = {"n": len(mine), **counts}
        if kind == "control":
            entry["false_alarm_rate"] = round(counts["alarm"] / decided, 3) if decided else None
        else:
            entry["recall"] = round(counts["caught"] / decided, 3) if decided else None
            entry["recall_with_partial"] = round((counts["caught"] + counts["soft"]) / decided, 3) if decided else None
        result[kind] = entry
    return result


def run_canary(saved_reports: list[dict], judge_llm, per_report: int = 8) -> dict:
    tracker = UsageTracker()
    rows = []
    for index, saved in enumerate(saved_reports):
        foreign = {
            1000 * (other + 1) + int(n): text
            for other, report in enumerate(saved_reports) if other != index
            for n, text in (report.get("source_texts") or {}).items()
        }
        variants, texts = build_variants(saved, per_report, extra=foreign)
        # Controls and altered claims are judged in separate batches. A sentence and its misattributed copy
        # in one batch confused the judge in the first run: it described the copy's missing source as the
        # original's, and rejected an unchanged claim.
        for group in ([v for v in variants if v["kind"] == "control"], [v for v in variants if v["kind"] != "control"]):
            numbered = [{**variant, "id": number} for number, variant in enumerate(group, 1)]
            if not numbered:
                continue
            verdicts = judge.judge_claims(judge_llm, numbered, texts, [tracker])["rows"]
            for variant, verdict in zip(numbered, verdicts, strict=True):
                rows.append({
                    "case": saved.get("case_id", saved["id"]), "kind": variant["kind"], "text": variant["text"],
                    "original": variant["original"], "cites": variant["cites"], "verdict": verdict["verdict"],
                    "judge_verdict": verdict["judge_verdict"], "quote": verdict["quote"], "note": verdict["note"],
                    "outcome": outcome(variant["kind"], verdict["verdict"]),
                })
    return {"judge_version": judge.VERSION, "tally": tally(rows), "rows": rows, "usage": tracker.snapshot()}


def print_report(result: dict) -> None:
    print(f"\nJudge version {result['judge_version']}")
    print(f"{'kind':16s} {'n':>3s}  result")
    for kind, entry in result["tally"].items():
        if kind == "control":
            rate = entry["false_alarm_rate"]
            detail = (f"ok {entry.get('ok', 0)}, flagged {entry.get('alarm', 0)}, abstained {entry['abstained']}"
                      f"  -> false alarm rate {'-' if rate is None else format(rate, '.0%')}")
        else:
            rate = entry["recall"]
            detail = (f"caught {entry.get('caught', 0)}, partial {entry.get('soft', 0)}, missed {entry.get('missed', 0)}, "
                      f"abstained {entry['abstained']}  -> recall {'-' if rate is None else format(rate, '.0%')}")
        print(f"{kind:16s} {entry['n']:3d}  {detail}")
    for outcome_name, title in (("missed", "Altered claims the judge called supported"),
                                ("alarm", "Unchanged claims the judge rejected")):
        rows = [row for row in result["rows"] if row["outcome"] == outcome_name]
        if rows:
            print(f"\n{title}:")
            for row in rows:
                print(f"  [{row['case']}/{row['kind']}] {row['verdict']}: {row['text'][:150]}")
                print(f"      judge: {row['note'][:200]}")


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="python -m evals.canary", description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", help="a results folder from evals.run, with saved reports")
    parser.add_argument("--per-report", type=int, default=8, help="claims taken from each report (default 8)")
    parser.add_argument("--cases", help="comma-separated case ids (default: all in the folder)")
    parser.add_argument("--judge-model", help="model for the judge (default: the app's main model)")
    parser.add_argument("--yes", action="store_true", help="really run (it spends OpenAI credits)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Not a results folder: {folder}")
    ids = args.cases.split(",") if args.cases else None
    saved = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json")) if p.name != "summary.json"]
    saved = [s for s in saved if s.get("source_texts") and (ids is None or s.get("case_id", s["id"]) in ids)]
    if not saved:
        raise SystemExit(f"No saved reports with source texts in {folder}")
    print(f"{len(saved)} report(s), up to {args.per_report} claims each, each shown once unchanged and once altered.")
    print(f"Rough cost: ~{len(saved) * TOKENS_PER_REPORT:,} tokens.")
    if not args.yes:
        print("Nothing was run. Add --yes to start.")
        return 0
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing in the environment: OPENAI_API_KEY")

    result = run_canary(saved, make_judge(args.judge_model), args.per_report)
    print_report(result)
    out = RESULTS_DIR / f"canary-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
