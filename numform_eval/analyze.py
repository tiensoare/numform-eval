"""
Combine and compare results CSVs written by runner.py.

Each row already carries a `method` column, but that alone isn't enough to
tell apart two `limit` runs at different --max-den (e.g. den=100 vs
den=1000) once their CSVs are combined -- max_den isn't stored per row, only
baked into the output filename runner.py chooses
(results_<model>_limit_den<N>_<jobid>.csv). So this script derives a
`series` label per row: `limit` rows get `limit(den=N)` recovered from the
source filename, everything else keeps its method name as-is.

Output goes under analysis/ (created automatically) -- a bare filename for
--out or --summary lands there; a path with its own directory is used as-is.

Usage:
    python -m numform_eval.analyze results/results_*_7247733.csv
    python -m numform_eval.analyze results/results_*_7247733.csv --out combined.csv
"""

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import List

ANALYSIS_DIR = "analysis"
DEN_RE = re.compile(r"den(\d+)")


def series_for(row: dict, path: str) -> str:
    if row["method"] == "limit":
        m = DEN_RE.search(path)
        return f"limit(den={m.group(1)})" if m else "limit"
    return row["method"]


def load(paths: List[str]) -> List[dict]:
    rows = []
    for pattern in paths:
        matches = glob.glob(pattern) or [pattern]
        for path in matches:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    row["series"] = series_for(row, path)
                    row["_source"] = path
                    rows.append(row)
    return rows


def _render_table(title: str, row_header: str, row_keys: List[str],
                   col_keys: List[str], cell: dict) -> List[str]:
    """cell: dict[(row_key, col_key)] -> formatted string, missing = '-'."""
    col_w = {
        c: max(len(c), max((len(cell.get((r, c), "-")) for r in row_keys), default=0))
        for c in col_keys
    }
    row_w = max(len(row_header), max((len(r) for r in row_keys), default=0))

    lines = [title]
    header = f"  {row_header:<{row_w}} | " + " | ".join(f"{c:<{col_w[c]}}" for c in col_keys)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for r in row_keys:
        vals = " | ".join(f"{cell.get((r, c), '-'):<{col_w[c]}}" for c in col_keys)
        lines.append(f"  {r:<{row_w}} | {vals}")
    return lines


def _acc_cells(rows: List[dict], group_key) -> dict:
    """dict[(group_key(r) or 'all', series)] -> 'pct% (c/n)'."""
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        key = ("all", r["series"]) if group_key is None else (group_key(r), r["series"])
        counts[key][1] += 1
        counts[key][0] += int(r["correct"] == "True")
    return {k: f"{c/n:.0%} ({c}/{n})" for k, (c, n) in counts.items()}


def _token_cells(rows: List[dict]) -> dict:
    sums = defaultdict(lambda: [0, 0])
    for r in rows:
        ct = r.get("completion_tokens")
        if ct in (None, "", "None"):
            continue
        sums[("tokens", r["series"])][0] += int(ct)
        sums[("tokens", r["series"])][1] += 1
    return {k: f"{total/n:.0f}" for k, (total, n) in sums.items()}


def build_report(rows: List[dict], n_paths: int) -> str:
    lines = []
    series = sorted({r["series"] for r in rows})
    lines.append(f"Loaded {len(rows)} rows from {n_paths} path(s); "
                  f"series: {', '.join(series)}\n")

    lines += _render_table("Overall accuracy by series:", "", ["all"], series,
                            _acc_cells(rows, group_key=None))

    kinds = sorted({r["kind"] for r in rows})
    lines.append("")
    lines += _render_table("Accuracy by kind x series:", "kind", kinds, series,
                            _acc_cells(rows, group_key=lambda r: r["kind"]))

    tasks = sorted({r["task_id"] for r in rows})
    lines.append("")
    lines += _render_table("Accuracy by task x series:", "task", tasks, series,
                            _acc_cells(rows, group_key=lambda r: r["task_id"]))

    lines.append("")
    lines += _render_table("Avg completion tokens by series:", "", ["tokens"], series,
                            _token_cells(rows))

    return "\n".join(lines) + "\n"


def _under_analysis_dir(path: str) -> str:
    if not os.path.dirname(path):
        path = os.path.join(ANALYSIS_DIR, path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="results CSV(s), globs allowed")
    ap.add_argument("--out", help="write combined rows (with `series` column) to this CSV; "
                                  f"a bare filename lands under {ANALYSIS_DIR}/")
    ap.add_argument("--summary", default="summary.txt",
                    help=f"write the report here instead of printing it; "
                         f"a bare filename lands under {ANALYSIS_DIR}/ (default: summary.txt)")
    args = ap.parse_args()

    rows = load(args.files)
    if not rows:
        raise SystemExit("No rows loaded -- check the file paths/globs.")

    report = build_report(rows, len(args.files))

    summary_path = _under_analysis_dir(args.summary)
    with open(summary_path, "w") as f:
        f.write(report)
    print(f"Wrote summary to {summary_path}")

    if args.out:
        out_path = _under_analysis_dir(args.out)
        fields = list(rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} combined rows to {out_path}")


if __name__ == "__main__":
    main()
