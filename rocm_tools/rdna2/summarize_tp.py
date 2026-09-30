#!/usr/bin/env python3
"""Repo-native summariser for `tp_run.py` reports (stdlib only, no GPU work).

Promotes the private `summarize.py` used for the context-batch runs into an
importable, path-agnostic CLI:

    python3 -m rocm_tools.rdna2.summarize_tp REPORT.json [REPORT2.json ...]

For every report it writes the sibling artifact `<report-stem>-summary.json`
and prints one concise line of per-language medians to stdout.

Arithmetic is NOT reimplemented here: the batched-throughput block comes from
`tp_run.group_throughput_metrics` (identical semantics to the run itself), and
the extra figures keep the exact formulas the published runs were summarised
with --

  * full decode span: start = earliest delivery of any job, end = latest
    delivery of any job; tokens delivered at `start` are the step-function
    count (last sample with t <= start, 0 if none), the delta is the last
    cumulative count minus that, and `aggregate_tps` = delta / span duration,
    `per_sequence_average_tps` = the same divided by the job count. This span
    INCLUDES the final queue-drain delivery delay and EXCLUDES prefill and the
    first delivery burst;
  * common window / per-job / TTFT / end-to-end / medians: `group_throughput_metrics`;
  * conservative prefill: summed prompt tokens over the LAST first-delivery
    (`sum(prompt_tokens) / max(first_delivery_s)`), never a per-job average;
  * acceptance: summed accepted over summed accepted+rejected draft tokens;
  * per language: median / min / max across that language's timed groups.

Incomplete, `capacity_only` or `validation_only` reports are rejected: they
carry no timed runs, so any rate derived from them would be fiction. Group
membership is verified against the bound `ids_sha256` and the run index, and
`runs` must be fully consumed -- a report whose rows and groups disagree fails
loudly instead of silently mixing batches.

Nothing in this module touches a GPU: `tp_run` imports its native stack lazily,
so `import` and `--help` are safe on a CPU-only box.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from rocm_tools.rdna2.tp_run import group_throughput_metrics

USAGE = "usage: python3 -m rocm_tools.rdna2.summarize_tp REPORT.json [REPORT2.json ...]"


class ReportError(ValueError):
    """A report is unusable as evidence (incomplete, non-timed, or inconsistent)."""


def check_report(data, *, name="report"):
    """Return `data` if it is a complete, timed run report; raise ReportError."""
    if not isinstance(data, dict):
        raise ReportError(f"{name}: top level must be a JSON object")
    for flag in ("capacity_only", "validation_only"):
        if data.get(flag):
            raise ReportError(f"{name}: {flag} report carries no timed runs; refusing to "
                              "derive throughput from it")
    if not data.get("complete"):
        raise ReportError(f"{name}: report is not complete (it stopped before the run "
                          "finished); refusing to summarise partial timing")
    for key in ("groups", "runs"):
        if not isinstance(data.get(key), list):
            raise ReportError(f"{name}: missing/invalid '{key}' list")
    if not data["groups"]:
        raise ReportError(f"{name}: no groups recorded")
    return data


def _row_field(row, key, where):
    if key not in row:
        raise ReportError(f"{where}: run row is missing '{key}'")
    return row[key]


_REQUIRED_ROW_KEYS = ("ids_sha256", "delivery_events", "language", "repeat",
                      "prompt_tokens", "first_delivery_s")


def summarize(data, *, name="report"):
    """Summarise ONE validated report dict into the published summary structure."""
    check_report(data, name=name)
    runs = data["runs"]
    groups, idx = [], 0
    for g in data["groups"]:
        if not isinstance(g, dict):
            raise ReportError(f"{name}: group entry must be an object")
        where = f"{name}: group {g.get('group')!r}"
        try:
            group_id, timed = g["group"], g["timed"]
            jobs, wall_s, bound = g["jobs"], g["wall_s"], list(g["ids_sha256"])
        except (KeyError, TypeError) as e:
            raise ReportError(f"{where}: malformed group entry ({e!r})") from e
        rows = runs[idx:idx + jobs]
        if len(rows) != jobs:
            raise ReportError(f"{where}: wants {jobs} rows but only {len(rows)} remain at "
                              f"index {idx}")
        if [_row_field(r, "ids_sha256", where) for r in rows] != bound:
            raise ReportError(f"{where}: bound ids_sha256 does not match runs[{idx}:{idx + jobs}]")
        for row in rows:
            for key in _REQUIRED_ROW_KEYS:
                _row_field(row, key, where)
        metrics = group_throughput_metrics(rows, wall_s=wall_s, run_index_offset=idx)

        events = [r["delivery_events"] for r in rows]
        try:
            start = min(e[0][0] for e in events)
            end = max(e[-1][0] for e in events)
            delivered_start = sum(max((n for t, n in e if t <= start), default=0) for e in events)
            delta = sum(e[-1][1] for e in events) - delivered_start
            span = {"start_s": start, "end_s": end, "duration_s": end - start,
                    "token_delta": delta, "aggregate_tps": delta / (end - start),
                    "per_sequence_average_tps": delta / (end - start) / len(rows),
                    "note": "includes final queue-drain delivery delay; excludes prefill and "
                            "the first delivery burst"}
            # conservative prefill: ALL inputs over the LAST first delivery
            prefill_tps = sum(r["prompt_tokens"] for r in rows) / max(
                r["first_delivery_s"] for r in rows)
        except (IndexError, TypeError, ValueError, ZeroDivisionError) as e:
            raise ReportError(f"{where}: delivery events / prompt fields do not give a usable "
                              f"decode span ({e!r})") from e
        metrics["full_decode_span"] = span

        groups.append({"group": group_id, "timed": timed,
                       "language": rows[0]["language"], "repeat": rows[0]["repeat"],
                       "metrics": metrics,
                       "prompt_tokens_each": [r["prompt_tokens"] for r in rows],
                       "aggregate_prefill_to_last_first_delivery_tps": prefill_tps,
                       "accepted": sum(r.get("accepted_draft_tokens") or 0 for r in rows),
                       "rejected": sum(r.get("rejected_draft_tokens") or 0 for r in rows)})
        idx += jobs
    if idx != len(runs):
        raise ReportError(f"{name}: groups cover {idx} of {len(runs)} runs; report is "
                          "internally inconsistent")

    out = {"report": name, "groups": groups, "languages": _by_language(groups, name)}
    audit = data.get("tp_final_audit") or {}
    out["peak_GiB"] = {str(rank["device"]): rank["torch_peak_bytes"] / 2 ** 30
                       for rank in audit.get("ranks", [])}
    return out


def _per_sequence_common(g):
    """Common-window aggregate over the job count; nulls propagate, never imputed."""
    agg = g["metrics"]["common_window"]["aggregate_decode_tps"]
    jobs = g["metrics"]["jobs"]
    return None if agg is None or not jobs else agg / jobs


def _by_language(groups, name="report"):
    """median/min/max across each language's TIMED groups, plus token-weighted acceptance.

    Only `group_throughput_metrics`' own nulls are refused, never imputed: a timed
    group without a usable window is evidence the report must not be published.
    """
    languages = {}
    for lang in sorted({g["language"] for g in groups}):
        timed = [g for g in groups if g["timed"] and g["language"] == lang]
        if not timed:
            continue
        values = {
            "full_span_aggregate_decode_tps": [g["metrics"]["full_decode_span"]["aggregate_tps"]
                                              for g in timed],
            "full_span_per_sequence_decode_tps": [
                g["metrics"]["full_decode_span"]["per_sequence_average_tps"] for g in timed],
            "aggregate_decode_tps": [g["metrics"]["common_window"]["aggregate_decode_tps"]
                                    for g in timed],
            "per_sequence_common_decode_tps": [_per_sequence_common(g) for g in timed],
            "aggregate_prefill_tps": [g["aggregate_prefill_to_last_first_delivery_tps"]
                                     for g in timed],
            "last_ttft_s": [g["metrics"]["first_delivery_s"]["max"] for g in timed],
            "end_to_end_tps": [g["metrics"]["end_to_end"]["aggregate_tps"] for g in timed],
        }
        for key, vals in values.items():
            if any(v is None for v in vals):
                raise ReportError(f"{name}: language {lang!r} has a timed group with a null "
                                  f"{key}; refusing to summarise unusable timing")
        accepted = sum(g["accepted"] for g in timed)
        rejected = sum(g["rejected"] for g in timed)
        languages[lang] = {"groups": len(timed),
                           "acceptance": accepted / (accepted + rejected)
                                       if accepted + rejected else None,
                           **{key: {"median": statistics.median(vals), "min": min(vals),
                                    "max": max(vals)} for key, vals in values.items()}}
    return languages


def load_and_summarize(path):
    """Read, validate and summarise one report FILE -> (summary dict, sibling path)."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except OSError as e:
        raise ReportError(f"{path}: cannot read ({e!r})") from e
    except ValueError as e:
        raise ReportError(f"{path}: not valid JSON ({e})") from e
    return summarize(raw, name=str(path)), path.with_name(path.stem + "-summary.json")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or any(a in ("-h", "--help") for a in argv):
        print(__doc__.strip().splitlines()[0])
        print(USAGE)
        return 0 if argv else 2
    failed = False
    for arg in argv:
        try:
            out, dest = load_and_summarize(arg)
        except ReportError as e:
            print(f"summarize_tp: error: {e}", file=sys.stderr)
            failed = True
            continue
        dest.write_text(json.dumps(out, indent=2) + "\n")
        print(Path(arg).stem, json.dumps(out["languages"], sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
