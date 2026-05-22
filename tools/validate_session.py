"""
Validate a recorded session. Reads CSVs + manifest and reports quality.

Usage:
    python -m tools.validate_session <session-dir>

Checks:
1. manifest.json exists and parses
2. each declared stream CSV exists, has >1 row, header looks right
3. per-stream stats: row count, time range, avg Hz, % gaps > 200ms
4. wall-clock overlap between streams
5. episode time ranges fall inside actual data
6. calibration profiles, if any, are loadable

Exit code 0 = pass, 1 = fail (with reasons), 2 = warnings only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable


def _read_t_wall(csv_path: Path, col: str) -> list[int]:
    out: list[int] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if col not in (rdr.fieldnames or []):
            return out
        for row in rdr:
            v = row.get(col)
            if v is None or v == "":
                continue
            try:
                out.append(int(v))
            except (ValueError, TypeError):
                continue
    return out


def _summarize_times(ts: list[int]) -> dict:
    if not ts:
        return {"count": 0, "ok": False}
    ts.sort()
    dts_ns = [b - a for a, b in zip(ts, ts[1:])]
    total_s = (ts[-1] - ts[0]) / 1e9
    rate = (len(ts) - 1) / total_s if total_s > 0 else 0.0
    big_gaps = sum(1 for d in dts_ns if d > 200_000_000)
    return {
        "count": len(ts),
        "first_ns": ts[0],
        "last_ns": ts[-1],
        "duration_s": total_s,
        "avg_hz": rate,
        "max_gap_ms": max(dts_ns) / 1e6 if dts_ns else 0,
        "gaps_over_200ms": big_gaps,
        "ok": True,
    }


def _overlap(a: dict, b: dict) -> dict:
    if not a.get("ok") or not b.get("ok"):
        return {"ok": False}
    lo = max(a["first_ns"], b["first_ns"])
    hi = min(a["last_ns"], b["last_ns"])
    return {"ok": hi > lo, "overlap_s": max(0, (hi - lo)) / 1e9}


def validate(session_dir: Path) -> tuple[int, list[str]]:
    msgs: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    def err(m): errors.append(m); msgs.append(f"  FAIL  {m}")
    def warn(m): warnings.append(m); msgs.append(f"  WARN  {m}")
    def ok(m): msgs.append(f"  ok    {m}")

    msgs.append(f"Session: {session_dir}")

    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        err("manifest.json missing")
        return 1, msgs
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        err(f"manifest.json unreadable: {e}")
        return 1, msgs
    ok(f"manifest.json — session_id={manifest.get('session_id')}")

    # Streams that are nice-to-have but not required (e.g. depends on
    # Manus Core configuration or glove tier).
    OPTIONAL_STREAMS = {"manus_raw_devices"}

    streams = manifest.get("streams", {})
    stream_stats: dict[str, dict] = {}
    for name, rel in streams.items():
        path = session_dir / rel
        if not path.exists():
            err(f"missing stream file: {rel}")
            continue
        col = "t_wall_ns" if name.startswith("manus_") else "wall_time_ns"
        ts = _read_t_wall(path, col)
        stats = _summarize_times(ts)
        stream_stats[name] = stats
        if not stats["ok"]:
            if name in OPTIONAL_STREAMS:
                warn(f"optional stream {name}: empty (Manus Core may not "
                     f"publish this; non-fatal)")
            else:
                err(f"stream {name}: no parseable {col} rows in {rel}")
            continue
        if stats["count"] < 30:
            warn(f"stream {name}: only {stats['count']} rows")
        if stats["avg_hz"] < 5:
            warn(f"stream {name}: low rate {stats['avg_hz']:.1f} Hz")
        if stats["gaps_over_200ms"] > 0:
            warn(f"stream {name}: {stats['gaps_over_200ms']} gaps > 200ms "
                 f"(max {stats['max_gap_ms']:.0f}ms)")
        ok(f"stream {name}: n={stats['count']} "
           f"dur={stats['duration_s']:.1f}s rate={stats['avg_hz']:.1f}Hz")

    # Cross-stream overlap. Skip pairs that include an empty optional stream.
    keys = [k for k in stream_stats if stream_stats[k].get("ok")
            or k not in OPTIONAL_STREAMS]
    keys = [k for k in keys if stream_stats[k].get("ok")]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ov = _overlap(stream_stats[a], stream_stats[b])
            if not ov["ok"]:
                err(f"streams {a} and {b} have no wall-time overlap")
            elif ov["overlap_s"] < 1.0:
                warn(f"streams {a} & {b} overlap only "
                     f"{ov['overlap_s']:.2f}s")
            else:
                ok(f"streams {a} & {b} overlap "
                   f"{ov['overlap_s']:.1f}s")

    # Episodes
    ep_path = session_dir / "episodes.jsonl"
    if ep_path.exists():
        episodes = [json.loads(l) for l in ep_path.read_text(
            encoding="utf-8").splitlines() if l.strip()]
        ok(f"episodes.jsonl: {len(episodes)} episodes")
        for ep in episodes:
            for name, stats in stream_stats.items():
                if not stats.get("ok"):
                    continue
                if (ep["start_wall_ns"] < stats["first_ns"]
                        or ep["end_wall_ns"] > stats["last_ns"]):
                    warn(f"episode {ep['idx']} not fully covered by {name}")
    else:
        msgs.append("  ---   no episodes.jsonl (continuous session)")

    # Calibration
    cal_dir = session_dir / "calibration"
    if cal_dir.exists():
        for f in cal_dir.glob("*.json"):
            try:
                cal = json.loads(f.read_text(encoding="utf-8"))
                rmsd = cal.get("phase1_rmsd_deg", "?")
                ok(f"calibration {f.name}: rot residual {rmsd}°")
            except Exception as e:
                warn(f"calibration {f.name}: parse failed ({e})")

    # Summary
    if errors:
        rc = 1
        msgs.append(f"\nResult: FAIL  ({len(errors)} errors, "
                    f"{len(warnings)} warnings)")
    elif warnings:
        rc = 2
        msgs.append(f"\nResult: warnings  ({len(warnings)} warnings)")
    else:
        rc = 0
        msgs.append("\nResult: PASS")
    return rc, msgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path)
    args = ap.parse_args()
    rc, msgs = validate(args.session_dir)
    for m in msgs:
        print(m)
    return rc


if __name__ == "__main__":
    sys.exit(main())
