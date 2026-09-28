#!/usr/bin/env python3
"""
fix_episode_indices.py — Re-sequence episode indices in a LeRobot v3 dataset
so they are contiguous 0…N-1.

Use this after delete_episodes.py has already removed episode rows but
left gaps in the episode_index column.

Videos are NOT modified.  The from_timestamp / to_timestamp in
meta/episodes already point to correct positions in the unchanged .mp4
chunks, so they stay valid after re-indexing.

Usage (inside Docker container):
    python3 /workspace/scripts/fix_episode_indices.py \\
        --dataset /workspace/data/peg_in_hole [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def _read_chunk_dir(d: Path) -> pd.DataFrame:
    files = sorted(d.glob("file-*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _write_chunk_dir(df: pd.DataFrame, d: Path, n: int) -> None:
    for f in sorted(d.glob("file-*.parquet")):
        f.unlink()
    if df.empty or n == 0:
        return
    size = max(1, (len(df) + n - 1) // n)
    for i in range(n):
        chunk = df.iloc[i * size: (i + 1) * size].reset_index(drop=True)
        if not chunk.empty:
            pq.write_table(pa.Table.from_pandas(chunk, preserve_index=False),
                           d / f"file-{i:03d}.parquet")


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _const_stats(prefix: str, val: float, count: int) -> dict:
    return {f"{prefix}/{k}": [v] for k, v in {
        "min": val, "max": val, "mean": val, "std": 0.0, "count": count,
        "q01": val, "q10": val, "q50": val, "q90": val, "q99": val,
    }.items()}


def _range_stats(prefix: str, lo: int, hi: int) -> dict:
    vals = np.arange(lo, hi, dtype=np.float64)
    pcts = np.percentile(vals, [1, 10, 50, 90, 99])
    return {f"{prefix}/{k}": [v] for k, v in {
        "min": float(vals.min()), "max": float(vals.max()),
        "mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
        "count": len(vals),
        "q01": float(pcts[0]), "q10": float(pcts[1]), "q50": float(pcts[2]),
        "q90": float(pcts[3]), "q99": float(pcts[4]),
    }.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    if not root.exists():
        sys.exit(f"[ERROR] Dataset not found: {root}")

    info_path   = root / "meta" / "info.json"
    data_dir    = root / "data"  / "chunk-000"
    meta_ep_dir = root / "meta"  / "episodes" / "chunk-000"
    align_path  = root / "meta"  / "timestamp_alignment.json"

    with open(info_path) as f:
        info = json.load(f)

    df_data = _read_chunk_dir(data_dir)
    df_eps  = _read_chunk_dir(meta_ep_dir)
    n_data_files = len(sorted(data_dir.glob("file-*.parquet")))
    n_ep_files   = len(sorted(meta_ep_dir.glob("file-*.parquet")))

    existing = sorted(df_eps["episode_index"].unique().tolist())
    n        = len(existing)

    old_to_new = {old: new for new, old in enumerate(existing)}
    gaps       = [i for i in range(existing[-1] + 1) if i not in set(existing)]

    print(f"\nDataset  : {root}")
    print(f"Episodes : {len(existing)} present, gaps at {gaps}")
    print(f"Mapping  : {dict(list(old_to_new.items())[:5])} … (first 5 shown)")
    print(f"Frames   : {len(df_data)}")
    print(f"Videos   : NOT modified\n")

    vid_fi = "videos/observation.images.front/file_index"
    vid_t0 = "videos/observation.images.front/from_timestamp"
    vid_t1 = "videos/observation.images.front/to_timestamp"

    print(f"{'Old':>4}  {'New':>4}  {'Len':>6}  {'Video':>10}  {'Start':>8}  {'End':>8}")
    print("─" * 58)
    for _, row in df_eps.sort_values("episode_index").iterrows():
        ep  = int(row["episode_index"])
        new = old_to_new[ep]
        fi  = int(row.get(vid_fi, 0))
        t0  = float(row.get(vid_t0, 0))
        t1  = float(row.get(vid_t1, 0))
        arrow = " ←" if ep != new else ""
        print(f"{ep:>4}  {new:>4}  {int(row['length']):>6}  "
              f"file-{fi:03d}.mp4  {t0:>8.2f}  {t1:>8.2f}{arrow}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    print()
    if input("Proceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    # ── 1. data parquet ──────────────────────────────────────────────────
    print("Rewriting data parquet …", end=" ", flush=True)
    df_d = df_data.copy()
    df_d["episode_index"] = df_d["episode_index"].map(old_to_new)
    df_d = df_d.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    df_d["index"] = df_d.index
    _write_chunk_dir(df_d, data_dir, n_data_files)
    print("done.")

    # ── 2. meta/episodes parquet ─────────────────────────────────────────
    print("Rewriting meta/episodes parquet …", end=" ", flush=True)
    df_e = df_eps.copy()
    cumulative = 0
    new_rows = []
    for _, row in df_e.sort_values("episode_index").iterrows():
        old_ep = int(row["episode_index"])
        new_ep = old_to_new[old_ep]
        length = int(row["length"])
        r = row.to_dict()

        r["episode_index"]      = new_ep
        r["dataset_from_index"] = cumulative
        r["dataset_to_index"]   = cumulative + length
        cumulative += length

        r.update(_const_stats("stats/episode_index", float(new_ep), length))
        r.update(_range_stats("stats/index",
                              r["dataset_from_index"],
                              r["dataset_to_index"]))
        new_rows.append(r)

    df_e_new = pd.DataFrame(new_rows).reset_index(drop=True)
    _write_chunk_dir(df_e_new, meta_ep_dir, n_ep_files)
    print("done.")

    # ── 3. info.json ─────────────────────────────────────────────────────
    print("Updating meta/info.json …", end=" ", flush=True)
    info["total_episodes"] = n
    info["total_frames"]   = len(df_data)
    if "num_episodes" in info:
        info["num_episodes"] = n
    if "splits" in info:
        info["splits"]["train"] = f"0:{n}"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print("done.")

    # ── 4. timestamp_alignment.json ──────────────────────────────────────
    if align_path.exists():
        print("Updating timestamp_alignment.json …", end=" ", flush=True)
        with open(align_path) as f:
            align = json.load(f)
        align_new = sorted(
            [dict(r, episode_index=old_to_new[r["episode_index"]])
             for r in align if r["episode_index"] in old_to_new],
            key=lambda r: r["episode_index"],
        )
        with open(align_path, "w") as f:
            json.dump(align_new, f, indent=2)
        print("done.")

    print(f"""
Done.
  Episodes : {n}, re-indexed 0 … {n-1}
  Frames   : {len(df_data)}
  Videos   : unchanged

Next steps:
  1. Set data_collection.yaml → total_episodes: {n + 2}
  2. ./scripts/launch_data_collection.sh
     Recorder will append episodes {n} and {n+1}.
""")


if __name__ == "__main__":
    main()
