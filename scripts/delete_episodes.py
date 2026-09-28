#!/usr/bin/env python3
"""
delete_episodes.py — Remove bad episodes from a LeRobot v3 dataset.

LeRobot v3 requires dense episode indices 0..N-1.  After deletion we must
re-sequence the surviving episode indices so they are contiguous again.

The video files (*.mp4) are NOT touched.  Each episode's
  videos/.../from_timestamp  and  to_timestamp
already encode the correct byte-offset positions inside the original video
chunks, so they remain valid after pure parquet re-indexing.

What this script changes:
  data/chunk-000/file-*.parquet
    • episode_index  — re-mapped to 0…N-1
    • index          — re-sequenced as global frame counter 0…F-1

  meta/episodes/chunk-000/file-*.parquet
    • episode_index              — re-mapped
    • dataset_from_index         — cumulative frame start
    • dataset_to_index           — cumulative frame end
    • stats/episode_index/*      — updated (constant per episode)
    • stats/index/*              — updated (range per episode)

  meta/info.json
    • total_episodes, total_frames

  meta/timestamp_alignment.json
    • episode_index entries re-mapped

Videos: UNTOUCHED.

Usage (inside Docker container):
    python3 /workspace/scripts/delete_episodes.py \\
        --dataset /workspace/data/peg_in_hole \\
        --episodes 17 25 [--dry-run]
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
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


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
# Stats helpers  (episode-level stats stored in meta/episodes)
# ---------------------------------------------------------------------------

def _const_stats(prefix: str, val: float, count: int) -> dict:
    """Stats for a column that is constant = val across all `count` frames."""
    return {f"{prefix}/{k}": [v] for k, v in {
        "min": val, "max": val, "mean": val, "std": 0.0, "count": count,
        "q01": val, "q10": val, "q50": val, "q90": val, "q99": val,
    }.items()}


def _range_stats(prefix: str, lo: int, hi: int) -> dict:
    """Stats for integer range [lo, hi)  (hi exclusive)."""
    vals = np.arange(lo, hi, dtype=np.float64)
    pcts = np.percentile(vals, [1, 10, 50, 90, 99])
    count = len(vals)
    return {f"{prefix}/{k}": [v] for k, v in {
        "min": float(vals.min()), "max": float(vals.max()),
        "mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
        "count": count,
        "q01": float(pcts[0]), "q10": float(pcts[1]), "q50": float(pcts[2]),
        "q90": float(pcts[3]), "q99": float(pcts[4]),
    }.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",  required=True)
    ap.add_argument("--episodes", nargs="+", type=int, required=True)
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    root      = Path(args.dataset).expanduser().resolve()
    to_delete = set(args.episodes)

    if not root.exists():
        sys.exit(f"[ERROR] Dataset not found: {root}")

    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        sys.exit("[ERROR] meta/info.json not found")

    with open(info_path) as f:
        info = json.load(f)

    data_dir    = root / "data"  / "chunk-000"
    meta_ep_dir = root / "meta"  / "episodes" / "chunk-000"
    align_path  = root / "meta"  / "timestamp_alignment.json"

    df_data = _read_chunk_dir(data_dir)
    df_eps  = _read_chunk_dir(meta_ep_dir)
    n_data_files = len(sorted(data_dir.glob("file-*.parquet")))
    n_ep_files   = len(sorted(meta_ep_dir.glob("file-*.parquet")))

    all_idx   = sorted(df_eps["episode_index"].tolist())
    surviving = [e for e in all_idx if e not in to_delete]
    missing   = [e for e in to_delete if e not in all_idx]
    if missing:
        sys.exit(f"[ERROR] Episodes not found: {missing}")

    old_to_new = {old: new for new, old in enumerate(surviving)}
    frames_before = len(df_data)
    frames_after  = int((~df_data["episode_index"].isin(to_delete)).sum())

    # ── preview ──────────────────────────────────────────────────────────
    vid_fi = "videos/observation.images.front/file_index"
    vid_t0 = "videos/observation.images.front/from_timestamp"
    vid_t1 = "videos/observation.images.front/to_timestamp"

    print(f"\nDataset  : {root}")
    print(f"Episodes : {len(all_idx)} total  |  deleting {sorted(to_delete)}")
    print(f"Frames   : {frames_before} → {frames_after}")
    print(f"Re-index : episodes re-sequenced 0 … {len(surviving)-1}")
    print(f"Videos   : NOT modified (timestamps kept as-is)\n")

    print(f"{'Old':>4}  {'New':>4}  {'Len':>6}  {'Video':>10}  {'Start':>8}  {'End':>8}  Status")
    print("─" * 65)
    for _, row in df_eps.sort_values("episode_index").iterrows():
        ep  = int(row["episode_index"])
        new = old_to_new.get(ep, -1)
        fi  = int(row.get(vid_fi, 0))
        t0  = float(row.get(vid_t0, 0))
        t1  = float(row.get(vid_t1, 0))
        tag = "DELETE" if ep in to_delete else f"→ {new}"
        print(f"{ep:>4}  {new if new >= 0 else '':>4}  {int(row['length']):>6}  "
              f"file-{fi:03d}.mp4  {t0:>8.2f}  {t1:>8.2f}  {tag}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    print()
    if input("Proceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    # ── 1. data parquet: filter + re-index episode_index + re-sequence global index ──
    print("Rewriting data parquet …", end=" ", flush=True)
    df_d = (df_data[~df_data["episode_index"].isin(to_delete)]
            .copy().reset_index(drop=True))
    df_d["episode_index"] = df_d["episode_index"].map(old_to_new)
    df_d["index"]         = df_d.index          # contiguous global frame index
    _write_chunk_dir(df_d, data_dir, n_data_files)
    print("done.")

    # ── 2. meta/episodes parquet: filter + re-index + fix frame ranges + fix stats ──
    print("Rewriting meta/episodes parquet …", end=" ", flush=True)
    df_e = df_eps[~df_eps["episode_index"].isin(to_delete)].copy()
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

        # video timestamps are intentionally kept unchanged (video not modified)

        # update stats that reference episode_index (constant within episode)
        r.update(_const_stats("stats/episode_index", float(new_ep), length))

        # update stats that reference global index (range within episode)
        r.update(_range_stats("stats/index",
                              r["dataset_from_index"],
                              r["dataset_to_index"]))

        new_rows.append(r)

    df_e_new = pd.DataFrame(new_rows).reset_index(drop=True)
    _write_chunk_dir(df_e_new, meta_ep_dir, n_ep_files)
    print("done.")

    # ── 3. info.json ─────────────────────────────────────────────────────
    print("Updating meta/info.json …", end=" ", flush=True)
    info["total_episodes"] = len(surviving)
    info["total_frames"]   = frames_after
    if "num_episodes" in info:
        info["num_episodes"] = len(surviving)
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
             for r in align if r["episode_index"] not in to_delete],
            key=lambda r: r["episode_index"],
        )
        with open(align_path, "w") as f:
            json.dump(align_new, f, indent=2)
        print("done.")

    n = len(surviving)
    print(f"""
Done.
  Deleted  : {sorted(to_delete)}
  Remaining: {n} episodes, re-indexed 0 … {n-1}
  Frames   : {frames_after}
  Videos   : unchanged (orphan segments in file-002.mp4 are harmless)

Next steps:
  1. data_collection.yaml → total_episodes: {n + 2}
  2. ./scripts/launch_data_collection.sh
     Recorder will append episodes {n} and {n+1}.
""")


if __name__ == "__main__":
    main()
