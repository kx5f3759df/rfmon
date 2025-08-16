#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate scanner CSVs into continuous-signal segments with frequency merging.

Rules:
- Frequencies within ±0.002 MHz are considered the same; merge and use the
  average frequency (rounded to 0.001 MHz) as output.
- Two detections whose time gap is < 5 minutes belong to the same continuous signal.
- Duty uses the "duty" column (single-bin). If only duty in % under a different
  name is present (e.g., duty_center_pct / duty_pct), it will be used.
- Output a single CSV with columns:
  start_utc, end_utc, freq, samples, pwr_avg, pwr_sd, duty_avg, duty_sd

Usage:
  python aggregate_signals.py [--dir DIR] [--out OUT.csv]
"""

import os
import glob
import argparse
import math
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any

import pandas as pd
import numpy as np


FREQ_TOL = 0.002       # MHz, merge frequencies within ±0.002
CONTIG_GAP = 5 * 60    # seconds, 5 minutes


def parse_time(s: str) -> datetime:
    # Expect ISO like "2025-08-01T12:34:56Z"
    # Fall back to pandas if needed.
    try:
        if s.endswith('Z'):
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        # try with pandas
        return pd.to_datetime(s, utc=True).to_pydatetime()
    except Exception:
        return pd.to_datetime(s, utc=True).to_pydatetime()


def load_rows_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names: strip spaces and lower for detection, but keep originals for values.
    cols = {c.lower().strip(): c for c in df.columns}

    # Required baseline columns (with flexible names for duty)
    time_col = None
    for cand in ["utc_iso", "time_utc", "timestamp", "time"]:
        if cand in cols:
            time_col = cols[cand]; break
    if time_col is None:
        raise ValueError(f"{os.path.basename(path)}: missing time column (utc_iso/time_utc/timestamp/time)")

    freq_col = None
    for cand in ["center_mhz", "freq_mhz", "frequency_mhz", "frequency"]:
        if cand in cols:
            freq_col = cols[cand]; break
    if freq_col is None:
        raise ValueError(f"{os.path.basename(path)}: missing frequency column (center_mhz/freq_mhz/...)")

    p_col = None
    for cand in ["p_dbfs", "power_db", "p_db", "power"]:
        if cand in cols:
            p_col = cols[cand]; break
    if p_col is None:
        raise ValueError(f"{os.path.basename(path)}: missing power column (p_dbfs/p_db/power)")

    # Duty: prefer 'duty' (single-bin). Fallbacks allowed.
    duty_col = None
    for cand in ["duty", "duty_center_pct", "duty_pct", "duty_wide", "duty_wide_pct"]:
        if cand in cols:
            duty_col = cols[cand]; break
    if duty_col is None:
        # If absent, create NaN to allow concatenation but will drop later.
        df["__duty__"] = np.nan
        duty_col = "__duty__"

    # Build normalized DataFrame
    out = pd.DataFrame({
        "utc": [parse_time(x) for x in df[time_col].astype(str).tolist()],
        "freq_mhz": pd.to_numeric(df[freq_col], errors="coerce"),
        "p_dbfs": pd.to_numeric(df[p_col], errors="coerce"),
        "duty": pd.to_numeric(df[duty_col], errors="coerce"),
    })
    # Remove rows missing essentials
    out = out.dropna(subset=["utc", "freq_mhz", "p_dbfs"])
    return out


def load_all_csvs(indir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(indir, "*.csv")))
    frames: List[pd.DataFrame] = []
    for p in paths:
        try:
            df = load_rows_from_csv(p)
            frames.append(df)
        except Exception as e:
            # Skip files that don't match expected schema
            print(f"[w] skip {os.path.basename(p)}: {e}")
    if not frames:
        return pd.DataFrame(columns=["utc", "freq_mhz", "p_dbfs", "duty"])
    all_df = pd.concat(frames, ignore_index=True)
    return all_df.sort_values(["freq_mhz", "utc"]).reset_index(drop=True)


def cluster_frequencies_greedy(freqs: np.ndarray) -> np.ndarray:
    """
    Given sorted freqs (MHz), assign cluster ids so that each new freq joins the current
    cluster if it's within ±FREQ_TOL of the cluster's running mean. Otherwise start a new cluster.
    Returns cluster ids array aligned with freqs.
    """
    if len(freqs) == 0:
        return np.array([], dtype=int)
    ids = np.zeros(len(freqs), dtype=int)
    current_id = 0
    mean = freqs[0]
    count = 1
    for i in range(1, len(freqs)):
        f = freqs[i]
        if abs(f - mean) <= FREQ_TOL:
            # join
            mean = (mean * count + f) / (count + 1)
            count += 1
            ids[i] = current_id
        else:
            # new cluster
            current_id += 1
            ids[i] = current_id
            mean = f
            count = 1
    return ids


def split_continuous_segments(times: List[datetime]) -> List[slice]:
    """
    Given sorted timestamps for one frequency group, return a list of slices
    delineating continuous segments where gaps < CONTIG_GAP seconds.
    """
    if not times:
        return []
    idx_slices: List[slice] = []
    start = 0
    for i in range(1, len(times)):
        gap = (times[i] - times[i-1]).total_seconds()
        if gap >= CONTIG_GAP:
            idx_slices.append(slice(start, i))
            start = i
    idx_slices.append(slice(start, len(times)))
    return idx_slices


def aggregate(indir: str, out_path: str) -> None:
    df = load_all_csvs(indir)
    if df.empty:
        # write empty CSV with header
        cols = ["start_utc","end_utc","freq","samples","pwr_avg","pwr_sd","duty_avg","duty_sd"]
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
        print(f"[i] no valid rows found. Wrote empty file: {out_path}")
        return

    # Sort by frequency then time
    df = df.sort_values(["freq_mhz", "utc"]).reset_index(drop=True)

    # Cluster by frequency within ±0.002 using greedy running-mean criterion
    freqs = df["freq_mhz"].to_numpy()
    cluster_ids = cluster_frequencies_greedy(freqs)
    df["freq_cluster"] = cluster_ids

    records: List[Dict[str, Any]] = []

    # Iterate clusters
    for cid, g in df.groupby("freq_cluster", sort=True):
        g = g.sort_values("utc").reset_index(drop=True)
        times = g["utc"].tolist()

        # Continuous segments based on time gaps < 5 minutes
        seg_slices = split_continuous_segments(times)
        for sl in seg_slices:
            seg = g.iloc[sl]
            if seg.empty:
                continue

            # Frequency: average over segment, then round to 0.001 MHz
            f_avg = float(seg["freq_mhz"].mean())
            f_out = round(f_avg, 3)

            # Strength stats
            p_mean = float(seg["p_dbfs"].mean())
            p_std  = float(seg["p_dbfs"].std(ddof=0))  # population std

            # Duty stats (percent). If 'duty' is all NaN, skip duty stats.
            if seg["duty"].notna().any():
                d_mean = float(seg["duty"].mean())
                d_std  = float(seg["duty"].std(ddof=0))
            else:
                d_mean = float("nan")
                d_std  = float("nan")

            t_start = seg["utc"].iloc[0].astimezone(timezone.utc)
            t_end   = seg["utc"].iloc[-1].astimezone(timezone.utc)

            records.append({
                "start_utc": t_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_utc": t_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "freq": f_out,
                "samples": int(len(seg)),
                "pwr_avg": round(p_mean, 3),
                "pwr_sd": round(p_std, 3),
                "duty_avg": None if math.isnan(d_mean) else round(d_mean, 2),
                "duty_sd": None if math.isnan(d_std) else round(d_std, 2),
            })

    out_df = pd.DataFrame.from_records(records, columns=[
        "start_utc","end_utc","freq","samples","pwr_avg","pwr_sd","duty_avg","duty_sd"
    ])

    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[i] wrote: {out_path} ({len(out_df)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Aggregate scanner CSVs by frequency and continuity.")
    ap.add_argument("--dir", default=".", help="Input directory containing CSV files (default: current directory)")
    ap.add_argument("--out", default="aggregated_signals.csv", help="Output CSV filename (default: aggregated_signals.csv)")
    args = ap.parse_args()

    aggregate(args.dir, args.out)


if __name__ == "__main__":
    main()
