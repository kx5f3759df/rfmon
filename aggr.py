#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggr.py — RF aggregation for sliding-window scans (non-uniform coverage).

Key properties:
- Does NOT assume every scan covers the full band or that frames are regular.
- Builds a time–frequency connectivity graph across ALL observations:
  • Two points link if their time gap ≤ link_time_max AND their frequency
    difference ≤ link_freq_base_mhz + link_freq_per_sec_mhz * Δt_seconds.
  • Points with the SAME UTC timestamp (same FFT) are NEVER linked.
- Connected components are interpreted as "sources" (tracks). Within each
  component, observations are ordered by time and cut into segments where the
  time gap exceeds max_gap_min.
- Optional component-level canonical snapping stabilizes segment centers.
- Global frequency clustering uses 1D complete-link with a PHYSICAL tolerance
  (MHz), independent of any histogram/bin grid. This guarantees that very close
  centers (e.g., 449.9969 & 449.9973) merge whenever fuse_tol_mhz ≥ 0.0004.
- FINAL rounding only at export using Decimal half-up respecting --round_freq.

Output CSV columns:
  start_utc, end_utc, freq, samples, pwr_avg, pwr_sd, duty_avg, duty_sd
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------
def parse_utc(s: str) -> datetime:
    """Parse various UTC string formats to a timezone-aware UTC datetime."""
    s = str(s).strip()
    fmts = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S.%f",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    dt = pd.to_datetime(s, utc=True).to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def round_to_step(x: float, step: float) -> float:
    """Round x to nearest multiple of 'step' using Decimal half-up; preserve step decimals."""
    from decimal import Decimal, ROUND_HALF_UP, getcontext
    getcontext().prec = 20
    d = Decimal(str(x))
    s = Decimal(str(step))
    q = (d / s).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * s
    decimals = max(0, -s.as_tuple().exponent)
    quant = Decimal('1').scaleb(-decimals)
    q = q.quantize(quant, rounding=ROUND_HALF_UP)
    return float(q)


def nanmean(a) -> float:
    arr = np.asarray(a, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def nansd(a) -> float:
    arr = np.asarray(a, dtype=float)
    return float(np.nanstd(arr, ddof=0)) if arr.size else float("nan")


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median for stable cluster centers."""
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    idx = np.searchsorted(cum, cutoff)
    idx = min(idx, len(v) - 1)
    return float(v[idx])


# -------------------------
# Data loading / normalization
# -------------------------
def load_rows_from_csv(path: str) -> pd.DataFrame:
    """Normalize CSV to columns: utc, freq_mhz, p_dbfs, duty."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    # Time column
    time_col = None
    for cand in ["utc_iso", "time_utc", "timestamp", "time", "utc"]:
        if cand in cols:
            time_col = cols[cand]
            break
    if time_col is None:
        raise ValueError(f"{os.path.basename(path)}: missing time column")

    # Frequency column
    freq_col = None
    for cand in ["center_mhz", "freq_mhz", "frequency_mhz", "frequency", "freq"]:
        if cand in cols:
            freq_col = cols[cand]
            break
    if freq_col is None:
        raise ValueError(f"{os.path.basename(path)}: missing frequency column")

    # Power column
    p_col = None
    for cand in ["p_dbfs", "power_db", "p_db", "power"]:
        if cand in cols:
            p_col = cols[cand]
            break
    if p_col is None:
        raise ValueError(f"{os.path.basename(path)}: missing power column")

    # Duty column (optional)
    duty_col = None
    for cand in ["duty", "duty_center_pct", "duty_pct", "duty_wide", "duty_wide_pct"]:
        if cand in cols:
            duty_col = cols[cand]
            break

    out = pd.DataFrame(
        {
            "utc": [parse_utc(x) for x in df[time_col].astype(str).tolist()],
            "freq_mhz": pd.to_numeric(df[freq_col], errors="coerce"),
            "p_dbfs": pd.to_numeric(df[p_col], errors="coerce"),
        }
    )
    if duty_col is not None:
        duty_vals = pd.to_numeric(df[duty_col], errors="coerce")
        if duty_vals.max(skipna=True) is not None and duty_vals.max(skipna=True) > 1.5:
            duty_vals = duty_vals / 100.0
        out["duty"] = duty_vals
    else:
        out["duty"] = np.nan

    out = out.dropna(subset=["utc", "freq_mhz"]).reset_index(drop=True)
    return out[["utc", "freq_mhz", "p_dbfs", "duty"]]


def load_all_rows(directory: str) -> pd.DataFrame:
    """Recursively load all CSVs under directory and concatenate them."""
    paths: List[str] = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".csv"):
                paths.append(os.path.join(root, f))
    frames: List[pd.DataFrame] = []
    for p in sorted(paths):
        try:
            frames.append(load_rows_from_csv(p))
        except Exception as e:
            print(f"[w] skip {os.path.basename(p)}: {e}")
    if not frames:
        return pd.DataFrame(columns=["utc", "freq_mhz", "p_dbfs", "duty"])
    all_df = pd.concat(frames, ignore_index=True)
    return all_df.sort_values(["utc", "freq_mhz"]).reset_index(drop=True)


# -------------------------
# Connectivity graph (DSU in sorted-time space)
# -------------------------
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def build_timefreq_components(df: pd.DataFrame,
                              link_time_max: timedelta,
                              link_freq_base_mhz: float,
                              link_freq_per_sec_mhz: float) -> List[List[int]]:
    """
    Build connectivity by sweeping in time: for each point, only compare to
    following points whose Δt ≤ link_time_max. Never link points with identical UTC.
    Two points i<j are linked if:
        Δt = (utc_j - utc_i) ≤ link_time_max
        Δf = |f_j - f_i| ≤ link_freq_base_mhz + link_freq_per_sec_mhz * Δt_seconds
    Returns components as lists of ORIGINAL DataFrame indices.
    """
    n = len(df)
    if n == 0:
        return []

    # Sort by time; we do DSU in this sorted index space [0..n-1].
    order = np.argsort(df["utc"].values)
    t_sorted = df["utc"].values[order]         # datetime64[ns, UTC]
    f_sorted = df["freq_mhz"].values[order]    # float
    dsu = DSU(n)

    # Convert link_time_max (datetime.timedelta) to numpy timedelta64 for safe comparison
    max_dt_np = np.timedelta64(int(link_time_max.total_seconds()), 's')

    j = 0
    for i in range(n):
        # advance j to maintain Δt <= link_time_max
        while j < n and (t_sorted[j] - t_sorted[i]) <= max_dt_np:
            j += 1
        # compare i with (i+1...j-1)
        for k in range(i + 1, j):
            # Never link points with identical UTC (same FFT)
            if t_sorted[k] == t_sorted[i]:
                continue
            dt_sec = float((t_sorted[k] - t_sorted[i]) / np.timedelta64(1, "s"))
            dfreq = abs(float(f_sorted[k]) - float(f_sorted[i]))
            tol = link_freq_base_mhz + link_freq_per_sec_mhz * dt_sec
            if dfreq <= tol + 1e-12:
                dsu.union(i, k)

    # Gather components in sorted space, then map back to original indices.
    comp_dict: Dict[int, List[int]] = {}
    for pos in range(n):
        root = dsu.find(pos)
        comp_dict.setdefault(root, []).append(pos)

    components: List[List[int]] = []
    for _, pos_list in comp_dict.items():
        orig_indices = order[np.array(pos_list, dtype=int)].tolist()
        orig_indices.sort(key=lambda idx: df.loc[idx, "utc"])
        components.append(orig_indices)
    return components


# -------------------------
# Segmentation inside components
# -------------------------
df_global: pd.DataFrame = pd.DataFrame()  # populated in aggregate_df()


@dataclass
class Segment:
    idxs: List[int]   # indices into original DataFrame

    @property
    def start(self) -> datetime:
        return df_global.loc[self.idxs[0], "utc"]

    @property
    def end(self) -> datetime:
        return df_global.loc[self.idxs[-1], "utc"]

    @property
    def samples(self) -> int:
        return len(self.idxs)

    def mean_freq(self) -> float:
        return float(np.mean(df_global.loc[self.idxs, "freq_mhz"].values))

    def pwr_stats(self) -> Tuple[float, float]:
        arr = df_global.loc[self.idxs, "p_dbfs"].values.astype(float)
        return nanmean(arr), nansd(arr)

    def duty_stats(self) -> Tuple[Optional[float], Optional[float]]:
        arr = df_global.loc[self.idxs, "duty"].values.astype(float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return None, None
        return nanmean(arr), nansd(arr)


def cut_segments_in_component(comp_idxs: List[int], max_gap: timedelta) -> List[Segment]:
    """Split a component into time-contiguous segments by max_gap."""
    if not comp_idxs:
        return []
    comp_idxs = sorted(comp_idxs, key=lambda i: df_global.loc[i, "utc"])
    segs: List[Segment] = []
    cur: List[int] = [comp_idxs[0]]
    for i in range(1, len(comp_idxs)):
        prev_t = df_global.loc[comp_idxs[i - 1], "utc"]
        now_t = df_global.loc[comp_idxs[i], "utc"]
        if (now_t - prev_t) <= max_gap:
            cur.append(comp_idxs[i])
        else:
            segs.append(Segment(idxs=cur))
            cur = [comp_idxs[i]]
    segs.append(Segment(idxs=cur))
    return segs


# -------------------------
# Canonical snapping (component-level)
# -------------------------
def component_canonical(comp_idxs: List[int], first_n: int) -> float:
    """Median of the first N observations in a component."""
    if not comp_idxs:
        return float("nan")
    comp_idxs = sorted(comp_idxs, key=lambda i: df_global.loc[i, "utc"])
    n = min(first_n, len(comp_idxs))
    vals = df_global.loc[comp_idxs[:n], "freq_mhz"].values.astype(float)
    return float(np.median(vals))


def apply_canonical_to_center(center: float, canonical: float, snap_tol: float) -> float:
    """Return stabilized center (no rounding)."""
    if not math.isnan(canonical) and abs(center - canonical) <= snap_tol:
        return canonical
    return center


# -------------------------
# Global complete-link clustering (physical tolerance)
# -------------------------
def cluster_centers_complete_link(centers: List[float], weights: List[float], fuse_tol_mhz: float) -> Tuple[List[float], List[int]]:
    """
    1D complete-link clustering by maximum cluster span (MHz).
    - Sort centers; start a cluster with first center; track cluster_min.
    - If next value v satisfies v - cluster_min ≤ fuse_tol_mhz, keep in cluster.
      Otherwise, close cluster and start a new one.
    - Cluster representative = weighted median of member centers.
    Returns (cluster_reps, assignment_index_per_center).
    """
    if not centers:
        return [], []

    v = np.asarray(centers, float)
    w = np.asarray(weights, float)
    order = np.argsort(v)
    v_sorted = v[order]; w_sorted = w[order]

    clusters: List[Tuple[int, int]] = []  # (start_idx, end_idx_exclusive) in sorted space
    start = 0
    cluster_min = v_sorted[0]
    for i in range(1, len(v_sorted)):
        if (v_sorted[i] - cluster_min) <= fuse_tol_mhz + 1e-12:
            continue
        else:
            clusters.append((start, i))
            start = i
            cluster_min = v_sorted[i]
    clusters.append((start, len(v_sorted)))

    reps: List[float] = []
    assign_sorted = np.full(len(v_sorted), -1, dtype=int)
    for cid, (a, b) in enumerate(clusters):
        rep = weighted_median(v_sorted[a:b], w_sorted[a:b])
        reps.append(rep)
        assign_sorted[a:b] = cid

    # Map back to original order
    assign = np.full(len(v), -1, dtype=int)
    assign[order] = assign_sorted
    return reps, assign.tolist()


# -------------------------
# Pipeline
# -------------------------
def aggregate_df(df: pd.DataFrame, args) -> pd.DataFrame:
    global df_global
    df_global = df

    print(f"[dbg] input points: {len(df)}, unique UTC frames: {df['utc'].nunique()}")

    # 1) Connectivity components (time-frequency graph, no frame assumption)
    comps = build_timefreq_components(
        df=df,
        link_time_max=timedelta(minutes=float(args.link_time_max_min)),
        link_freq_base_mhz=float(args.link_freq_base_mhz),
        link_freq_per_sec_mhz=float(args.link_freq_per_sec_mhz),
    )
    print(f"[dbg] components: {len(comps)}")

    # 2) Cut segments inside each component; compute (optional) canonical
    all_segments: List[Segment] = []
    seg_centers: List[float] = []
    seg_weights: List[int] = []
    rows: List[Dict[str, object]] = []

    for comp in comps:
        segs = cut_segments_in_component(comp, max_gap=timedelta(minutes=float(args.max_gap_min)))
        if not segs:
            continue
        all_segments.extend(segs)

        canonical = component_canonical(comp, first_n=int(args.canon_use_first_n)) if int(args.snap_to_canonical) == 1 else float("nan")

        for seg in segs:
            m = seg.mean_freq()
            m_stab = apply_canonical_to_center(m, canonical, snap_tol=float(args.snap_tol_mhz)) if int(args.snap_to_canonical) == 1 else m

            pavg, psd = seg.pwr_stats()
            davg, dsd = seg.duty_stats()

            seg_centers.append(m_stab)
            seg_weights.append(seg.samples)

            rows.append(dict(
                start_utc=seg.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                end_utc=seg.end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                freq_float=m_stab,         # not rounded yet
                samples=int(seg.samples),
                pwr_avg=round(pavg, 3),
                pwr_sd=round(psd, 3),
                duty_avg=(None if davg is None else round(davg, 3)),
                duty_sd=(None if dsd is None else round(dsd, 3)),
            ))

    print(f"[dbg] segments(before global cluster): {len(seg_centers)}")
    df_rows = pd.DataFrame.from_records(rows)
    if df_rows.empty:
        return df_rows

    # 3) Stage-2 global complete-link clustering (physical tolerance)
    reps, assign = cluster_centers_complete_link(seg_centers, seg_weights, fuse_tol_mhz=float(args.fuse_tol_mhz))
    snapped = [reps[k] if (0 <= k < len(reps)) else c for c, k in zip(seg_centers, assign)]
    df_rows["freq_float"] = snapped

    return df_rows


# -------------------------
# Finalize & write
# -------------------------
def finalize_and_write(df_rows: pd.DataFrame, out_path: str, round_step: float) -> pd.DataFrame:
    """Final rounding (only here) and write CSV."""
    if df_rows.empty:
        out_df = pd.DataFrame(columns=["start_utc", "end_utc", "freq", "samples", "pwr_avg", "pwr_sd", "duty_avg", "duty_sd"])
        out_df.to_csv(out_path, index=False)
        print(f"[i] no data. wrote empty CSV: {out_path}")
        print(f"[i] unique frequencies: 0")
        return out_df

    df_rows = df_rows.copy()
    df_rows["freq"] = [round_to_step(x, round_step) for x in df_rows["freq_float"].astype(float)]

    out_df = df_rows[["start_utc", "end_utc", "freq", "samples", "pwr_avg", "pwr_sd", "duty_avg", "duty_sd"]].copy()
    out_df = out_df.sort_values(["freq", "start_utc"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)

    print(f"[i] wrote: {out_path} ({len(out_df)} rows)")
    print(f"[i] unique frequencies: {out_df['freq'].nunique()}")
    return out_df


# -------------------------
# Main entry
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="RF aggregator for sliding-window scans using time–frequency connectivity + complete-link clustering.")
    # Inputs
    ap.add_argument("--dir", default="./", help="Directory to recursively read CSVs from (default: ./). Ignored if --file is provided.")
    ap.add_argument("--file", default=None, help="Read a single CSV file (overrides --dir).")
    ap.add_argument("--out", default="aggregated_signals.csv", help="Output CSV filename (default: aggregated_signals.csv)")

    # Connectivity (graph) parameters
    ap.add_argument("--link_time_max_min", type=float, default=30.0, help="Max time gap (minutes) to allow linking across observations (default: 30).")
    ap.add_argument("--link_freq_base_mhz", type=float, default=0.0012, help="Base frequency tolerance (MHz) for linking (default: 0.0012).")
    ap.add_argument("--link_freq_per_sec_mhz", type=float, default=0.0000002, help="Extra tolerance per second (MHz/s) with time gap for linking (default: 2e-7 MHz/s = 0.2 Hz/s).")

    # Segmentation
    ap.add_argument("--max_gap_min", type=float, default=10.0, help="Segment break if time gap exceeds this many minutes (default: 10).")

    # Canonical snapping
    ap.add_argument("--snap_to_canonical", type=int, default=1, help="1 to enable per-component canonical snap (default: 1).")
    ap.add_argument("--snap_tol_mhz", type=float, default=0.0015, help="Tolerance (MHz) to snap a segment to its component canonical.")
    ap.add_argument("--canon_use_first_n", type=int, default=5, help="Use first N observations of a component to compute canonical.")

    # Global clustering
    ap.add_argument("--fuse_tol_mhz", type=float, default=0.0008, help="Global complete-link max span per cluster in MHz (default: 0.0008).")

    # Final rounding & stats
    ap.add_argument("--round_freq", type=float, default=0.001, help="Final rounding step in MHz (Decimal half-up; default: 0.001).")
    ap.add_argument("--stats_out", default=None, help="Optional path to write JSON with {'rows','unique_freqs'}.")

    args = ap.parse_args()

    # Load data
    if args.file:
        df = load_rows_from_csv(args.file)
    else:
        df = load_all_rows(args.dir)

    if df.empty:
        out_df = pd.DataFrame(columns=["start_utc", "end_utc", "freq", "samples", "pwr_avg", "pwr_sd", "duty_avg", "duty_sd"])
        out_df.to_csv(args.out, index=False)
        print(f"[i] no data. wrote empty CSV: {args.out}")
        print(f"[i] unique frequencies: 0")
        return

    # Aggregate
    df_rows = aggregate_df(df, args)

    # Finalize & write
    out_df = finalize_and_write(df_rows, out_path=args.out, round_step=float(args.round_freq))

    # Stats out
    if args.stats_out:
        stats = dict(rows=int(len(out_df)), unique_freqs=int(out_df["freq"].nunique()))
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[i] wrote stats: {args.stats_out}")


if __name__ == "__main__":
    main()
