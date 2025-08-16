#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggr.py — Aggregate RF scan CSVs into continuous segments using temporal tracking,
with strict guarantees and robust frequency stabilization:

- Same FFT frame (same UTC) observations are NEVER merged.
- Temporal tracking across frames (greedy 1-1 with hysteresis and miss tolerance).
- Segmenting by time gap (default 10 minutes).
- Two-stage stabilization (NO rounding until the very end):
  1) Per-track canonical snapping (align segments back to each track canonical).
  2) Global-mode snapping built from a smoothed histogram of segment frequencies,
     with STRICT adjacent-peak merging: peaks within <= 1 bin are fused into a
     single mode (weighted centroid), to collapse 1 kHz bucket splits like
     449.996 / 449.997 into ONE final label when they truly represent one source.

Final CSV columns:
  start_utc, end_utc, freq, samples, pwr_avg, pwr_sd, duty_avg, duty_sd
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Defaults (conservative & data-driven)
# -------------------------
FREQ_TOL = 0.0012           # MHz, base tolerance for frame-to-frame matching
FREQ_TOL_PER_SEC = 0.0      # MHz/s, extra tolerance per second (if needed)
CONTINUE_TOL_MULT = 1.6     # continuation hysteresis multiplier
ROUND_FREQ = 0.001          # MHz, rounding step (APPLIED ONLY AT FINAL EXPORT)
MAX_GAP = timedelta(minutes=10)  # >10 minutes -> discontinuity (new segment)
MAX_MISSES = 2              # allow this many consecutive misses
MIN_SEG_SAMPLES = 1         # minimum samples per segment to keep
MIN_CONFIRM_SAMPLES = 2     # track becomes confirmed after this many points
VEL_ALPHA = 0.5             # EMA factor for df/dt

# Stage-1: per-track canonical snapping (no rounding)
SNAP_TO_TRACK_CANONICAL = True
SNAP_TOL_MHZ = 0.0015       # if segment mean within this of canonical -> snap
CANON_USE_FIRST_N = 5       # canonical = median of first N obs

# Stage-2: global-mode snapping (no rounding)
GLOBAL_SNAP_TO_MODES = True
GLOBAL_BIN_STEP = 0.001     # MHz (match final rounding step)
GLOBAL_SNAP_TOL_MHZ = 0.0015  # MHz, tolerance to snap to nearest global mode
GLOBAL_MODE_MIN_SEG = 1     # min segments in a bin to consider for peaks
GLOBAL_PEAK_WINDOW = 1      # +/- bins around a peak for weighted centroid
# STRICT: merge any two peaks whose centers differ by <= 1 bin
GLOBAL_STRICT_MERGE_BINS = 1


# -------------------------
# Utilities
# -------------------------
def parse_utc(s: str) -> datetime:
    """Parse a UTC timestamp string into a timezone-aware datetime in UTC."""
    s = s.strip()
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
    """Round to nearest step; keep 3 decimals for MHz at kHz step."""
    return round(round(x / step) * step, 3)


def nanmean(a: List[float]) -> float:
    return float(np.nanmean(a)) if len(a) else float("nan")


def nansd(a: List[float]) -> float:
    return float(np.nanstd(a, ddof=0)) if len(a) else float("nan")


# -------------------------
# Data loading
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
# Tracking across frames
# -------------------------
@dataclass
class Obs:
    utc: datetime
    freq_mhz: float
    p_dbfs: float
    duty: float


@dataclass
class Track:
    """A track is a time-ordered sequence of observations belonging to one signal."""
    last_freq: float
    last_utc: datetime
    last_p: float
    last_duty: float
    points: List[Obs] = field(default_factory=list)
    misses: int = 0
    v_mhz_per_s: float = 0.0
    confirmed: bool = False

    def predict(self, utc: datetime) -> float:
        dt = (utc - self.last_utc).total_seconds()
        if dt <= 0:
            return self.last_freq
        return self.last_freq + self.v_mhz_per_s * dt

    def add(self, o: Obs) -> None:
        if self.points:
            dt = max(1e-9, (o.utc - self.last_utc).total_seconds())
            inst_v = (o.freq_mhz - self.last_freq) / dt
            self.v_mhz_per_s = (1 - VEL_ALPHA) * self.v_mhz_per_s + VEL_ALPHA * inst_v
        self.points.append(o)
        self.last_freq = o.freq_mhz
        self.last_utc = o.utc
        self.last_p = o.p_dbfs
        self.last_duty = o.duty
        self.misses = 0
        if not self.confirmed and len(self.points) >= MIN_CONFIRM_SAMPLES:
            self.confirmed = True


def _frame_groups(df: pd.DataFrame) -> List[Tuple[datetime, pd.DataFrame]]:
    frames: List[Tuple[datetime, pd.DataFrame]] = []
    for utc_val, g in df.groupby("utc", sort=True):
        g = g.sort_values("freq_mhz").reset_index(drop=True)
        frames.append((utc_val, g))
    return frames


def _match_frame_greedy(tracks: List[Track], obs_list: List[Obs], utc_now: datetime) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy one-to-one matching with hysteresis and prediction."""
    if not tracks or not obs_list:
        return [], list(range(len(tracks))), list(range(len(obs_list)))

    pairs: List[Tuple[float, int, int]] = []
    for ti, tr in enumerate(tracks):
        f_pred = tr.predict(utc_now)
        dt = max(0.0, (utc_now - tr.last_utc).total_seconds())
        base_tol = FREQ_TOL + FREQ_TOL_PER_SEC * dt
        tol = base_tol * CONTINUE_TOL_MULT
        for oi, ob in enumerate(obs_list):
            dfreq = abs(ob.freq_mhz - f_pred)
            if dfreq <= tol:
                p_term = 0.05 * abs(ob.p_dbfs - tr.last_p) if not math.isnan(ob.p_dbfs) and not math.isnan(tr.last_p) else 0.0
                cost = dfreq + p_term
                pairs.append((cost, ti, oi))

    pairs.sort(key=lambda x: x[0])
    matched_t = set()
    matched_o = set()
    matches: List[Tuple[int, int]] = []
    for _, ti, oi in pairs:
        if ti in matched_t or oi in matched_o:
            continue
        matched_t.add(ti)
        matched_o.add(oi)
        matches.append((ti, oi))

    unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_o = [i for i in range(len(obs_list)) if i not in matched_o]
    return matches, unmatched_t, unmatched_o


def track_frequencies_by_frame(df: pd.DataFrame) -> List[Track]:
    frames = _frame_groups(df)
    active: List[Track] = []
    finished: List[Track] = []

    for utc_now, g in frames:
        obs_list = [Obs(utc=row.utc, freq_mhz=float(row.freq_mhz), p_dbfs=float(row.p_dbfs), duty=float(row.duty) if not pd.isna(row.duty) else float("nan"))
                    for _, row in g.iterrows()]

        matches, unmatched_t, unmatched_o = _match_frame_greedy(active, obs_list, utc_now)

        for ti, oi in matches:
            active[ti].add(obs_list[oi])
        for ti in unmatched_t:
            active[ti].misses += 1
        for oi in unmatched_o:
            ob = obs_list[oi]
            tr = Track(last_freq=ob.freq_mhz, last_utc=ob.utc, last_p=ob.p_dbfs, last_duty=ob.duty, points=[ob], misses=0)
            tr.confirmed = (MIN_CONFIRM_SAMPLES <= 1)
            active.append(tr)

        still_active: List[Track] = []
        for tr in active:
            if tr.misses > MAX_MISSES:
                finished.append(tr)
            else:
                still_active.append(tr)
        active = still_active

    return finished + active


# -------------------------
# Segments & aggregation (no rounding here)
# -------------------------
def split_continuous_segments(points: List[Obs]) -> List[List[Obs]]:
    if not points:
        return []
    segs: List[List[Obs]] = []
    cur: List[Obs] = [points[0]]
    for i in range(1, len(points)):
        if (points[i].utc - points[i - 1].utc) <= MAX_GAP:
            cur.append(points[i])
        else:
            segs.append(cur)
            cur = [points[i]]
    segs.append(cur)
    return segs


@dataclass
class Segment:
    obs: List[Obs]
    canonical: float | None = None

    @property
    def start(self) -> datetime:
        return self.obs[0].utc

    @property
    def end(self) -> datetime:
        return self.obs[-1].utc

    @property
    def mean_freq(self) -> float:
        return float(np.mean([o.freq_mhz for o in self.obs]))

    def merge_with(self, other: "Segment") -> "Segment":
        return Segment(obs=self.obs + other.obs, canonical=self.canonical)


def _track_canonical_freq(tr: "Track") -> float:
    """Robust canonical for a track: median of the first N observations."""
    if not tr.points:
        return tr.last_freq
    n = min(CANON_USE_FIRST_N, len(tr.points))
    base = [o.freq_mhz for o in tr.points[:n]]
    return float(np.median(base))


def build_segments(tracks: List[Track]) -> List[Segment]:
    segs: List[Segment] = []
    for tr in tracks:
        if not tr.confirmed:
            continue
        canon = _track_canonical_freq(tr) if SNAP_TO_TRACK_CANONICAL else None
        for seg in split_continuous_segments(tr.points):
            if len(seg) < MIN_SEG_SAMPLES:
                continue
            segs.append(Segment(obs=seg, canonical=canon))
    return segs


def merge_adjacent_segments(segs: List[Segment], gap_limit: timedelta, freq_tol: float) -> List[Segment]:
    """Greedy merge of adjacent segments close in time & frequency."""
    if not segs:
        return segs
    segs = sorted(segs, key=lambda s: (s.mean_freq, s.start, s.end))
    merged: List[Segment] = []
    cur = segs[0]
    for nxt in segs[1:]:
        time_gap = (nxt.start - cur.end)
        if timedelta(0) <= time_gap <= gap_limit and abs(nxt.mean_freq - cur.mean_freq) <= freq_tol:
            cur = cur.merge_with(nxt)
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged


# -------------------------
# Stage-1 snapping: per-track canonical (no rounding)
# -------------------------
def apply_track_canonical_snap(freq: float, canonical: Optional[float]) -> float:
    if canonical is None:
        return freq
    if abs(freq - canonical) <= SNAP_TOL_MHZ:
        return canonical
    return freq


# -------------------------
# Stage-2 snapping: global modes via smoothed local peaks (no rounding)
# -------------------------
def _build_binned_counts(freqs: List[float], bin_step: float) -> Tuple[List[float], np.ndarray]:
    """Return (centers, raw_counts) by binning freqs to nearest bin center."""
    if not freqs:
        return [], np.zeros(0, dtype=float)
    def to_bin_center(x: float) -> float:
        return round_to_step(x, bin_step)
    counts: Dict[float, int] = {}
    for x in freqs:
        c = to_bin_center(x)
        counts[c] = counts.get(c, 0) + 1
    centers = sorted(counts.keys())
    raw = np.array([counts[c] for c in centers], dtype=float)
    return centers, raw


def _smooth_counts(raw: np.ndarray) -> np.ndarray:
    """3-point moving average smoothing."""
    if raw.size < 3:
        return raw.copy()
    sm = raw.copy()
    sm[1:-1] = (raw[:-2] + raw[1:-1] + raw[2:]) / 3.0
    return sm


def _local_peak_indices(centers: List[float], raw: np.ndarray, smoothed: np.ndarray, min_count: int) -> List[int]:
    peaks: List[int] = []
    for i in range(len(centers)):
        if raw[i] < min_count:
            continue
        left = smoothed[i - 1] if i - 1 >= 0 else -np.inf
        right = smoothed[i + 1] if i + 1 < len(smoothed) else -np.inf
        if smoothed[i] >= left and smoothed[i] >= right and (smoothed[i] > left or smoothed[i] > right):
            peaks.append(i)
    if not peaks and len(raw) > 0:
        peaks = [int(np.argmax(raw))]
    return peaks


def _weighted_centroid(centers: List[float], raw: np.ndarray, i: int, window_bins: int) -> Tuple[float, float]:
    lo = max(0, i - window_bins)
    hi = min(len(centers) - 1, i + window_bins)
    weights = raw[lo:hi + 1]
    cvals = np.array(centers[lo:hi + 1], dtype=float)
    if weights.sum() == 0:
        return centers[i], 0.0
    centroid = float(np.sum(weights * cvals) / np.sum(weights))
    strength = float(np.sum(weights))
    return centroid, strength


def _fuse_close_peaks(mode_centroids: List[Tuple[float, float]], bin_step: float, merge_bins: int) -> List[float]:
    """Fuse any two peak centroids whose distance <= merge_bins * bin_step."""
    if not mode_centroids:
        return []
    # Sort by centroid
    mode_centroids.sort(key=lambda t: t[0])
    fused: List[Tuple[float, float]] = []
    for c, s in mode_centroids:
        if not fused:
            fused.append((c, s))
            continue
        prev_c, prev_s = fused[-1]
        if abs(c - prev_c) <= (merge_bins * bin_step + 1e-12):
            # fuse by weighted centroid
            total = prev_s + s
            if total > 0:
                new_c = (prev_c * prev_s + c * s) / total
                new_s = total
            else:
                new_c, new_s = c, s
            fused[-1] = (new_c, new_s)
        else:
            fused.append((c, s))
    return [c for c, _ in fused]


def _histogram_modes_strict(freqs: List[float],
                            bin_step: float,
                            min_count: int,
                            peak_window_bins: int,
                            strict_merge_bins: int) -> List[float]:
    """Build global modes from smoothed histogram with STRICT adjacent-peak fusion."""
    centers, raw = _build_binned_counts(freqs, bin_step)
    if len(centers) == 0:
        return []
    smoothed = _smooth_counts(raw)
    peak_idx = _local_peak_indices(centers, raw, smoothed, min_count=min_count)

    # Turn peak indices into (centroid, strength) tuples
    candidates: List[Tuple[float, float]] = []
    for i in peak_idx:
        cen, stren = _weighted_centroid(centers, raw, i, window_bins=peak_window_bins)
        candidates.append((cen, stren))

    # STRICT fuse of adjacent peaks within <= strict_merge_bins
    modes = _fuse_close_peaks(candidates, bin_step=bin_step, merge_bins=strict_merge_bins)
    modes.sort()
    return modes


def apply_global_mode_snap(freqs: List[float],
                           bin_step: float,
                           snap_tol: float,
                           min_count: int,
                           peak_window_bins: int,
                           strict_merge_bins: int) -> List[float]:
    """Snap each frequency to the nearest global mode if within tolerance."""
    modes = _histogram_modes_strict(
        freqs=freqs,
        bin_step=bin_step,
        min_count=min_count,
        peak_window_bins=peak_window_bins,
        strict_merge_bins=strict_merge_bins,
    )
    if not modes:
        return freqs[:]
    out = []
    for x in freqs:
        m = min(modes, key=lambda z: abs(x - z))
        out.append(m if abs(x - m) <= snap_tol else x)
    return out


# -------------------------
# Finalization: build rows, snap, round, write
# -------------------------
def segments_to_rows(segs: List[Segment]) -> pd.DataFrame:
    """Return a DataFrame with float frequencies (no rounding yet)."""
    rows: List[Dict[str, object]] = []
    for s in segs:
        t_start = s.start
        t_end = s.end
        freqs = [p.freq_mhz for p in s.obs]
        pwr = [p.p_dbfs for p in s.obs]
        duty_vals = [p.duty for p in s.obs if not math.isnan(p.duty)]
        seg_mean = float(np.mean(freqs))

        # stage-1: snap to track canonical (no rounding)
        seg_stab = apply_track_canonical_snap(seg_mean, s.canonical)

        rows.append(dict(
            start_utc=t_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_utc=t_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            freq_float=seg_stab,
            samples=int(len(s.obs)),
            pwr_avg=round(nanmean(pwr), 3),
            pwr_sd=round(nansd(pwr), 3),
            duty_avg=(None if not duty_vals else round(nanmean(duty_vals), 3)),
            duty_sd=(None if not duty_vals else round(nansd(duty_vals), 3)),
        ))
    return pd.DataFrame.from_records(rows)


def finalize_and_write(df_rows: pd.DataFrame,
                       out_path: str,
                       round_step: float,
                       do_global_snap: bool,
                       global_params: Dict[str, float | int]) -> pd.DataFrame:
    """Apply stage-2 global snapping (optional), then final rounding and write CSV."""
    if df_rows.empty:
        out_df = pd.DataFrame(columns=["start_utc", "end_utc", "freq", "samples", "pwr_avg", "pwr_sd", "duty_avg", "duty_sd"])
        out_df.to_csv(out_path, index=False)
        print(f"[i] no data. wrote empty CSV: {out_path}")
        print(f"[i] unique frequencies: 0")
        return out_df

    freqs = df_rows["freq_float"].tolist()

    if do_global_snap:
        freqs = apply_global_mode_snap(
            freqs=freqs,
            bin_step=float(global_params["bin_step"]),
            snap_tol=float(global_params["snap_tol"]),
            min_count=int(global_params["min_count"]),
            peak_window_bins=int(global_params["peak_window"]),
            strict_merge_bins=int(global_params["strict_merge_bins"]),
        )

    # final rounding ONLY here
    df_rows["freq"] = [round_to_step(x, round_step) for x in freqs]

    out_df = df_rows[["start_utc", "end_utc", "freq", "samples", "pwr_avg", "pwr_sd", "duty_avg", "duty_sd"]].copy()
    out_df = out_df.sort_values(["freq", "start_utc"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)

    unique_freqs = out_df["freq"].nunique()
    print(f"[i] wrote: {out_path} ({len(out_df)} rows)")
    print(f"[i] unique frequencies: {unique_freqs}")
    return out_df


# -------------------------
# Top-level aggregation
# -------------------------
def build_segments(tracks: List[Track]) -> List[Segment]:
    segs: List[Segment] = []
    for tr in tracks:
        if not tr.confirmed:
            continue
        canon = _track_canonical_freq(tr) if SNAP_TO_TRACK_CANONICAL else None
        for seg in split_continuous_segments(tr.points):
            if len(seg) < MIN_SEG_SAMPLES:
                continue
            segs.append(Segment(obs=seg, canonical=canon))
    return segs


def aggregate_df(df: pd.DataFrame, out_path: str, args) -> pd.DataFrame:
    tracks = track_frequencies_by_frame(df)
    segs = build_segments(tracks)

    # Optional adjacent-merge
    if args.merge_gap_min > 0 and args.merge_freq_tol > 0:
        segs = merge_adjacent_segments(
            segs,
            gap_limit=timedelta(minutes=float(args.merge_gap_min)),
            freq_tol=float(args.merge_freq_tol),
        )

    df_rows = segments_to_rows(segs)

    out_df = finalize_and_write(
        df_rows=df_rows,
        out_path=out_path,
        round_step=float(args.round_freq),
        do_global_snap=bool(args.global_snap_to_modes),
        global_params=dict(
            bin_step=float(args.global_bin_step),
            snap_tol=float(args.global_snap_tol_mhz),
            min_count=int(args.global_mode_min_seg),
            peak_window=int(args.global_peak_window),
            strict_merge_bins=int(args.global_strict_merge_bins),
        ),
    )
    return out_df


def aggregate(directory: Optional[str], file_path: Optional[str], out_path: str, args) -> None:
    if file_path:
        df = load_rows_from_csv(file_path)
    else:
        df = load_all_rows(directory if directory else "./")
    if df.empty:
        out_df = pd.DataFrame(columns=["start_utc", "end_utc", "freq", "samples", "pwr_avg", "pwr_sd", "duty_avg", "duty_sd"])
        out_df.to_csv(out_path, index=False)
        print(f"[i] no data. wrote empty CSV: {out_path}")
        print(f"[i] unique frequencies: 0")
        return

    out_df = aggregate_df(df, out_path, args)

    if args.stats_out:
        stats = dict(
            rows=int(len(out_df)),
            unique_freqs=int(out_df["freq"].nunique()),
        )
        with open(args.stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"[i] wrote stats: {args.stats_out}")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Aggregate RF scan CSVs into continuous segments with temporal tracking and robust two-stage frequency stabilization.")
    # Inputs
    ap.add_argument("--dir", default="./", help="Directory to recursively read CSVs from (default: ./). Ignored if --file is provided.")
    ap.add_argument("--file", default=None, help="Read a single CSV file (overrides --dir).")
    ap.add_argument("--out", default="aggregated_signals.csv", help="Output CSV filename (default: aggregated_signals.csv)")

    # Matching & tracking
    ap.add_argument("--freq_tol", type=float, default=0.0012, help="Base frequency tolerance (MHz) for matching across frames.")
    ap.add_argument("--freq_tol_per_sec", type=float, default=0.0, help="Extra tolerance per second (MHz/s) between frames.")
    ap.add_argument("--continue_tol_mult", type=float, default=1.6, help="Continuation tolerance multiplier (hysteresis).")
    ap.add_argument("--max_gap_min", type=float, default=10.0, help="Max gap (minutes) inside one continuous segment (default: 10).")
    ap.add_argument("--max_misses", type=int, default=2, help="Max consecutive frame misses before a track is terminated (default: 2).")
    ap.add_argument("--min_seg_samples", type=int, default=1, help="Minimum samples per segment to keep (default: 1).")
    ap.add_argument("--min_confirm_samples", type=int, default=2, help="Minimum samples needed to confirm a track (default: 2).")
    ap.add_argument("--vel_alpha", type=float, default=0.5, help="EMA factor for velocity (df/dt).")

    # Post merge (adjacent segments)
    ap.add_argument("--merge_gap_min", type=float, default=1.0, help="Post-merge max time gap between adjacent segments in minutes (default: 1.0).")
    ap.add_argument("--merge_freq_tol", type=float, default=0.0015, help="Post-merge frequency tolerance in MHz (default: 0.0015).")

    # Stage-1: track canonical snapping
    ap.add_argument("--snap_to_track_canonical", type=int, default=1, help="1 to snap segment freq to track canonical if close (default: 1).")
    ap.add_argument("--snap_tol_mhz", type=float, default=0.0015, help="Tolerance (MHz) to snap a segment to its track canonical (default: 0.0015).")
    ap.add_argument("--canon_use_first_n", type=int, default=5, help="Use first N observations of the track to compute canonical freq (default: 5).")

    # Stage-2: global mode snapping (smoothed local peaks with STRICT fusion)
    ap.add_argument("--global_snap_to_modes", type=int, default=1, help="1 to snap frequencies to global modes (default: 1).")
    ap.add_argument("--global_bin_step", type=float, default=0.001, help="Histogram bin step in MHz (default: 0.001).")
    ap.add_argument("--global_snap_tol_mhz", type=float, default=0.0015, help="Tolerance (MHz) to snap to nearest global mode (default: 0.0015).")
    ap.add_argument("--global_mode_min_seg", type=int, default=1, help="Minimum segments in a bin to consider it for peak selection (default: 1).")
    ap.add_argument("--global_peak_window", type=int, default=1, help="±window (in bins) around a peak for weighted centroid (default: 1).")
    ap.add_argument("--global_strict_merge_bins", type=int, default=1, help="Fuse peaks whose distance <= this many bins by weighted centroid (default: 1).")

    # Final rounding & stats
    ap.add_argument("--round_freq", type=float, default=0.001, help="Final rounding step in MHz (default: 0.001).")
    ap.add_argument("--stats_out", default=None, help="Optional path to write a small JSON with {'rows', 'unique_freqs'}.")

    args = ap.parse_args()

    # Bind CLI to globals
    global FREQ_TOL, FREQ_TOL_PER_SEC, CONTINUE_TOL_MULT, MAX_GAP, MAX_MISSES
    global ROUND_FREQ, MIN_SEG_SAMPLES, MIN_CONFIRM_SAMPLES, VEL_ALPHA
    global SNAP_TO_TRACK_CANONICAL, SNAP_TOL_MHZ, CANON_USE_FIRST_N
    global GLOBAL_SNAP_TO_MODES, GLOBAL_BIN_STEP, GLOBAL_SNAP_TOL_MHZ
    global GLOBAL_MODE_MIN_SEG, GLOBAL_PEAK_WINDOW, GLOBAL_STRICT_MERGE_BINS

    FREQ_TOL = float(args.freq_tol)
    FREQ_TOL_PER_SEC = float(args.freq_tol_per_sec)
    CONTINUE_TOL_MULT = float(args.continue_tol_mult)
    MAX_GAP = timedelta(minutes=float(args.max_gap_min))
    MAX_MISSES = int(args.max_misses)
    ROUND_FREQ = float(args.round_freq)
    MIN_SEG_SAMPLES = int(args.min_seg_samples)
    MIN_CONFIRM_SAMPLES = int(args.min_confirm_samples)
    VEL_ALPHA = float(args.vel_alpha)

    SNAP_TO_TRACK_CANONICAL = bool(int(args.snap_to_track_canonical))
    SNAP_TOL_MHZ = float(args.snap_tol_mhz)
    CANON_USE_FIRST_N = int(args.canon_use_first_n)

    GLOBAL_SNAP_TO_MODES = bool(int(args.global_snap_to_modes))
    GLOBAL_BIN_STEP = float(args.global_bin_step)
    GLOBAL_SNAP_TOL_MHZ = float(args.global_snap_tol_mhz)
    GLOBAL_MODE_MIN_SEG = int(args.global_mode_min_seg)
    GLOBAL_PEAK_WINDOW = int(args.global_peak_window)
    GLOBAL_STRICT_MERGE_BINS = int(args.global_strict_merge_bins)

    aggregate(args.dir, args.file, args.out, args)


if __name__ == "__main__":
    main()
