#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
band_sentry_dual_duty.py — Wideband NFM activity scanner/logger for RTL-SDR
(Outputs two duty metrics: single-bin 'duty' and legacy cluster-based 'duty_wide')

Based on your original band_sentry.py with minimal, additive changes:
- Adds Track.active_center_s to accumulate single-bin activity time per frame.
- Keeps original active_s accumulation (now reported as duty_wide).
- Appends a second per-frame loop to count center-bin-only duty without touching the original loop.
- CSV & console now print both duty (single-bin) and duty_wide (legacy cluster/merged).

Deps: pyrtlsdr, numpy, scipy
"""

import argparse, math, time, sys, gzip, csv, signal, datetime, os
import numpy as np
from rtlsdr import RtlSdr
from scipy.signal import get_window
import random

# --------------------------- Utils ---------------------------

def measure_dc_width_hz(sdr, center_mhz, samp_hz, nfft=None, frames=12, margin_db=1.0):
    """Return (width_hz, width_bins, bin_hz, noise_db) of DC spike at center_mhz."""
    # auto-choose nfft so bin <=500 Hz
    if nfft is None:
        nfft = 1
        while samp_hz/nfft > 500 and nfft < 65536: nfft *= 2
    bin_hz = samp_hz / nfft
    win = get_window("hann", nfft, fftbins=True).astype(np.float32)
    win_pow = float((win**2).sum())
    sdr.center_freq = center_mhz*1e6; time.sleep(0.02)

    # avg spectrum
    p_lin = 0
    for _ in range(frames):
        iq = sdr.read_samples(nfft).astype(np.complex64)
        sp = np.fft.fftshift(np.fft.fft(iq*win, n=nfft))
        p_lin += (np.abs(sp)**2)/(win_pow*nfft)
    p_db = 10*np.log10(np.maximum(p_lin/frames,1e-20))

    # noise and threshold
    noise_db = float(np.median(p_db))
    thr_db = noise_db + margin_db

    # expand from center until below threshold
    mid, k = nfft//2, 0
    while k+1 < mid:
        if p_db[mid-(k+1)]>=thr_db or p_db[mid+(k+1)]>=thr_db: k+=1
        else: break

    width_bins = 2*k+1
    return width_bins*bin_hz, width_bins, bin_hz, noise_db

def weighted_tail_sample(lst, n):
    length = len(lst)
    if n >= length:
        return lst[:]

    indices = list(range(length))
    weights = list(range(1, length + 1))

    chosen_idx = random.sample(indices, k=n, counts=weights)

    return [lst[i] for i in chosen_idx]

def format_time(elapsed: float) -> str:
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    return f"{minutes}m{seconds:.3f}s"

def parse_freq_range(s):
    try:
        start, stop = map(float, s.split(","))
        if stop <= start:
            raise ValueError
        return [start, stop]
    except:
        raise argparse.ArgumentTypeError(
            "Each --f-range must be in start,stop format with stop > start, e.g. 440,450"
        )

def aggregate_to_center(values, tolerance=0.75, keep_latest=None):
    if not values:
        return []

    rev = values[::-1]
    groups = []
    cur = [rev[0]]

    for v in rev[1:]:
        mid = (min(cur) + max(cur)) / 2
        if abs(v - mid) <= tolerance:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)

    groups = groups[::-1]

    centers = [round(sum(g) / len(g), 3) for g in groups]

    if keep_latest is not None:
        centers = centers[-keep_latest:]
    return centers

def double_peak_sample(a=0.0, b=1.0, sigma=0.15, mix=0.5):
    if random.random() < mix:  
        val = random.random()
    elif random.random() < 0.5:
        val = min(max(random.gauss(0.25, sigma), 0), 1)
    else:
        val = min(max(random.gauss(0.75, sigma), 0), 1)
    return a + val * (b - a)

def iso_utc(ts=None):
    if ts is None:
        ts = time.time()
    return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")

def db10(x):
    return 10.0 * np.log10(np.maximum(x, 1e-20))

def median_noise_db(p_db, exclude_mid=0):
    """Median noise estimate; optionally exclude ±exclude_mid bins around DC."""
    if exclude_mid and exclude_mid > 0:
        n = p_db.size
        m = n // 2
        mask = np.ones(n, dtype=bool)
        lo = max(0, m - exclude_mid)
        hi = min(n, m + exclude_mid + 1)
        mask[lo:hi] = False
        vals = p_db[mask]
        if vals.size:
            return np.median(vals)
    return np.median(p_db)

def clusters_from_mask(mask):
    """Return list of (start_idx, end_idx) inclusive for contiguous True segments."""
    clusters = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            clusters.append((i, j))
            i = j + 1
        else:
            i += 1
    return clusters

# --------------------------- Tracking ---------------------------

class Track:
    __slots__ = ("f_hz", "active_s", "active_center_s", "max_db", "last_seen", "sum_wf", "sum_w")
    def __init__(self, f_hz, p_db, now):
        self.f_hz = float(f_hz)
        self.active_s = 0.0           # legacy (cluster/merge) activity time
        self.active_center_s = 0.0    # single-bin activity time
        self.max_db = float(p_db)
        self.last_seen = now
        self.sum_wf = 0.0  # freq centroid accumulator
        self.sum_w = 0.0

    def update_centroid(self, f_center_hz, weight=1.0):
        self.sum_wf += float(f_center_hz) * float(weight)
        self.sum_w  += float(weight)
        if self.sum_w > 0:
            self.f_hz = self.sum_wf / self.sum_w

# --------------------------- Core ---------------------------

def run(args):
    # Prepare CSV file (append, named by start UTC)
    start_iso = iso_utc()
    csv_name = f"scan_{start_iso.replace(':','-')}.csv"
    csv_f = open(csv_name, "a", newline="")
    csv_w = csv.writer(csv_f)
    if csv_f.tell() == 0:
        # Columns: original + duty (single-bin) + duty_wide (legacy)
        csv_w.writerow(["utc_iso", "center_mhz", "p_dbfs", "duty", "duty_wide"])
    print(f"[scan] CSV -> {csv_name}")

    ftt_csv_name = f"ftt_{start_iso.replace(':','-')}.csv.gz"
    ftt_csv_f = gzip.open(ftt_csv_name, "at", newline="", encoding="utf-8", compresslevel=5)
    ftt_csv_w = csv.writer(ftt_csv_f)
    print(f"[ftt] CSV -> {csv_name}")

    # RTL-SDR setup (no 0 Hz tuning!)
    sdr = RtlSdr()
    samp_hz = int(args.samp * 1e6)
    sdr.sample_rate = samp_hz

    # Only set PPM if non-zero to avoid buggy drivers complaining about "0 ppm"
    if int(args.ppm) != 0:
        try:
            sdr.freq_correction = int(args.ppm)
        except Exception as e:
            print(f"[w] set ppm failed ({e}); continuing without correction", file=sys.stderr)

    # Gain
    sdr.gain = args.gain if args.gain == "auto" else float(args.gain)

    # FFT params: choose nfft so that bin ~ 500 Hz (cap at 65536 for speed)
    target_bin_hz = 500.0
    nfft = 1
    while (samp_hz / nfft) > target_bin_hz:
        nfft *= 2
        if nfft >= 65536:
            nfft = 65536
            break
    frame = nfft
    hop = nfft  # no overlap for speed
    win = get_window("hann", nfft, fftbins=True).astype(np.float32)
    win_pow = float((win ** 2).sum())

    # Detection settings
    detect_bw_hz = float(args.nbw) * 1e3                  # e.g. 12.5 kHz
    merge_tol_hz = max(6000.0, detect_bw_hz * 0.5)        # cluster merge tolerance
    dc_bins_exclude = int(max(1, args.dc_khz * 1000.0 / (samp_hz / nfft))) if args.dc_suppress else 0
    dutyth = 0.20                                         # fixed 20% rule
    dwell_s = float(args.dwell)
    bin_hz = samp_hz / nfft

    # Scan stepping
    step_mhz = args.samp * (1.0 - args.overlap / 100.0)
    if step_mhz <= 0.0:
        step_mhz = args.samp

    print(f"[i] samp={args.samp:.3f} MHz, exact bin≈{bin_hz:.1f} Hz, nfft={nfft}, dwell={dwell_s:.2f}s")
    for r in args.f_range:
        print(f"[i] scan {r[0]:.3f}–{r[1]:.3f} MHz, step≈{step_mhz:.3f} MHz (overlap={args.overlap}%)")
    print(f"[i] detect BW≈{args.nbw:.2f} kHz, merge_tol≈{merge_tol_hz/1e3:.1f} kHz, DC suppress={args.dc_suppress} (±{args.dc_khz:.1f} kHz)")
    if args.auto_threshold is not None:
        print(f"[i] threshold: noise +{args.auto_threshold:.1f} dB")
    else:
        print(f"[i] threshold: absolute {args.abs_threshold:.1f} dBFS")

    # Generate center freq list
    centers_template = []
    spectrum_dict = {}
    for r in args.f_range:
        ws = r[0]
        while ws < r[1]:
            we = min(r[1], ws + args.samp)
            if we <= ws:
                break
            centers_template.append(0.5 * (ws + we))
            ws += step_mhz

        start_khz = int(round((r[0] - args.samp) * 1000))
        stop_khz  = int(round((r[1] + args.samp) * 1000))
        for khz in range(start_khz, stop_khz + 1):  # 步长1kHz
            key = f"{khz/1000.0:.3f}"
            spectrum_dict[key] = -120.0

    keys_sorted = sorted(spectrum_dict.keys(), key=lambda k: float(k))
    if ftt_csv_f.tell() == 0:
        ftt_csv_w.writerow(["utc"] + keys_sorted)

    # Signal handling
    stop_flag = False
    def on_stop(sig, frm):
        nonlocal stop_flag
        stop_flag = True
    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    # Tuning helper with tiny offsets to coax the PLL if needed
    def tune_with_retry(hz):
        offsets = [0, +1000, -1000, +2000, -2000]  # Hz
        last_err = None
        for off in offsets:
            try:
                sdr.center_freq = float(hz + off)
                time.sleep(0.01)  # small settle
                return True
            except Exception as e:
                last_err = e
        if last_err:
            print(f"[!] tune failed @ {hz/1e6:.6f} MHz: {last_err}", file=sys.stderr)
        return False
    

    w_hz, w_bins, bin_hz, noise = measure_dc_width_hz(sdr, 52, samp_hz)
    print(f"@52MHz DC width ≈ {w_hz/1e3:.1f} kHz ({w_bins} bins, {bin_hz:.1f} Hz/bin), noise {noise:.1f} dBFS")
    w_hz, w_bins, bin_hz, noise = measure_dc_width_hz(sdr, 143, samp_hz)
    print(f"@142MHz DC width ≈ {w_hz/1e3:.1f} kHz ({w_bins} bins, {bin_hz:.1f} Hz/bin), noise {noise:.1f} dBFS")
    w_hz, w_bins, bin_hz, noise = measure_dc_width_hz(sdr, 435, samp_hz)
    print(f"@435MHz DC width ≈ {w_hz/1e3:.1f} kHz ({w_bins} bins, {bin_hz:.1f} Hz/bin), noise {noise:.1f} dBFS")
    w_hz, w_bins, bin_hz, noise = measure_dc_width_hz(sdr, 1200, samp_hz)
    print(f"@1200MHz DC width ≈ {w_hz/1e3:.1f} kHz ({w_bins} bins, {bin_hz:.1f} Hz/bin), noise {noise:.1f} dBFS")


    w_hz_arr = []
    for i in range(20):
        f = random.uniform(30, 1200) 
        w_hz, w_bins, bin_hz, noise = measure_dc_width_hz(sdr, f, samp_hz)
        w_hz_arr.append(w_hz)
    w_hz = np.median(w_hz_arr)
    print(f"DC width ≈ {w_hz/1e3:.1f} kHz")

    # Main scan loop
    # win_start_mhz = args.f_start
    # while not stop_flag:
    #     win_stop_mhz = min(args.f_stop, win_start_mhz + args.samp)
    #     if win_stop_mhz <= win_start_mhz:
    #         # wrap to start
    #         win_start_mhz = args.f_start
    #         continue
    # 
    #     center_mhz = 0.5 * (win_start_mhz + win_stop_mhz)
    #     want_hz = center_mhz * 1e6
    #     if not tune_with_retry(want_hz):
    #         # Skip this window on failure
    #         win_start_mhz = args.f_start if (win_start_mhz + step_mhz) >= args.f_stop else (win_start_mhz + step_mhz)
    #         continue
    
    centers = []
    hits = []

    detailed_scan = False
    detailed_scan_multiplier = 1.0
    start_time = time.time()
    hits_len = 0

    utc_str = iso_utc()

    while not stop_flag:
        if not centers:
            
            #values_sorted = [val for k, val in sorted(spectrum_dict.items(), key=lambda kv: float(kv[0]))]
            values_sorted = [f"{spectrum_dict[k]:.1f}" for k in keys_sorted]
            ftt_csv_w.writerow([utc_str] + values_sorted)
            ftt_csv_f.flush()
            utc_str = iso_utc()

            hits_count = len(hits) - hits_len

            hits = aggregate_to_center(hits)[-1000:].copy()
            hits_len = len(hits)

            end_time = time.time()
            elapsed = end_time - start_time
            print("Full scan" if detailed_scan else "Revisite scan" ,f"elapsed: {format_time(elapsed)}", "hits:", hits_count)
            start_time = time.time()

            centers = centers_template.copy()
            dwell_s = float(args.dwell)

            if detailed_scan:
                centers_len = len(centers)
                centers += weighted_tail_sample(hits, int(elapsed / (float(args.dwell) * detailed_scan_multiplier)))
                print("Normal scan", centers_len, "with revisiting centers", len(centers) - centers_len, "/", len(hits))
                dwell_s = float(args.dwell) * detailed_scan_multiplier
            else:
                print("Normal scan", len(centers))

            
            # Add small random ±5 kHz dither to centers to avoid DC-bin suppression artifacts
            delta = 0
            delta = random.uniform(-(step_mhz / 2), (step_mhz / 2))
            #centers = list(map(lambda f: f + double_peak_sample(-0.005, 0.005) + delta, centers))

            random.shuffle(centers)

            detailed_scan = not detailed_scan

        center_mhz = centers.pop(0)
        want_hz = center_mhz * 1e6
        if not tune_with_retry(want_hz):
            # Skip this window on failure
            continue

        t0 = time.time()
        tracks = []
        fr_dur = frame / float(samp_hz)

        while not stop_flag and (time.time() - t0) < dwell_s:
            # One FFT frame
            iq = sdr.read_samples(frame).astype(np.complex64)
            sp = np.fft.fftshift(np.fft.fft(iq * win, n=nfft))
            p_lin = (np.abs(sp) ** 2) / win_pow / nfft
            p_db = db10(p_lin)

            # DC suppression
            if dc_bins_exclude > 0:
                mid = nfft // 2
                lo = max(0, mid - dc_bins_exclude)
                hi = min(nfft, mid + dc_bins_exclude + 1)
                p_db[lo:hi] = -300.0

            # Snapshot
            freq_axis_hz = want_hz + (np.arange(nfft) - (nfft // 2)) * bin_hz
            for f, db in zip(freq_axis_hz, p_db):
                key = f"{f/1e6:.3f}"
                val = float(db)
                if key in spectrum_dict:
                    spectrum_dict[key] = max(spectrum_dict[key], val)
                else:
                    spectrum_dict[key] = val

            # Threshold
            if args.auto_threshold is not None:
                noise_db = median_noise_db(p_db, exclude_mid=dc_bins_exclude + 3)
                thr_db = float(noise_db + args.auto_threshold)
            else:
                thr_db = float(args.abs_threshold)

            mask = (p_db >= thr_db)
            clusters = clusters_from_mask(mask)

            now = time.time()

            # -------- Original cluster/merge loop (kept intact) --------
            for (i0, i1) in clusters:
                idx = np.arange(i0, i1 + 1)
                slin = p_lin[i0:i1 + 1]
                if slin.size == 0 or slin.sum() <= 0:
                    continue

                # Bin -> frequency offset (FFT is fftshifted)
                f_off = (idx - (nfft // 2)) * bin_hz
                # Power-weighted centroid
                f_center_hz = float((f_off * slin).sum() / slin.sum())
                abs_freq_hz = want_hz + f_center_hz

                # Mean power for this cluster
                p_db_mean = float(db10(slin.mean()))

                # Merge into tracks by frequency proximity
                merged = False
                for tr in tracks:
                    if abs(tr.f_hz - abs_freq_hz) <= merge_tol_hz:
                        tr.active_s += fr_dur
                        if p_db_mean > tr.max_db:
                            tr.max_db = p_db_mean
                        tr.update_centroid(abs_freq_hz, weight=float(slin.sum()))
                        tr.last_seen = now
                        merged = True
                        break
                if not merged:
                    tr = Track(abs_freq_hz, p_db_mean, now)
                    tr.active_s += fr_dur
                    tr.update_centroid(abs_freq_hz, weight=float(slin.sum()))
                    tracks.append(tr)
            # -------- End original loop --------

            # -------- New: single-bin (center-bin) duty accumulation --------
            # To avoid double-counting same track multiple clusters in one frame,
            # record which tracks we've already incremented this frame.
            incremented_ids = set()

            for (i0, i1) in clusters:
                mid_bin = (i0 + i1) // 2
                # Sanity check on index
                if mid_bin < 0 or mid_bin >= nfft:
                    continue
                # If the center bin is above threshold, count this frame for single-bin duty
                if p_db[mid_bin] >= thr_db:
                    # Convert mid_bin to absolute frequency
                    f_off_mid = (mid_bin - (nfft // 2)) * bin_hz
                    abs_freq_hz_mid = want_hz + f_off_mid

                    # Find the nearest track (created by the original loop) within merge tolerance
                    best_tr = None
                    best_df = None
                    for tr in tracks:
                        df = abs(tr.f_hz - abs_freq_hz_mid)
                        if df <= merge_tol_hz and (best_df is None or df < best_df):
                            best_df = df
                            best_tr = tr

                    if best_tr is not None:
                        # Ensure only once per frame per track
                        if id(best_tr) not in incremented_ids:
                            best_tr.active_center_s += fr_dur
                            incremented_ids.add(id(best_tr))
                    else:
                        # If for some reason no track exists (extremely rare, since clusters created above),
                        # create a minimal track so that duty counting is not lost.
                        tr_new = Track(abs_freq_hz_mid, float(p_db[mid_bin]), now)
                        tr_new.active_center_s += fr_dur
                        tr_new.active_s += 0.0  # keep legacy untouched here
                        tracks.append(tr_new)
            # -------- End new loop --------

        # Emit hits
        for tr in tracks:
            # Clip to [0, 1] to avoid >100% due to edge cases
            duty_center = tr.active_center_s / dwell_s if dwell_s > 0 else 0.0
            duty_wide   = tr.active_s         / dwell_s if dwell_s > 0 else 0.0

            if duty_wide >= dutyth or duty_center >= dutyth:
                utc = iso_utc()
                center_mhz_out = tr.f_hz / 1e6
                if round(center_mhz_out, 3) in hits:
                    hits.remove(round(center_mhz_out, 3))
                hits.append(round(center_mhz_out, 3))
                row = [
                    utc,
                    f"{center_mhz_out:.6f}",
                    f"{tr.max_db:.2f}",
                    f"{duty_center*100:.1f}",  # duty (single-bin)
                    f"{duty_wide*100:.1f}",    # duty_wide (legacy)
                ]
                csv_w.writerow(row); csv_f.flush()
                #print(
                #    f"[HIT] {utc}  {center_mhz_out:.6f} MHz  {tr.max_db:.2f} dBFS  "
                #    f"duty={duty_center*100:.1f}%  duty_wide={duty_wide*100:.1f}%"
                #)
                sys.stdout.flush()

        # Advance window
        # next_start = win_start_mhz + step_mhz
        # win_start_mhz = args.f_start if next_start >= args.f_stop else next_start

    # Cleanup
    try:
        sdr.close()
    except Exception:
        pass
    csv_f.close()
    print("[i] exit.")

# --------------------------- CLI ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Wideband NFM activity scanner/logger for RTL-SDR (dual duty outputs)")
    #p.add_argument("--f-start", type=float, default=440.0, help="Start frequency (MHz)")
    #p.add_argument("--f-stop",  type=float, default=450.0, help="Stop  frequency (MHz)")
    p.add_argument("--samp",    type=float, default=2.0,   help="Sample rate / span (MHz), e.g., 2.0~2.4")
    p.add_argument("--dwell",   type=float, default=5.0,     help="Dwell time per window (s)")
    p.add_argument("--gain",    default="20",              help="Gain dB (float) or 'auto' (default 20)")
    p.add_argument("--overlap", type=int,   default=10,    help="Window overlap percent (0-90)")
    p.add_argument("--ppm",     type=int,   default=0,     help="Frequency correction (ppm), default 0")

    # Threshold modes (mutually exclusive)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--auto-threshold", type=float, help="Use noise floor +X dB")
    g.add_argument("--abs-threshold",  type=float, help="Use absolute threshold X dBFS")

    p.add_argument("--nbw", type=float, default=12.5, help="NFM detection bandwidth (kHz), default 12.5")
    # DC suppression default ON; allow disabling with --no-dc-suppress
    p.add_argument("--dc-suppress", action="store_true", default=True, help="Suppress DC at center (default ON)")
    p.add_argument("--no-dc-suppress", dest="dc_suppress", action="store_false")
    p.add_argument("--dc-khz", type=float, default=2.0, help="DC suppression half-width (kHz), default 2")
    p.add_argument("--f-range", action="append", type=parse_freq_range, required=True, help="Frequency range in MHz, e.g. --f-range 440,450 --f-range 470,490")

    args = p.parse_args()
    #if args.f_stop <= args.f_start:
    #    print("[!] f-stop must be > f-start", file=sys.stderr); sys.exit(1)
    if args.overlap < 0 or args.overlap >= 100:
        print("[!] overlap must be in [0, 99]", file=sys.stderr); sys.exit(1)
    return args

if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as e:
        print(f"[!] fatal: {e}", file=sys.stderr)
        sys.exit(2)
