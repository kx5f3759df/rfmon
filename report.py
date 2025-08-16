#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Build time-frequency overview charts and weekly heatmaps; Chinese CSV headers; English UI.
# Tabs for each frequency now show a progress-fill background that encodes occupancy:
# <30% green, 30–70% yellow, >70% red. The fill extends from the left to the occupancy %.
#
# Added:
# - tz_offset_hours CLI/param (e.g., +8 or -12). All timelines, heatmap bins, and tables
#   use time shifted from UTC by this offset. Axis labels and table headers show "UTC±Xh".

from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict
import argparse

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def parse_ts_utc(s: str) -> datetime:
    s = str(s)
    try:
        if s.endswith('Z'):
            return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        return pd.to_datetime(s, utc=True).to_pydatetime()
    except Exception:
        return pd.to_datetime(s, utc=True).to_pydatetime()

def freq_limits_with_padding(freq_series: pd.Series, pad_ratio: float = 0.05) -> Tuple[float, float]:
    fmin = float(freq_series.min())
    fmax = float(freq_series.max())
    if not np.isfinite(fmin) or not np.isfinite(fmax):
        return 0.0, 1.0
    if fmax == fmin:
        pad = max(0.001, (abs(fmax) if fmax != 0 else 1.0) * pad_ratio)
        return fmin - pad, fmax + pad
    span = fmax - fmin
    pad = span * pad_ratio
    return fmin - pad, fmax + pad

def rgba_to_hex(rgba) -> str:
    r, g, b, a = rgba
    return '#{:02x}{:02x}{:02x}'.format(int(max(0,min(1,r))*255), int(max(0,min(1,g))*255), int(max(0,min(1,b))*255))

def _merge_intervals_total_duration(intervals: List[Tuple[datetime, datetime]]) -> float:
    """
    Merge possibly overlapping intervals and return total duration in seconds.
    Intervals with zero duration are ignored (they don't contribute to active time).
    """
    # Filter out zero or negative durations for active time calculation
    norm = [(s, e) for (s, e) in intervals if e > s]
    if not norm:
        return 0.0
    # Sort by start
    norm.sort(key=lambda x: x[0])
    merged: List[Tuple[datetime, datetime]] = []
    cur_s, cur_e = norm[0]
    for s, e in norm[1:]:
        if s <= cur_e:
            # overlap / touch
            if e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    total = sum((e - s).total_seconds() for s, e in merged)
    return float(total)

def build_report(input_csv: str = 'aggregated_signals.csv',
                 out_dir: str = 'report',
                 tz_offset_hours: int = 0) -> None:
    tz_delta = timedelta(hours=int(tz_offset_hours))
    tz_label = f"UTC{int(tz_offset_hours):+d}h"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f'File not found: {input_csv}. Please generate it first.')
    df = pd.read_csv(csv_path)
    req = ['start_utc', 'end_utc', 'freq']
    for col in req:
        if col not in df.columns:
            raise ValueError(f'Missing column: {col}')

    # Parse as UTC, then shift to local (offset) time for all downstream logic/plots/tables
    df['t_start'] = df['start_utc'].map(parse_ts_utc) + tz_delta
    df['t_end']   = df['end_utc'].map(parse_ts_utc) + tz_delta
    df['freq']    = pd.to_numeric(df['freq'], errors='coerce')
    df['t_mid']   = df['t_start'] + (df['t_end'] - df['t_start']) / 2
    df = df.dropna(subset=['t_start','t_end','t_mid','freq']).sort_values('t_start').reset_index(drop=True)
    if df.empty:
        raise ValueError('aggregated_signals.csv has no usable rows.')
    df['freq_key'] = df['freq'].round(3)
    freqs = sorted(df['freq_key'].dropna().unique().tolist())
    tab = matplotlib.colormaps.get_cmap('tab20')
    def color_for_idx(i: int):
        return tab(i % tab.N)
    freq_color: Dict[float, tuple] = {f: color_for_idx(i) for i, f in enumerate(freqs)}

    # --- Compute occupancy per frequency (union of active times) ---
    occupancy_pct: Dict[float, float] = {}
    for f in freqs:
        sub = df[df['freq_key'] == f]
        if sub.empty:
            occupancy_pct[f] = 0.0
            continue
        earliest = min(sub['t_start'])
        latest   = max(sub['t_end'])
        total_span = max(3 * 3600.0, (latest - earliest).total_seconds())
        intervals = [(row['t_start'], row['t_end']) for _, row in sub.iterrows() if row['t_end'] > row['t_start']]
        active_seconds = _merge_intervals_total_duration(intervals)
        occ = (active_seconds / total_span) if total_span > 0 else 0.0
        occ = float(max(0.0, min(1.0, occ)))
        occupancy_pct[f] = occ

    windows = [
        ('30m', timedelta(minutes=30)),
        ('1h', timedelta(hours=1)),
        ('3h', timedelta(hours=3)),
        ('6h', timedelta(hours=6)),
        ('12h', timedelta(hours=12)),
        ('1d', timedelta(days=1)),
        ('7d', timedelta(days=7)),
        ('30d', timedelta(days=30)),
    ]
    # "now" in shifted (local) time
    now_local = datetime.now(timezone.utc) + tz_delta

    overview_images: List[Tuple[str,str]] = []
    for label, delta in windows:
        t_from = now_local - delta
        sub = df[df['t_end'] >= t_from].copy()  # overlap with window

        # Dynamic height: >=25 px per 1 MHz at 150 DPI
        dpi = 150
        base_min_px = 600  # 4 inches baseline
        if sub.empty:
            ymin, ymax = 0.0, 1.0
            span_mhz = float(ymax - ymin)
        else:
            ymin, ymax = freq_limits_with_padding(sub['freq'])
            span_mhz = max(0.0, float(ymax - ymin))
        height_px = max(25.0 * span_mhz, base_min_px)
        height_in = height_px / dpi

        fig = plt.figure(figsize=(12, height_in), dpi=dpi)
        plt.grid(True, alpha=0.3)
        if sub.empty:
            plt.title(f'Last {label} (no data)')
            plt.xlabel(f'Time ({tz_label})')
            plt.ylabel('Frequency (MHz)')
            plt.xlim(t_from, now_local)
            plt.tight_layout()
        else:
            sub['freq_key'] = sub['freq'].round(3)
            labeled = set()
            # Draw in ascending frequency order so legend is sorted
            for f in sorted(sub['freq_key'].unique()):
                freq_rows = sub[sub['freq_key'] == f]
                for _, row in freq_rows.iterrows():
                    x0 = max(row['t_start'], t_from)
                    x1 = min(row['t_end'], now_local)
                    if x1 <= x0:
                        continue
                    if f not in labeled:
                        plt.hlines(y=f, xmin=x0, xmax=x1, linewidth=2, color=freq_color.get(f), label=f'{f:.3f} MHz')
                        labeled.add(f)
                    else:
                        plt.hlines(y=f, xmin=x0, xmax=x1, linewidth=2, color=freq_color.get(f))
            plt.ylim(ymin, ymax)
            from matplotlib.ticker import FormatStrFormatter
            start_tick = int(np.floor(ymin / 5) * 5)
            end_tick = int(np.ceil(ymax / 5) * 5)
            plt.yticks(np.arange(start_tick, end_tick + 5, 5))
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d'))
            plt.title(f'Last {label}: Frequency activity overview')
            plt.xlabel(f'Time ({tz_label})')
            plt.ylabel('Frequency (MHz)')
            plt.xlim(t_from, now_local)
            n_unique = len(sub['freq_key'].unique())
            ncol = max(1, min(6, n_unique))
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=ncol, frameon=True)
            plt.tight_layout(rect=[0, 0.12, 1, 1])
        fname = f'overview_{label}.png'
        fig.savefig(out_path / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        overview_images.append((label, fname))

    # --- Weekly heatmaps & tables (shifted time) ---
    heatmap_images: List[Tuple[float,str]] = []
    table_html_map: Dict[float, str] = {}

    # NOTE: table headers reflect shifted tz
    col_order = ['start_utc','end_utc','freq','samples','pwr_avg','pwr_sd','duty_avg','duty_sd']
    col_rename_display = {
        'start_utc': 'Start (UTC)',
        'end_utc': 'End (UTC)',
        'freq': 'Freq (MHz)',
        'samples': 'Samples',
        'pwr_avg': 'Avg Power (dBFS)',
        'pwr_sd': 'Power Std',
        'duty_avg': 'Avg Duty',
        'duty_sd': 'Duty Std',
    }
    start_col_name = f'Start ({tz_label})'
    end_col_name   = f'End ({tz_label})'

    for f in freqs:
        g = df[df['freq_key'] == f].copy()

        # Build hour bins in shifted time
        seen = set()  # (date_iso, hour, weekday)
        for _, row in g.iterrows():
            cur = row['t_start'].replace(minute=0, second=0, microsecond=0)
            end = row['t_end']
            while cur < end:
                seen.add((cur.date().isoformat(), cur.hour, cur.weekday()))
                cur = cur + timedelta(hours=1)

        mat = np.zeros((24,7), dtype=int)
        for date_iso, hour, wd in seen:
            if 0 <= hour <= 23 and 0 <= wd <= 6:
                mat[hour, wd] += 1

        fig = plt.figure(figsize=(7, 6))
        ax = plt.gca()
        im = ax.imshow(mat, aspect='auto', origin='lower')
        plt.title(f'{f:.3f} MHz - Weekly activity heatmap')
        plt.xlabel('Weekday (Mon=0 ... Sun=6)')
        plt.ylabel(f'Hour ({tz_label})')
        cbar = plt.colorbar(im)
        cbar.set_label('Unique active days')
        ax.set_xticks(range(7))
        ax.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
        ax.set_yticks(range(0,24,2))
        ax.invert_yaxis()
        for x in range(7): ax.axvline(x-0.5, linewidth=0.5)
        for y in range(24): ax.axhline(y-0.5, linewidth=0.2)
        plt.tight_layout()
        fname = f'heatmap_{f:.3f}.png'
        fig.savefig(out_path / fname, dpi=150)
        plt.close(fig)
        heatmap_images.append((f, fname))

        # ----- Frequency table (shifted Start/End) -----
        # Try to include known columns if present
        cols_exist = [c for c in col_order if c in g.columns]
        # We'll render Start/End from shifted timestamps regardless
        fmt_time = '%Y-%m-%d %H:%M:%S'
        start_series = g['t_start']
        end_series   = g['t_end']
        # Build base display DataFrame in time order
        base = pd.DataFrame({
            start_col_name: start_series.dt.strftime(fmt_time),
            end_col_name:   end_series.dt.strftime(fmt_time),
        })
        if cols_exist:
            tbl = g.sort_values('t_start')[cols_exist].copy()
            if 'freq' in tbl.columns:
                tbl['freq'] = pd.to_numeric(tbl['freq'], errors='coerce').round(3)
            rename_map = {k:v for k,v in col_rename_display.items() if k in tbl.columns}
            tbl = tbl.rename(columns=rename_map)
            # Combine: place Start/End first
            tbl_disp = pd.concat([base.reset_index(drop=True), tbl.reset_index(drop=True)], axis=1)
            table_html_map[f] = tbl_disp.to_html(index=False, escape=True, border=0, classes='datatable')
        else:
            # Fallback minimal columns
            fallback = pd.DataFrame({
                'Freq (MHz)': pd.to_numeric(g['freq'], errors='coerce').round(3)
            })
            tbl_disp = pd.concat([base.reset_index(drop=True), fallback.reset_index(drop=True)], axis=1)
            table_html_map[f] = tbl_disp.to_html(index=False, escape=True, border=0, classes='datatable')

    # --- HTML with occupancy progress on frequency tabs ---
    html_path = out_path / 'report.html'
    top_tabs_html_parts = []
    for label, fn in overview_images:
        tab_id = f'top_{label}'
        btn = '<button class="tablink" data-src="{src}" data-target="{tid}">Last {lbl}</button>'.format(src=fn, tid=tab_id, lbl=label)
        top_tabs_html_parts.append(btn)
    top_tabs_html = ' '.join(top_tabs_html_parts)

    tabs_html_parts = []
    content_html_parts = []
    for f, fn in heatmap_images:
        hexcol = rgba_to_hex(freq_color.get(f, (0.2,0.2,0.2,1.0)))
        tab_id = 'tab_{fid}'.format(fid=str(f).replace('.', '_'))
        # occupancy
        occ = float(occupancy_pct.get(f, 0.0))
        pct = int(round(occ * 100))
        # color by threshold
        if occ < 0.30:
            fill = '#2ecc71'   # green
        elif occ <= 0.70:
            fill = '#f1c40f'   # yellow
        else:
            fill = '#e74c3c'   # red
        # background gradient to visualize occupancy
        bg_style = (
            'background-image: linear-gradient(to right, {fill} {pct}%, #f3f3f3 {pct}%);'
            'background-color: #f3f3f3;'
        ).format(fill=fill, pct=pct)
        title_attr = f'title="Occupancy: {pct}% (from earliest start to latest end)"'
        tabs_html_parts.append(
            '<button class="tablink occ" data-target="{tid}" style="border-bottom: 3px solid {col}; {bg}" {title}>{txt}</button>'.format(
                tid=tab_id, col=hexcol, bg=bg_style, title=title_attr, txt=f"{f:.3f} MHz"
            )
        )
        # content
        block = []
        block.append('<div class="tabcontent" id="{tid}" style="display:none">'.format(tid=tab_id))
        block.append('  <h3 style="color:{col}">{txt} — Occupancy: {pct}%</h3>'.format(col=hexcol, txt=f"{f:.3f} MHz", pct=pct))
        block.append('  <img src="{src}" style="max-width:50%; height:auto;"/>'.format(src=fn))
        block.append('  <div class="table-wrap">')
        block.append(table_html_map.get(f, '<p>No data</p>'))
        block.append('  </div>')
        block.append('</div>')
        content_html_parts.append('\n'.join(block))
    tabs_html = '\n'.join(tabs_html_parts)
    tabcontent_html = '\n'.join(content_html_parts)

    html_head = []
    html_head.append('<!DOCTYPE html>')
    html_head.append('<html lang="en">')
    html_head.append('<head>')
    html_head.append('<meta charset="utf-8"/>')
    html_head.append('<title>Spectrum Occupancy Report</title>')
    html_head.append('<style>')
    html_head.append('body { font-family: Arial, sans-serif; margin: 20px; }')
    html_head.append('#top { margin-bottom: 24px; }')
    html_head.append('.tabbar { margin-top: 16px; border-bottom: 1px solid #ccc; padding-bottom: 8px; }')
    html_head.append('.tablink { margin-right: 8px; padding: 8px 12px; cursor: pointer; background: #f3f3f3; border: 1px solid #ccc; border-radius: 6px 6px 0 0; position: relative; }')
    html_head.append('.tablink.active { border-bottom-color: #fff; }')
    html_head.append('.tablink.occ { color: #222; }')
    html_head.append('.tabcontent { padding: 12px 0; }')
    html_head.append('select { padding: 6px 8px; }')
    html_head.append('.table-wrap { overflow-x:auto; margin-top:8px; }')
    html_head.append('.datatable { width:100%; border-collapse: collapse; font-size: 12px; }')
    html_head.append('.datatable th, .datatable td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }')
    html_head.append('.datatable th { background: #f7f7f7; }')
    html_head.append('</style>')
    html_head.append('</head>')
    html_head.append('<body>')
    html_head.append('<h1>Spectrum Occupancy Report</h1>')
    html_head.append(f'<p style="margin-top:-8px; color:#555;">Times shown in {tz_label} (shifted from CSV UTC)</p>')
    html_head.append('<div id="top">')
    html_head.append('  <div class="tabbar" id="topTabs">')
    html_head.append('    __TOP_TABS__')
    html_head.append('  </div>')
    html_head.append('  <div>')
    html_head.append('    <img id="overviewImg" src="__INIT_SRC__" style="max-width:100%; height:auto;"/>')
    html_head.append('  </div>')
    html_head.append('</div>')
    html_head.append('<h2>Weekly heatmaps by frequency</h2>')
    html_head.append('<div class="tabbar">')
    html_head.append('__TABS__')
    html_head.append('</div>')
    html_head.append('__TABCONTENTS__')
    html_head.append('<script>')
    html_head.append('const img = document.getElementById(\'overviewImg\');')
    html_head.append('// Top tabs')
    html_head.append('const topTabs = document.querySelectorAll(\'#topTabs .tablink\');')
    html_head.append('function activateTop(btn){ topTabs.forEach(b => b.classList.toggle(\'active\', b===btn)); img.src = btn.dataset.src; }')
    html_head.append('if (topTabs.length > 0) { activateTop(topTabs[Math.max(0, topTabs.length-3)]); }')
    html_head.append('// Attach click listeners for top tabs')
    html_head.append('for (const b of topTabs) { b.addEventListener(\'click\', () => activateTop(b)); }')
    html_head.append('// Bottom tabs')
    html_head.append('const links = document.querySelectorAll(\'.tabbar .tablink\');')
    html_head.append('const contents = document.querySelectorAll(\'.tabcontent\');')
    html_head.append('function activateTab(targetId) { contents.forEach(c => c.style.display = (c.id === targetId ? \'block\' : \'none\')); links.forEach(l => l.classList.toggle(\'active\', l.dataset.target === targetId)); }')
    html_head.append('const bottomTabs = Array.from(links).filter(l => l.dataset.target && l.closest(\'#topTabs\') === null);')
    html_head.append('if (bottomTabs.length > 0) { activateTab(bottomTabs[0].dataset.target); }')
    html_head.append('bottomTabs.forEach(l => l.addEventListener(\'click\', () => activateTab(l.dataset.target)));')
    html_head.append('</script>')
    html_head.append('</body>')
    html_head.append('</html>')
    html = '\n'.join(html_head)
    html = html.replace('__TOP_TABS__', top_tabs_html)
    html = html.replace('__INIT_SRC__', (overview_images[-1][1] if overview_images else ''))
    html = html.replace('__TABS__', tabs_html)
    html = html.replace('__TABCONTENTS__', tabcontent_html)
    html_path.write_text(html, encoding='utf-8')
    print(f'[i] Report generated: {html_path}')
    html_path.write_text(html, encoding='utf-8')
    print(f'[i] Report generated: {html_path}')

def _parse_args():
    ap = argparse.ArgumentParser(description="Generate spectrum occupancy report with optional time-zone hour offset.")
    ap.add_argument("--input-csv", default="aggregated_signals.csv", help="Input aggregated CSV (default: aggregated_signals.csv)")
    ap.add_argument("--out-dir", default="report", help="Output directory for HTML and images (default: report)")
    ap.add_argument("--tz-offset", type=int, default=0, help="Hour offset from UTC, e.g., +8 for UTC+8, -12 for UTC-12 (default: 0)")
    return ap.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    build_report(input_csv=args.input_csv, out_dir=args.out_dir, tz_offset_hours=args.tz_offset)
