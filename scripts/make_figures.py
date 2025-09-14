"""
CLI to produce the three figures described in the write-up.

Usage:
    python -m scripts.make_figures --input ./data/hp_extract.xlsx --datetime-col "date time" --usage-col kWh --temp-col temp
"""
import argparse
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.tacoma.features import load_and_prepare_data

def detect_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def fig_hourly_by_season_daytype(df, usage_col, outpath):
    g = df.groupby(["season_daytype", "hour_of_day"])[usage_col].mean().unstack(0)
    ax = g.plot(figsize=(8,5))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(usage_col)
    ax.set_title("Hourly load by season & day type")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def fig_daily_vs_temp(df, usage_col, temp_col, outpath):
    daily = df.groupby("date")[ [usage_col] + ([temp_col] if temp_col else []) ].mean()
    ax = daily[usage_col].plot(figsize=(10,4), label=usage_col)
    if temp_col:
        daily[temp_col].plot(ax=ax, secondary_y=True, label=temp_col)
        ax.right_ax.set_ylabel(temp_col)
        ax.right_ax.legend(loc="upper right")
    ax.set_title("Daily usage with temperature overlay")
    ax.set_xlabel("Date")
    ax.set_ylabel(usage_col)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def fig_polyfit_usage_temp(df, usage_col, temp_col, outpath):
    if not temp_col:
        return
    x = df[temp_col].values
    y = df[usage_col].values
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return
    coeffs = np.polyfit(x, y, 2)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = np.polyval(coeffs, xs)
    plt.figure(figsize=(6,5))
    plt.scatter(x, y, s=5, alpha=0.3, label="observations")
    plt.plot(xs, ys, linewidth=2, label="quadratic fit")
    plt.xlabel(temp_col)
    plt.ylabel(usage_col)
    plt.title("Usage vs. temperature (quadratic fit)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to Excel/CSV data")
    p.add_argument("--datetime-col", default="date time")
    p.add_argument("--usage-col", default="kWh")
    p.add_argument("--temp-col", default=None)
    p.add_argument("--tz", default=None)
    p.add_argument("--outdir", default="./figures")
    args = p.parse_args()

    df = load_and_prepare_data(
        args.input,
        datetime_col=args.datetime_col,
        usage_col=args.usage_col,
        temp_col=args.temp_col,
        tz=args.tz,
        set_index=True,
    )

    usage_col = args.usage_col if args.usage_col in df.columns else detect_col(df, ["kWh", "usage", "consumption"])
    temp_col = args.temp_col if (args.temp_col and args.temp_col in df.columns) else detect_col(df, ["temp", "temperature", "t_mean"])

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if usage_col is None:
        raise SystemExit("Could not find usage column—pass --usage-col explicitly.")

    fig_hourly_by_season_daytype(df, usage_col, outdir / "fig_hourly_by_season_daytype.png")
    fig_daily_vs_temp(df, usage_col, temp_col, outdir / "fig_daily_vs_temp.png")
    fig_polyfit_usage_temp(df, usage_col, temp_col, outdir / "fig_polyfit_usage_temp.png")

if __name__ == "__main__":
    main()
