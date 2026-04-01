import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import argparse
import adios2
import os
import re
import numpy as np
import ReaderClass
from rich.traceback import install
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────
#  COLOURS
# ─────────────────────────────────────────────
RESOLUTION_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
    "tab:olive", "tab:cyan", "black", "navy",
    "darkgreen", "crimson", "gold", "teal",
    "magenta", "coral", "lime", "indigo",
]

# ─────────────────────────────────────────────
#  TIME-STEP PER RESOLUTION
# ─────────────────────────────────────────────
DT_PER_RES = {
    "1025-2049": 0.00002500,
    "2049-4097": 0.00002500 * 2,
    "513-1025":  0.00002500 * 4,
    "257-513":   0.00002500 * 8,
    "129-257":   0.00002500 * 16,
}

# ─────────────────────────────────────────────
#  DOMAIN (edit if yours differs)
# ─────────────────────────────────────────────
LX = 1.0
LY = 1.0


# ═════════════════════════════════════════════
#  DNS / ILES CLASSIFICATION
# ═════════════════════════════════════════════
def kolmogorov_scale(re_number):
    """η ≈ Re^(-3/4)  for unit-domain flows."""
    return re_number ** (-3.0 / 4.0)


def classify_run(res_label, re_number):
    """
    Returns a dict with grid spacings, η, ratio, and a status string.
    Δx/η ≤ 1  → DNS
    Δx/η > 1  → ILES (under-resolved)
    """
    nx_str, ny_str = res_label.split("-")
    nx, ny = int(nx_str), int(ny_str)
    dx  = LX / (nx - 1)
    dy  = LY / (ny - 1)
    eta = kolmogorov_scale(re_number)
    ratio_x = dx / eta
    ratio_y = dy / eta
    ratio   = max(ratio_x, ratio_y)          # worst-case direction
    status  = "DNS ✓" if ratio <= 1.0 else f"ILES  (Δ/η ≈ {ratio:.1f}×)"
    return dict(nx=nx, ny=ny, dx=dx, dy=dy, eta=eta,
                ratio=ratio, status=status)


def print_resolution_table(bp_map, re_tag_to_num):
    """Pretty-print classification table to stdout."""
    print("\n" + "═" * 72)
    print("  RESOLUTION / DNS-vs-ILES CLASSIFICATION")
    print("═" * 72)
    header = f"  {'Resolution':<14} {'RE':>8}  {'Δx':>8}  {'Δy':>8}  {'η':>10}  {'Δ/η':>6}  Status"
    print(header)
    print("─" * 72)
    for (res, re_tag) in sorted(bp_map.keys()):
        re_num = re_tag_to_num.get(re_tag, None)
        if re_num is None:
            continue
        c = classify_run(res, re_num)
        print(f"  {res:<14} {re_num:>8}  {c['dx']:>8.5f}  {c['dy']:>8.5f}"
              f"  {c['eta']:>10.6f}  {c['ratio']:>6.2f}  {c['status']}")
    print("═" * 72 + "\n")


# ═════════════════════════════════════════════
#  CONVERGENCE ANALYSIS
# ═════════════════════════════════════════════
def time_averaged_ke(t_arr, e_arr, frac=0.5):
    """Return mean KE over the last `frac` fraction of the time series."""
    n = max(1, int(len(e_arr) * frac))
    return np.nanmean(e_arr[-n:])


def convergence_analysis(energy_series, resolutions, re_tag, re_tag_to_num):
    """
    For a given RE tag, compute:
      - time-averaged KE per resolution
      - L2 difference between successive resolutions (interpolated onto finer grid)
      - estimated convergence rate
    Returns a dict of results.
    """
    # collect (nx, KE_avg, t_arr, e_arr) sorted coarse → fine
    data = []
    for res in resolutions:
        key = (res, re_tag)
        if key not in energy_series:
            continue
        t_arr, e_arr = energy_series[key]
        nx = int(res.split("-")[0])
        ke_avg = time_averaged_ke(t_arr, e_arr)
        data.append((nx, res, ke_avg, t_arr, e_arr))
    data.sort(key=lambda x: x[0])          # coarse → fine

    if len(data) < 2:
        return None

    # ── L2 differences on KE time series ──────────────────
    diffs  = []
    labels = []
    for i in range(len(data) - 1):
        nx_c, res_c, _, t_c, e_c = data[i]
        nx_f, res_f, _, t_f, e_f = data[i + 1]

        # interpolate coarse onto fine time grid
        t_min = max(t_c[0],  t_f[0])
        t_max = min(t_c[-1], t_f[-1])
        if t_max <= t_min:
            continue
        t_common = np.linspace(t_min, t_max, 500)
        ec_i = interp1d(t_c, e_c, kind="linear", fill_value="extrapolate")(t_common)
        ef_i = interp1d(t_f, e_f, kind="linear", fill_value="extrapolate")(t_common)
        l2   = np.sqrt(np.nanmean((ef_i - ec_i) ** 2))
        diffs.append(l2)
        labels.append(f"{res_c}→{res_f}")

    # ── convergence rates ──────────────────────────────────
    rates = []
    for i in range(len(diffs) - 1):
        if diffs[i] > 0 and diffs[i + 1] > 0:
            rate = np.log2(diffs[i] / diffs[i + 1])
        else:
            rate = float("nan")
        rates.append(rate)

    # ── is it converged? ───────────────────────────────────
    ke_avgs  = [d[2] for d in data]
    rel_diff = abs(ke_avgs[-1] - ke_avgs[-2]) / (abs(ke_avgs[-2]) + 1e-30)
    converged = rel_diff < 0.01          # <1 % change in KE avg → converged

    return dict(data=data, diffs=diffs, labels=labels,
                rates=rates, ke_avgs=ke_avgs,
                rel_diff=rel_diff, converged=converged)


def print_convergence_table(conv, re_tag):
    """Print convergence summary to stdout."""
    print(f"  Convergence report — {re_tag}")
    print("  " + "─" * 60)
    print(f"  {'Resolution':<14}  {'KE (time-avg)':>14}  {'Rel Δ to finer':>16}")
    data    = conv["data"]
    ke_avgs = conv["ke_avgs"]
    for i, (nx, res, ke, *_) in enumerate(data):
        if i < len(data) - 1:
            rel = abs(ke_avgs[i+1] - ke) / (abs(ke) + 1e-30)
            print(f"  {res:<14}  {ke:>14.6f}  {rel:>15.2%}")
        else:
            print(f"  {res:<14}  {ke:>14.6f}  {'(finest)':>16}")

    print()
    print(f"  {'Pair':<22}  {'L2 diff':>10}  {'Rate':>8}")
    print("  " + "─" * 44)
    for i, (lab, diff) in enumerate(zip(conv["labels"], conv["diffs"])):
        rate_str = f"{conv['rates'][i-1]:.2f}" if i > 0 and i-1 < len(conv["rates"]) else "—"
        print(f"  {lab:<22}  {diff:>10.2e}  {rate_str:>8}")

    verdict = "✅ CONVERGED" if conv["converged"] else "❌ NOT CONVERGED"
    print(f"\n  Final two resolutions differ by {conv['rel_diff']:.2%} in time-avg KE")
    print(f"  Verdict: {verdict}")
    print("  " + "─" * 60 + "\n")


# ═════════════════════════════════════════════
#  FILE DETECTION
# ═════════════════════════════════════════════
def detect_structure(data_dir):
    bp_map = {}
    for res_entry in sorted(os.listdir(data_dir)):
        res_path = os.path.join(data_dir, res_entry)
        if not os.path.isdir(res_path):
            continue
        if res_entry not in DT_PER_RES:
            print(f"  SKIP (no dt defined): {res_entry}")
            continue
        for fname in sorted(os.listdir(res_path)):
            if not fname.endswith(".bp5"):
                continue
            m = re.search(r'RE(\d+)', fname)
            if not m:
                continue
            re_tag  = f"RE{m.group(1)}"
            bp_path = os.path.realpath(os.path.join(res_path, fname))
            key     = (res_entry, re_tag)
            bp_map[key] = bp_path
            print(f"  Found: res={res_entry:<12}  RE={re_tag:<10}  {bp_path}")
    resolutions = sorted(set(r for r, _ in bp_map))
    re_numbers  = sorted(set(re for _, re in bp_map))
    return resolutions, re_numbers, bp_map


# ═════════════════════════════════════════════
#  READ KE FROM ADIOS2
# ═════════════════════════════════════════════
def read_ke(readIO, bp_path, ux_var, uy_var, xml=None):
    energies = []
    try:
        r = ReaderClass.Reader(readIO, bp_path, xml=xml)
        while True:
            if r.begin_step() != adios2.bindings.StepStatus.OK:
                break
            r.set_read_vars([ux_var, uy_var])
            ux = r.read_step(ux_var)
            uy = r.read_step(uy_var)
            r.end_step()
            if ux is not None and uy is not None:
                energies.append(0.5 * np.mean(ux.squeeze()**2 + uy.squeeze()**2))
            else:
                energies.append(np.nan)
    except Exception as exc:
        print(f"  ERROR reading {bp_path}: {exc}")
        return None
    return np.array(energies) if energies else None


# ═════════════════════════════════════════════
#  PLOTTING
# ═════════════════════════════════════════════
def plot_ke_and_convergence(energy_series, resolutions, re_tag, res_color,
                            conv, output_dir, re_tag_to_num, bp_map):
    """
    One figure per RE tag with three panels:
      left  – KE vs time (all resolutions)
      top-right  – time-averaged KE vs resolution (convergence)
      bot-right  – L2 difference vs resolution pair
    """
    mpl.rcParams.update({
        "font.size": 13, "axes.labelsize": 14,
        "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    })

    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.07, right=0.97,
                            top=0.91, bottom=0.12,
                            wspace=0.35, hspace=0.45)

    ax_ke   = fig.add_subplot(gs[:, 0])      # left — full height
    ax_avg  = fig.add_subplot(gs[0, 1])      # top-right
    ax_l2   = fig.add_subplot(gs[1, 1])      # bottom-right

    # ── panel 1: KE vs time ───────────────────────────────
    for res in resolutions:
        key = (res, re_tag)
        if key not in energy_series:
            continue
        t_arr, e_arr = energy_series[key]
        re_num = re_tag_to_num.get(re_tag, 0)
        c_info = classify_run(res, re_num)
        style  = "-" if "DNS" in c_info["status"] else "--"
        ax_ke.plot(t_arr, e_arr,
                   color=res_color[res], linewidth=1.6,
                   linestyle=style,
                   label=f"{res}  [{c_info['status']}]")

    ax_ke.set_xlabel("Physical time [s]")
    ax_ke.set_ylabel("Kinetic energy")
    ax_ke.set_xlim(left=0)
    ax_ke.set_ylim(bottom=0)
    ax_ke.legend(loc="best", frameon=True, fontsize=9)
    ax_ke.set_title(f"{re_tag} — KE history\n(solid=DNS, dashed=ILES)")

    if conv is None:
        fig.suptitle(f"{re_tag} — insufficient data for convergence", fontsize=13)
        out = os.path.join(output_dir, f"ke_{re_tag}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")
        return

    # ── panel 2: time-averaged KE vs resolution ───────────
    data    = conv["data"]
    nx_vals = [d[0] for d in data]
    ke_avgs = conv["ke_avgs"]
    colors  = [res_color.get(d[1], "black") for d in data]

    ax_avg.plot(range(len(nx_vals)), ke_avgs, "o-",
                color="steelblue", linewidth=2, zorder=3)
    for xi, (ke, col) in enumerate(zip(ke_avgs, colors)):
        ax_avg.scatter(xi, ke, color=col, s=70, zorder=4)
    ax_avg.set_xticks(range(len(nx_vals)))
    ax_avg.set_xticklabels([d[1] for d in data], rotation=30, ha="right", fontsize=9)
    ax_avg.set_ylabel("Time-avg KE")
    verdict = "✅ Converged" if conv["converged"] else "❌ Not converged"
    ax_avg.set_title(f"Grid convergence  —  {verdict}", fontsize=11)
    ax_avg.grid(True, ls="--", alpha=0.4)

    # shade the last two points to highlight convergence check
    if len(nx_vals) >= 2:
        ax_avg.axvspan(len(nx_vals)-2 - 0.3, len(nx_vals)-1 + 0.3,
                       color="green" if conv["converged"] else "red",
                       alpha=0.08)

    # ── panel 3: L2 differences ───────────────────────────
    if conv["diffs"]:
        ax_l2.bar(range(len(conv["diffs"])), conv["diffs"],
                  color=["steelblue"] * len(conv["diffs"]), alpha=0.8, edgecolor="k")
        ax_l2.set_xticks(range(len(conv["diffs"])))
        ax_l2.set_xticklabels(conv["labels"], rotation=30, ha="right", fontsize=8)
        ax_l2.set_ylabel("L2 difference")
        ax_l2.set_yscale("log")
        ax_l2.set_title("L2 norm (KE, successive resolutions)", fontsize=10)
        ax_l2.grid(True, which="both", ls="--", alpha=0.4)

        # annotate convergence rates
        for i, rate in enumerate(conv["rates"]):
            if not np.isnan(rate):
                ax_l2.text(i + 0.5, conv["diffs"][i],
                           f"  rate≈{rate:.1f}", fontsize=8,
                           va="bottom", color="darkred")

    fig.suptitle(f"{re_tag}  |  DNS/ILES classification + grid convergence",
                 fontsize=13, fontweight="bold")

    out = os.path.join(output_dir, f"ke_convergence_{re_tag}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_ke_only(energy_series, resolutions, re_tag, res_color, output_dir):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    for res in resolutions:
        key = (res, re_tag)
        if key not in energy_series:
            continue

        t_arr, e_arr = energy_series[key]

        plt.plot(
            t_arr,
            e_arr,
            color=res_color[res],
            linewidth=1.8,
            label=res
        )

    plt.xlabel("Physical time [s]")
    plt.ylabel("Kinetic energy")
    plt.title(f"{re_tag} — KE vs Time")
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    out = os.path.join(output_dir, f"ke_{re_tag}.png")
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"Saved: {out}")


# ═════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="KE history + DNS/ILES classification + grid convergence")
    parser.add_argument("--data_dir", "-ed", type=str, default="../my_data")
    parser.add_argument("--ux_var",   type=str, default="ux")
    parser.add_argument("--uy_var",   type=str, default="uy")
    parser.add_argument("--readIO",   "-rio", type=str, default="reader1")
    parser.add_argument("--xml",      "-x",   type=str, default=None)
    parser.add_argument("--output_dir", "-o", type=str, default="../RESULTS")
    return parser.parse_args()


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════
def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nScanning: {args.data_dir}\n")
    resolutions, re_numbers, bp_map = detect_structure(args.data_dir)

    if not bp_map:
        print("No .bp5 files found.")
        return

    # map "RE14084" → 14084
    re_tag_to_num = {}
    for re_tag in re_numbers:
        m = re.search(r'\d+', re_tag)
        if m:
            re_tag_to_num[re_tag] = int(m.group())

    # ── print DNS/ILES table ───────────────────────────────
    print_resolution_table(bp_map, re_tag_to_num)

    # ── colours per resolution ────────────────────────────
    res_color = {res: RESOLUTION_COLORS[i % len(RESOLUTION_COLORS)]
                 for i, res in enumerate(resolutions)}

    # ── read all KE series ────────────────────────────────
    energy_series = {}
    for (res, re_tag), bp_path in sorted(bp_map.items()):
        print(f"  Reading: {res}  {re_tag} …")
        e_arr = read_ke(args.readIO, bp_path,
                        args.ux_var, args.uy_var, xml=args.xml)
        if e_arr is None:
            continue
        t_arr = np.linspace(0, 20.0, len(e_arr))
        energy_series[(res, re_tag)] = (t_arr, e_arr)

    # ── convergence + plots per RE tag ────────────────────
    print("\n" + "═" * 72)
    print("  CONVERGENCE ANALYSIS")
    print("═" * 72 + "\n")

    for re_tag in sorted(re_numbers):
        plot_ke_only(
    energy_series,
    resolutions,
    re_tag,
    res_color,
    args.output_dir
)

    # ── DNS/ILES heatmap summary ──────────────────────────
    plot_dns_iles_summary(bp_map, re_tag_to_num, resolutions, args.output_dir)

    print("\nAll done.")


if __name__ == "__main__":
    install()
    main()
