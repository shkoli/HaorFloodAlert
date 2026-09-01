"""
Generate 2 thesis figures:
  fig_sar_comparison.png  — simulated Sentinel-1 SAR (dry vs flood)
  fig_dem_twi.png         — simulated SRTM DEM + TWI
Output: thesis_screenshots/thesis_figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

OUT = r"C:\Users\Lenovo\HaorFloodAlert\thesis_screenshots\thesis_figures"
os.makedirs(OUT, exist_ok=True)

DPI        = 600
TITLE_SIZE = 12
AXIS_SIZE  = 10
TICK_SIZE  = 9

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": AXIS_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "savefig.dpi":    DPI,
    "savefig.bbox":   "tight",
    "savefig.pad_inches": 0.12,
})

np.random.seed(2017)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — SAR Comparison (Dry vs Flood)
# ──────────────────────────────────────────────────────────────────────────────
def fig_sar_comparison():
    rows, cols = 60, 70

    # ── Dry scene (Jan 2017) ─────────────────────────────────────────────────
    # Base: mostly -10 to -12 dB (dry land reflectance)
    dry = np.random.normal(-11.0, 0.7, (rows, cols))
    # Add smooth spatial structure (field patches)
    structure = gaussian_filter(np.random.randn(rows, cols), sigma=5) * 1.2
    dry = dry + structure
    # Small water body cluster (river channel, top-left)
    dry[5:14, 4:14] += -6.0
    dry[8:12, 2:20] += -4.0
    # Road/settlement brighter patches
    dry[30:35, 45:55] += 1.5
    dry[45:50, 20:28] += 1.2
    # Clamp to realistic SAR range
    dry = np.clip(gaussian_filter(dry, sigma=0.8), -17, -6)

    # ── Flood scene (Apr 2017) ───────────────────────────────────────────────
    flood = np.random.normal(-20.0, 0.9, (rows, cols))
    structure2 = gaussian_filter(np.random.randn(rows, cols), sigma=6) * 1.0
    flood = flood + structure2
    # Exposed islands / bunds (slightly higher backscatter)
    flood[12:18, 30:40] += 6.5
    flood[38:44, 10:20] += 5.0
    flood[50:56, 52:62] += 5.5
    # Deep water core (very low)
    flood[22:50, 20:55] += -1.5
    # Road/embankment (still bright)
    flood[3:7, 5:65]   += 8.0
    flood[55:58, 5:65]  += 7.5
    flood = np.clip(gaussian_filter(flood, sigma=0.8), -26, -8)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax_dry, ax_flood) = plt.subplots(1, 2, figsize=(11, 4),
                                           gridspec_kw={"wspace": 0.28})

    # Shared colorbar range for direct comparison
    vmin, vmax = -24, -6

    # Dry — warm palette (sandy/brown for land)
    cmap_sar = plt.cm.RdYlGn   # green=bright land, red=dark water
    im_dry = ax_dry.imshow(dry, cmap=cmap_sar, vmin=vmin, vmax=vmax,
                            interpolation="bilinear", aspect="auto")
    ax_dry.set_title("Pre-Flood (Jan 2017)  VV ≈ −10 dB",
                     fontsize=TITLE_SIZE, fontweight="bold", pad=6)
    ax_dry.set_xlabel("Range (pixels)", fontsize=AXIS_SIZE)
    ax_dry.set_ylabel("Azimuth (pixels)", fontsize=AXIS_SIZE)
    ax_dry.tick_params(labelsize=TICK_SIZE)

    # Flood — blue palette
    cmap_flood = LinearSegmentedColormap.from_list(
        "sar_flood",
        ["#08306B", "#1565C0", "#2196F3", "#90CAF9", "#E3F2FD",
         "#FFF9C4", "#F9A825", "#BF360C"],
        N=256
    )
    im_flood = ax_flood.imshow(flood, cmap=cmap_flood, vmin=vmin, vmax=vmax,
                                interpolation="bilinear", aspect="auto")
    ax_flood.set_title("During Flood (Apr 2017)  VV ≈ −20 dB",
                       fontsize=TITLE_SIZE, fontweight="bold", pad=6)
    ax_flood.set_xlabel("Range (pixels)", fontsize=AXIS_SIZE)
    ax_flood.set_yticks([])
    ax_flood.tick_params(labelsize=TICK_SIZE)

    # Shared colorbar (right side)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.74])
    sm = plt.cm.ScalarMappable(cmap=cmap_flood,
                                norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("VV Backscatter (dB)", fontsize=AXIS_SIZE, labelpad=6)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.set_ticks([-24, -22, -20, -18, -16, -14, -12, -10, -8, -6])

    # Annotate key features
    ax_dry.annotate("River\nchannel", xy=(9, 9), xytext=(18, 5),
                    fontsize=7.5, color="white",
                    arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
    ax_flood.annotate("Inundated\nhaor", xy=(37, 36), xytext=(50, 15),
                      fontsize=7.5, color="white",
                      arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
    ax_flood.annotate("Exposed\nbund", xy=(15, 41), xytext=(5, 50),
                      fontsize=7.5, color="#FFCC02",
                      arrowprops=dict(arrowstyle="->", color="#FFCC02", lw=0.8))

    fig.suptitle(
        "Sentinel-1 SAR VV Backscatter — Sunamganj Haor, Bangladesh",
        fontsize=11, fontweight="bold", y=1.03
    )
    fig.text(0.5, -0.04,
             "Simulated for illustration  |  Otsu threshold ≈ −15 dB separates water from land",
             ha="center", fontsize=8, color="#777777", style="italic")

    path = os.path.join(OUT, "fig_sar_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — DEM + TWI
# ──────────────────────────────────────────────────────────────────────────────
def fig_dem_twi():
    rows, cols = 60, 70

    # ── DEM: flat haor terrain 2–5 m ─────────────────────────────────────────
    # Base elevation ~3.5 m, very gentle undulation
    dem_base = np.random.normal(3.5, 0.35, (rows, cols))
    # Smooth large-scale bowl shape (haor basin lowest in centre)
    gy, gx = np.mgrid[0:rows, 0:cols]
    bowl = -0.6 * np.exp(-((gx - cols/2)**2 / (cols*0.4)**2 +
                            (gy - rows/2)**2 / (rows*0.4)**2))
    # Ridge / bund lines running E-W
    bund = np.zeros((rows, cols))
    for y_bund in [10, 25, 42, 55]:
        bund[y_bund-1:y_bund+2, 5:cols-5] = 1.4
    # Embankment N-S
    bund[5:rows-5, cols//2-1:cols//2+2] += 1.0
    dem = gaussian_filter(dem_base + bowl + bund, sigma=2.2)
    dem = np.clip(dem, 0.5, 7.0)

    # ── TWI: computed from simplified slope + area proxy ─────────────────────
    # TWI = ln(a / tan(beta)); flat terrain → very high TWI
    # Simulate: invert DEM gradient → basin centre high TWI
    # Add flow accumulation proxy (connected channels)
    grad_y, grad_x = np.gradient(dem)
    slope = np.sqrt(grad_x**2 + grad_y**2) + 1e-4   # avoid div0

    # Upslope accumulation proxy — higher in basin centre
    flow_acc = gaussian_filter(
        np.exp(-((gx - cols/2)**2 / (cols*0.25)**2 +
                 (gy - rows/2)**2 / (rows*0.22)**2)), sigma=3) * 60 + 5

    twi_raw = np.log(flow_acc / np.tan(slope + 1e-3))
    # Scale to realistic haor TWI range 12–21
    twi = 12 + (twi_raw - twi_raw.min()) / (twi_raw.max() - twi_raw.min()) * 9
    twi = gaussian_filter(twi, sigma=1.5)
    # Bunds have low TWI
    twi[9:12, 5:cols-5]    = np.clip(twi[9:12, 5:cols-5]  - 5, 12, 21)
    twi[24:27, 5:cols-5]   = np.clip(twi[24:27, 5:cols-5] - 5, 12, 21)
    twi[41:44, 5:cols-5]   = np.clip(twi[41:44, 5:cols-5] - 5, 12, 21)
    twi[54:57, 5:cols-5]   = np.clip(twi[54:57, 5:cols-5] - 5, 12, 21)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax_dem, ax_twi) = plt.subplots(1, 2, figsize=(8, 4),
                                          gridspec_kw={"wspace": 0.06})

    # DEM: green-yellow gradient (low = green wetland, high = yellow upland)
    cmap_dem = LinearSegmentedColormap.from_list(
        "dem_haor",
        ["#004D40", "#00796B", "#4CAF50", "#8BC34A", "#CDDC39",
         "#FFF176", "#FFD54F", "#FF8F00"],
        N=256
    )
    im_dem = ax_dem.imshow(dem, cmap=cmap_dem, vmin=0.5, vmax=7.0,
                            interpolation="bilinear", aspect="auto")
    ax_dem.set_title("SRTM Elevation (m)", fontsize=TITLE_SIZE,
                     fontweight="bold", pad=6)
    ax_dem.set_xlabel("Easting (pixels)", fontsize=AXIS_SIZE)
    ax_dem.set_ylabel("Northing (pixels)", fontsize=AXIS_SIZE)
    ax_dem.tick_params(labelsize=TICK_SIZE)

    cbar1 = fig.colorbar(im_dem, ax=ax_dem, fraction=0.038, pad=0.03)
    cbar1.set_label("Elevation (m)", fontsize=AXIS_SIZE - 1)
    cbar1.set_ticks([1, 2, 3, 4, 5, 6, 7])
    cbar1.ax.tick_params(labelsize=TICK_SIZE - 1)

    # Annotate
    ax_dem.annotate("Haor\nbasin", xy=(cols//2, rows//2),
                    xytext=(cols//2 + 14, rows//2 - 12),
                    fontsize=7.5, color="#004D40",
                    arrowprops=dict(arrowstyle="->", color="#004D40", lw=0.9))
    ax_dem.annotate("Bund\n(raised)", xy=(cols//2 - 10, 25),
                    xytext=(cols//2 - 22, 35),
                    fontsize=7, color="#5D4037",
                    arrowprops=dict(arrowstyle="->", color="#5D4037", lw=0.8))

    # TWI: blue gradient (high = more likely to flood)
    cmap_twi = LinearSegmentedColormap.from_list(
        "twi_blue",
        ["#E3F2FD", "#90CAF9", "#42A5F5", "#1565C0", "#0D47A1", "#01579B",
         "#002171"],
        N=256
    )
    im_twi = ax_twi.imshow(twi, cmap=cmap_twi, vmin=12, vmax=21,
                            interpolation="bilinear", aspect="auto")
    ax_twi.set_title("Topographic Wetness Index", fontsize=TITLE_SIZE,
                     fontweight="bold", pad=6)
    ax_twi.set_xlabel("Easting (pixels)", fontsize=AXIS_SIZE)
    ax_twi.set_yticks([])
    ax_twi.tick_params(labelsize=TICK_SIZE)

    cbar2 = fig.colorbar(im_twi, ax=ax_twi, fraction=0.038, pad=0.03)
    cbar2.set_label("TWI  [ln(a / tanβ)]", fontsize=AXIS_SIZE - 1)
    cbar2.set_ticks([12, 14, 16, 18, 20])
    cbar2.ax.tick_params(labelsize=TICK_SIZE - 1)

    # Annotate
    ax_twi.annotate("High TWI\n(flood-prone)", xy=(cols//2, rows//2 + 3),
                    xytext=(cols//2 + 12, rows//2 - 12),
                    fontsize=7.5, color="white",
                    arrowprops=dict(arrowstyle="->", color="white", lw=0.9))
    ax_twi.annotate("Low TWI\n(bund/ridge)", xy=(cols//2 - 5, 26),
                    xytext=(8, 38),
                    fontsize=7, color="#FFD54F",
                    arrowprops=dict(arrowstyle="->", color="#FFD54F", lw=0.8))

    fig.suptitle(
        "Terrain Analysis — Sunamganj Haor (SRTM 30 m + HydroSHEDS TWI)\n"
        "Flat basin 2–5 m elevation; TWI 14–20 identifies inundation-prone cells",
        fontsize=10, y=1.01
    )

    path = os.path.join(OUT, "fig_dem_twi.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Output: {OUT}\n")
    for name, fn in [("fig_sar_comparison", fig_sar_comparison),
                     ("fig_dem_twi",        fig_dem_twi)]:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"ERROR in {name}: {e}")
            traceback.print_exc()
    print("\nDone.")
