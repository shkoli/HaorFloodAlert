"""
Generate fig_dashboard_mockup.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

# Load Nirmala UI (ships with Windows, supports Bengali)
_NIRMALA_PATH = r"C:\Windows\Fonts\Nirmala.ttc"
font_manager.fontManager.addfont(_NIRMALA_PATH)
_bengali_prop = font_manager.FontProperties(fname=_NIRMALA_PATH)

OUT = r"C:\Users\Lenovo\HaorFloodAlert\thesis_screenshots\thesis_figures"
os.makedirs(OUT, exist_ok=True)

fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor("#0F1923")

# ── Main title ────────────────────────────────────────────────────────────────
fig.text(0.5, 0.955, "HaorFloodAlert Dashboard",
         ha="center", va="center", fontsize=20, fontweight="bold",
         color="white", fontfamily="DejaVu Sans")
fig.text(0.5, 0.918, "Sunamganj Haor Flood Early Warning System  |  Live Forecast",
         ha="center", va="center", fontsize=9, color="#90CAF9")

# ── Top row: 4 metric boxes ───────────────────────────────────────────────────
box_ax = fig.add_axes([0.0, 0.73, 1.0, 0.17])
box_ax.set_xlim(0, 1)
box_ax.set_ylim(0, 1)
box_ax.axis("off")
box_ax.set_facecolor("#0F1923")

cards = [
    # (x_left, face_color, edge_color, icon, top_label, value_label, sub_label)
    (0.02,  "#3E1010", "#F44336", "[!]", "FLOOD RISK",     "70.9%",       "HIGH RISK"),
    (0.27,  "#0D1B3E", "#1E88E5", "~~~", "72-HR RAINFALL", "93.7 mm",     "Forecast Total"),
    (0.52,  "#2D1600", "#FF8F00", "^^^", "BARAK RIVER",    "1311 cumecs", "WARNING Level"),
    (0.765, "#0A2E10", "#43A047", "[M]", "INUNDATED AREA", "107.2 km^2",  "Est. Flood Extent"),
]

for x, fc, ec, icon, top_lbl, val, sub in cards:
    rect = FancyBboxPatch((x, 0.06), 0.215, 0.86,
                          boxstyle="round,pad=0.03",
                          facecolor=fc, edgecolor=ec, linewidth=2.0,
                          transform=box_ax.transAxes, clip_on=False)
    box_ax.add_patch(rect)
    cx = x + 0.1075
    box_ax.text(cx, 0.82, icon,      ha="center", va="center", fontsize=16,
                transform=box_ax.transAxes)
    box_ax.text(cx, 0.63, top_lbl,   ha="center", va="center", fontsize=7.5,
                color="#aaa", fontweight="bold", transform=box_ax.transAxes)
    box_ax.text(cx, 0.38, val,        ha="center", va="center", fontsize=15,
                color="white", fontweight="bold", transform=box_ax.transAxes)
    box_ax.text(cx, 0.16, sub,        ha="center", va="center", fontsize=7.5,
                color=ec, fontweight="bold", transform=box_ax.transAxes)

# ── Middle: 5-day rainfall bar chart ─────────────────────────────────────────
ax_bar = fig.add_axes([0.08, 0.27, 0.55, 0.40])
ax_bar.set_facecolor("#141F2B")

days  = ["Day 1\n(Today)", "Day 2", "Day 3", "Day 4", "Day 5"]
rain  = [18.4, 24.7, 31.2, 12.8, 6.6]
risk_colors = ["#EF9A9A", "#F44336", "#B71C1C", "#FF8F00", "#4CAF50"]

bars = ax_bar.bar(days, rain, color=risk_colors, edgecolor="#222",
                  linewidth=0.8, width=0.55, zorder=3)

for bar, val in zip(bars, rain):
    ax_bar.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.6,
                f"{val} mm", ha="center", va="bottom",
                fontsize=8.5, color="white", fontweight="bold")

ax_bar.set_ylim(0, 42)
ax_bar.set_ylabel("Rainfall (mm)", color="#ccc", fontsize=9)
ax_bar.set_title("5-Day Rainfall Forecast (Open-Meteo)",
                 color="white", fontsize=10, fontweight="bold", pad=6)
ax_bar.tick_params(colors="#ccc", labelsize=8.5)
ax_bar.spines[:].set_color("#334")
ax_bar.yaxis.grid(True, color="#253040", lw=0.7, zorder=0)
ax_bar.set_axisbelow(True)

# Cumulative line
cumulative = np.cumsum(rain)
ax2 = ax_bar.twinx()
ax2.set_facecolor("#141F2B")
ax2.plot(days, cumulative, color="#FFD54F", lw=2.0,
         marker="o", markersize=5, label="Cumulative", zorder=4)
ax2.set_ylim(0, 110)
ax2.set_ylabel("Cumulative (mm)", color="#FFD54F", fontsize=8.5)
ax2.tick_params(colors="#FFD54F", labelsize=8)
ax2.spines[:].set_color("#334")
ax2.legend(fontsize=8, loc="upper left",
           facecolor="#1C2B3A", edgecolor="#445", labelcolor="#FFD54F")

# ── Right panel: 3-layer probability gauge ───────────────────────────────────
ax_gauge = fig.add_axes([0.68, 0.27, 0.28, 0.40])
ax_gauge.set_facecolor("#141F2B")
ax_gauge.set_xlim(0, 1)
ax_gauge.set_ylim(0, 1)
ax_gauge.axis("off")
ax_gauge.set_title("3-Layer Probability", color="white",
                   fontsize=10, fontweight="bold", pad=6)

layers = [
    ("Layer 1 — ML Ensemble",    0.519, "#42A5F5", 0.78),
    ("Layer 2 — Barak Discharge",0.619, "#FF8F00", 0.55),
    ("Layer 3 — Trend Adjust",   0.709, "#EF5350", 0.32),
]
for label, prob, color, y in layers:
    # Background track
    track = FancyBboxPatch((0.05, y - 0.055), 0.90, 0.095,
                           boxstyle="round,pad=0.01",
                           facecolor="#1C2B3A", edgecolor="#334", lw=0.8)
    ax_gauge.add_patch(track)
    # Filled bar
    fill = FancyBboxPatch((0.05, y - 0.055), 0.90 * prob, 0.095,
                          boxstyle="round,pad=0.01",
                          facecolor=color, edgecolor="none", alpha=0.85)
    ax_gauge.add_patch(fill)
    ax_gauge.text(0.5, y + 0.075, label, ha="center", va="center",
                  fontsize=7.5, color="#ccc")
    ax_gauge.text(0.97, y, f"{prob*100:.1f}%", ha="right", va="center",
                  fontsize=9, color="white", fontweight="bold")

# Final probability badge
badge = FancyBboxPatch((0.15, 0.04), 0.70, 0.18,
                       boxstyle="round,pad=0.03",
                       facecolor="#B71C1C", edgecolor="#F44336", lw=2.0)
ax_gauge.add_patch(badge)
ax_gauge.text(0.50, 0.155, "p_final = 70.9%", ha="center", va="center",
              fontsize=11, color="white", fontweight="bold")
ax_gauge.text(0.50, 0.075, "HIGH RISK", ha="center", va="center",
              fontsize=9, color="#FFCDD2", fontweight="bold")

# ── Bottom banner ─────────────────────────────────────────────────────────────
ax_bot = fig.add_axes([0.0, 0.0, 1.0, 0.17])
ax_bot.set_facecolor("#0D1B0E")
ax_bot.axis("off")

ax_bot.add_patch(FancyBboxPatch((0.01, 0.12), 0.98, 0.76,
                 boxstyle="round,pad=0.02",
                 facecolor="#0D2010", edgecolor="#2E7D32", lw=1.5,
                 transform=ax_bot.transAxes))

ax_bot.text(0.50, 0.72,
            u"বন্যার ঝুঁকি: উচ্চ  |  SMS Alert Ready",
            ha="center", va="center", fontsize=14,
            color="#A5D6A7", fontweight="bold",
            fontproperties=_bengali_prop,
            transform=ax_bot.transAxes)

ax_bot.text(0.50, 0.30,
            "Target: Farmers · Fishers · Union Parishad · DDMC Focal Points  "
            "|  Lead Time: ~36 h (Barak upstream)  |  Updated: 2026-05-20 06:00 UTC",
            ha="center", va="center", fontsize=8, color="#81C784",
            transform=ax_bot.transAxes)

# ── Save ──────────────────────────────────────────────────────────────────────
path = os.path.join(OUT, "fig_dashboard_mockup.png")
fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.10,
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {path}")
