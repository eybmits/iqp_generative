# Final Plot Style (Locked)

This style is embedded directly inside the final scripts.

## Single-panel base (Fig1-Fig5)

- figure size: `243.12/72 x 185.52/72` inch
- `font.size = 12`
- `axes.labelsize = 12`
- `xtick.labelsize = 10`
- `ytick.labelsize = 10`
- `legend.fontsize = 7.2` (or script-local smaller override)
- `lines.linewidth = 2.0`
- `lines.markersize = 6`
- `axes.linewidth = 1.2`
- `xtick.major.width = 1.0`
- `ytick.major.width = 1.0`
- `xtick.major.size = 4`
- `ytick.major.size = 4`
- `legend.framealpha = 0.9`
- `legend.edgecolor = gray`
- `pdf.fonttype = 42`
- `ps.fonttype = 42`

## Fig6 grid style

Fig6 uses a script-specific compact serif style for the 2x4 panel grid while preserving publication-safe export settings (`pdf.fonttype=42`, `ps.fonttype=42`).

## Fig7 appendix style

Fig7 uses a compact two-panel horizontal layout:
- figure size: `6.95 x 2.60` inch
- two axes: `q(H)` vs `n` (left), `R(10000)` vs `n` (right)
- line+marker with mean±std error bars across seeds
- same export font embedding (`pdf.fonttype=42`, `ps.fonttype=42`)

## Rule

Any new final update must keep this style family unless explicitly re-locked.
