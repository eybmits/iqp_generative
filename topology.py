import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, Circle, Patch
from matplotlib.lines import Line2D
import numpy as np
import itertools

# ------------------------------------------------------------------------------
# 1) Topologie-Logik
# ------------------------------------------------------------------------------
def get_iqp_topology(n, arch):
    pairs, quads = [], []
    def clean(l): return sorted(list(set(l)))

    if arch in ["A","B","C","D"]:
        pairs.extend([tuple(sorted((i, (i+1)%n))) for i in range(n)])
    
    if arch in ["C","D"]:
        pairs.extend([tuple(sorted((i, (i+2)%n))) for i in range(n)])
    
    if arch in ["B","D"]:
        quads.extend([tuple(sorted((i, (i+1)%n, (i+2)%n, (i+3)%n))) for i in range(n)])
    
    if arch == "E":
        pairs = list(itertools.combinations(range(n), 2))

    return clean(pairs), clean(quads)

# ------------------------------------------------------------------------------
# 2) Visualisierung (Finaler Stil)
# ------------------------------------------------------------------------------

COLORS = {
    "target":   "#000000",   # NN Linien (Tiefschwarz)
    "nnn":      "#666666",   # NNN Linien (Dunkelgrau)
    "arch_e":   "#D62728",   # Arch E (Rot, wie am Anfang)
    "quad_line":"#AAAAAA",   # 4-Body Umrisse (Mittelgrau)
    "node":     "white",
}

def plot_architectures_final(n=8):
    archs = ["A", "B", "C", "D", "E"]
    
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), constrained_layout=True)
    
    # Koordinaten (Start bei 12 Uhr)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    pos = np.stack([np.sin(angles), np.cos(angles)], axis=1)

    for ax, arch in zip(axes, archs):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Arch {arch}", fontsize=14, fontweight="bold", y=1.02)
        
        pairs, quads = get_iqp_topology(n, arch)
        
        # --- 1. QUADS (Polygone als Linien) ---
        # Keine Füllung mehr, nur "Outline" (Gestrichelt)
        for q in quads:
            points = pos[list(q)]
            poly = Polygon(points, closed=True, 
                           facecolor="none",       # Transparent innen
                           edgecolor=COLORS["quad_line"], 
                           linestyle=":",          # Gepunktet/Gestrichelt
                           linewidth=1.5,
                           zorder=0)               # Ganz im Hintergrund
            ax.add_patch(poly)

        # --- 2. PAIRS (Verbindungslinien) ---
        for (i, j) in pairs:
            p1, p2 = pos[i], pos[j]
            dist = min((i - j) % n, (j - i) % n)
            
            # Logik für Linienstile
            if dist == 1:
                # Nearest Neighbor (Schwarz, dick)
                c, lw, ls, zo, al = COLORS["target"], 2.2, "-", 3, 1.0
            
            elif dist == 2 and arch != "E":
                # Next-Nearest Neighbor (Grau, gestrichelt)
                c, lw, ls, zo, al = COLORS["nnn"], 1.8, "--", 2, 0.9
            
            else:
                # Arch E: All-to-All (Rot, dünn, transparent)
                c, lw, ls, zo, al = COLORS["arch_e"], 0.8, "-", 1, 0.4

            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                    color=c, lw=lw, ls=ls, alpha=al, zorder=zo)

        # --- 3. KNOTEN ---
        for i in range(n):
            # Kreis
            circle = Circle(pos[i], radius=0.13, 
                            facecolor=COLORS["node"], edgecolor="black", 
                            lw=1.5, zorder=4)
            ax.add_patch(circle)
            
            # Text
            txt = ax.text(pos[i,0], pos[i,1], str(i), 
                          ha="center", va="center", 
                          fontsize=10, fontweight="bold", zorder=5)
            # Weißer Rand um Text für besseren Kontrast
            txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])

    # Legende
    legend_elements = [
        Line2D([0], [0], color=COLORS["target"], lw=2, label='NN (1-hop)'),
        Line2D([0], [0], color=COLORS["nnn"], lw=1.8, ls='--', label='NNN (2-hops)'),
        # Hier zeigen wir ein graues Rechteck mit gepunktetem Rand für die 4-Body Legende
        Patch(facecolor="none", edgecolor=COLORS["quad_line"], linestyle=":", linewidth=1.5, label='4-Body (ZZZZ)'),
        Line2D([0], [0], color=COLORS["arch_e"], lw=1, label='All-to-All'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', 
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=11)
    
    plt.savefig("arch_final_style.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_architectures_final(n=8)