import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, Circle, Patch
from matplotlib.lines import Line2D
import numpy as np
import itertools

# ------------------------------------------------------------------------------
# 1) Topologie-Logik
# ------------------------------------------------------------------------------
def _clean_tuples(items):
    return sorted({tuple(sorted(item)) for item in items})

def get_ring_topology(n=8):
    pairs_nn = [tuple(sorted((i, (i+1)%n))) for i in range(n)]
    pairs_nnn = [tuple(sorted((i, (i+2)%n))) for i in range(n)]
    quads = [tuple(sorted((i, (i+1)%n, (i+2)%n, (i+3)%n))) for i in range(n)]
    return _clean_tuples(pairs_nn), _clean_tuples(pairs_nnn), _clean_tuples(quads)

def get_iqp_topology(n, arch):
    pairs, quads = [], []
    pairs_nn, pairs_nnn, ring_quads = get_ring_topology(n)

    if arch in ["A","B","C","D"]:
        pairs.extend(pairs_nn)
    
    if arch in ["C","D"]:
        pairs.extend(pairs_nnn)
    
    if arch in ["B","D"]:
        quads.extend(ring_quads)
    
    if arch == "E":
        pairs = list(itertools.combinations(range(n), 2))
        quads = []

    return _clean_tuples(pairs), _clean_tuples(quads)

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

# ------------------------------------------------------------------------------
# 3) Topology + RZ Comparison
# ------------------------------------------------------------------------------
def plot_topology_and_rz(n=8):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    
    # Koordinaten (Start oben, im Uhrzeigersinn)
    angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, n, endpoint=False)
    pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    pairs_nn, pairs_nnn, quads = get_ring_topology(n)

    STYLE = {
        "nn":       {"color": "black", "lw": 2.0, "ls": "-"},
        "nnn":      {"color": "#444444", "lw": 1.5, "ls": "--"},
        "quad":     {
                     "facecolor": "none",
                     "edgecolor": "#AAAAAA",
                     "linestyle": ":",
                     "linewidth": 2.0
                    }, 
        "node":     {"facecolor": "white", "edgecolor": "black", "lw": 1.5, "radius": 0.12},
        "rz_ring":  {"edgecolor": "#D62728", "facecolor": "none", "lw": 2.5, "radius": 0.18}
    }

    for ax_idx, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.axis("off")
        
        for q in quads:
            points = pos[list(q)]
            poly = Polygon(points, closed=True, zorder=1, **STYLE["quad"])
            ax.add_patch(poly)

        for (i, j) in pairs_nnn:
            p1, p2 = pos[i], pos[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], zorder=2, **STYLE["nnn"])

        for (i, j) in pairs_nn:
            p1, p2 = pos[i], pos[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], zorder=3, **STYLE["nn"])

        if ax_idx == 1:
            for i in range(n):
                ring = Circle(pos[i], zorder=3.5, **STYLE["rz_ring"])
                ax.add_patch(ring)

        for i in range(n):
            circle = Circle(pos[i], zorder=4, **STYLE["node"])
            ax.add_patch(circle)
            
            txt = ax.text(pos[i,0], pos[i,1], str(i), 
                          ha="center", va="center", 
                          fontsize=11, fontweight="bold", zorder=5)
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    legend_elements = [
        Line2D([0], [0], color=STYLE["nn"]["color"], lw=2, label='NN ZZ'),
        Line2D([0], [0], color=STYLE["nnn"]["color"], lw=1.5, ls='--', label='NNN ZZ'),
        Patch(facecolor="none", edgecolor=STYLE["quad"]["edgecolor"], 
              linestyle=":", linewidth=2, label='local 4-body ZZZZ'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='#D62728', 
               markeredgewidth=2, markersize=10, label='Z-field (RZ)')
    ]

    fig.legend(handles=legend_elements, loc='lower center', 
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.savefig("topology_comparison_lines.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_architectures_final(n=8)
    plot_topology_and_rz(n=8)
