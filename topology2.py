import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, Circle, Patch
from matplotlib.lines import Line2D
import numpy as np

# ------------------------------------------------------------------------------
# 1) Topologie-Logik
# ------------------------------------------------------------------------------
def get_topology_data(n=8):
    # NN (Nearest Neighbor) -> Distanz 1
    pairs_nn = [tuple(sorted((i, (i+1)%n))) for i in range(n)]
    
    # NNN (Next-Nearest Neighbor) -> Distanz 2
    pairs_nnn = [tuple(sorted((i, (i+2)%n))) for i in range(n)]
    
    # 4-Body (ZZZZ) -> 4 benachbarte Qubits
    quads = [tuple(sorted((i, (i+1)%n, (i+2)%n, (i+3)%n))) for i in range(n)]
    
    return pairs_nn, pairs_nnn, quads

# ------------------------------------------------------------------------------
# 2) Plotting Funktion (Verbessert)
# ------------------------------------------------------------------------------
def plot_topology_and_rz(n=8):
    # Setup: 2 Plots nebeneinander
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    
    # Koordinaten (Start oben, im Uhrzeigersinn)
    angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, n, endpoint=False)
    pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    pairs_nn, pairs_nnn, quads = get_topology_data(n)

    # --- STYLES ---
    # Hier wurde 'quad' angepasst: Keine Füllung, dafür eine Linie
    STYLE = {
        "nn":       {"color": "black", "lw": 2.0, "ls": "-"},
        "nnn":      {"color": "#444444", "lw": 1.5, "ls": "--"}, # Dunkelgrau, gestrichelt
        "quad":     {
                     "facecolor": "none",      # WICHTIG: Keine Füllung (Transparent)
                     "edgecolor": "#AAAAAA",   # Mittelgrau für die Linie
                     "linestyle": ":",         # Gepunktet
                     "linewidth": 2.0          # Etwas dicker, damit man die Punkte sieht
                    }, 
        "node":     {"facecolor": "white", "edgecolor": "black", "lw": 1.5, "radius": 0.12},
        "rz_ring":  {"edgecolor": "#D62728", "facecolor": "none", "lw": 2.5, "radius": 0.18}
    }

    # Optional: Titel entfernen oder anpassen
    # titles = ["Topology (ZZ + ZZZZ)", "With Z-field (RZ)"]

    for ax_idx, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.axis("off")
        
        # --- A) 4-Body Umrisse (ZZZZ) ---
        # Jetzt als gepunktete Linien im Hintergrund (zorder=1)
        for q in quads:
            points = pos[list(q)]
            poly = Polygon(points, closed=True, zorder=1, **STYLE["quad"])
            ax.add_patch(poly)

        # --- B) NNN Linien (Gestrichelt) ---
        for (i, j) in pairs_nnn:
            p1, p2 = pos[i], pos[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], zorder=2, **STYLE["nnn"])

        # --- C) NN Linien (Durchgezogen) ---
        for (i, j) in pairs_nn:
            p1, p2 = pos[i], pos[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], zorder=3, **STYLE["nn"])

        # --- D) RZ Ringe (Nur im rechten Plot) ---
        if ax_idx == 1:
            for i in range(n):
                ring = Circle(pos[i], zorder=3.5, **STYLE["rz_ring"])
                ax.add_patch(ring)

        # --- E) Knoten & Labels ---
        for i in range(n):
            # Knoten
            circle = Circle(pos[i], zorder=4, **STYLE["node"])
            ax.add_patch(circle)
            
            # Text
            txt = ax.text(pos[i,0], pos[i,1], str(i), 
                          ha="center", va="center", 
                          fontsize=11, fontweight="bold", zorder=5)
            # Outline um Text
            txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    # --- Legende ---
    legend_elements = [
        Line2D([0], [0], color=STYLE["nn"]["color"], lw=2, label='NN ZZ'),
        Line2D([0], [0], color=STYLE["nnn"]["color"], lw=1.5, ls='--', label='NNN ZZ'),
        
        # Legende für 4-Body angepasst auf den neuen Style
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
    plot_topology_and_rz(n=8)