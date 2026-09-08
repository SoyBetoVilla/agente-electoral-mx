"""graficas_ine.py — Wall map esquemático de distritos (PNG y HTML interactivo)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

PALETA = {"PAN": "#0F62AC", "PRI": "#E33A3E", "PRD": "#F6C445", "PT": "#7A1C1C",
          "PVEM": "#1E8449", "MC": "#F58220", "MORENA": "#8F1D22",
          "PAN-PRI-PRD": "#2E6DB4", "MORENA-PT-PVEM": "#8F1D22",
          "OTROS": "#9AA0A6", "CAMBIÓ": "#C0392B", "IGUAL": "#C4C9CE"}

LAYOUT_ENTIDADES = {
    "02": (0, 0), "26": (2, 0), "06": (3, 0), "08": (4, 0), "19": (5, 0), "28": (6, 0),
    "03": (0, 1), "25": (1, 1), "10": (2, 1), "32": (3, 1), "24": (4, 1),
    "18": (1, 2), "14": (2, 2), "01": (3, 2), "11": (4, 2), "22": (5, 2),
    "13": (6, 2), "30": (7, 2),
    "09": (1, 3), "15": (3, 3), "07": (4, 3), "21": (5, 3), "29": (6, 3),
    "31": (8, 3), "23": (9, 3),
    "16": (1, 4), "12": (2, 4), "17": (4, 4), "04": (8, 4),
    "20": (3, 5), "27": (5, 5),
    "05": (4, 6)}
NOMBRE_CORTO = {"01": "AGU", "02": "BC", "03": "BCS", "04": "CAM", "05": "CHIS",
                "06": "CHIH", "07": "CDMX", "08": "COAH", "09": "COL", "10": "DGO",
                "11": "GTO", "12": "GRO", "13": "HGO", "14": "JAL", "15": "EDOMEX",
                "16": "MICH", "17": "MOR", "18": "NAY", "19": "NL", "20": "OAX",
                "21": "PUE", "22": "QRO", "23": "QROO", "24": "SLP", "25": "SIN",
                "26": "SON", "27": "TAB", "28": "TAMP", "29": "TLAX", "30": "VER",
                "31": "YUC", "32": "ZAC"}


def _color(g) -> str:
    return PALETA.get(str(g).strip(), "#9AA0A6")


def _texto_color(hexcol: str) -> str:
    h = hexcol.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 140 else "white"


def _layout_y_base(g: pd.DataFrame) -> tuple[dict, dict, int]:
    counts = g.groupby("ID_ENTIDAD")["ID_DISTRITO"].count().to_dict()
    layout = dict(LAYOUT_ENTIDADES)
    faltantes = [e for e in counts if e not in layout]
    x_extra = max(c for c, _ in layout.values()) + 1
    for i, e in enumerate(faltantes):
        layout[e] = (x_extra + i, 0)
    filas = sorted({f for _, f in layout.values()})
    y_base, y = {}, 0
    for f in filas:
        ents = [e for e, (c, ff) in layout.items() if ff == f]
        y_base[f], y = y, y + max(math.ceil(counts.get(e, 1) / 5) for e in ents) + 1
    return layout, y_base, y


def wall_map(ganadores: pd.DataFrame, col_ganador: str, titulo: str,
             ruta_png, fuente: str = "") -> Path:
    g = ganadores.copy()
    g["ID_ENTIDAD"] = g["ID_ENTIDAD"].astype(str).str.zfill(2)
    g["ID_DISTRITO"] = pd.to_numeric(g["ID_DISTRITO"], errors="coerce")
    g = g[g["ID_DISTRITO"].notna()].sort_values(["ID_ENTIDAD", "ID_DISTRITO"])
    g["ID_DISTRITO"] = g["ID_DISTRITO"].astype(int)
    layout, y_base, y = _layout_y_base(g)
    total_cols = (max(c for c, _ in layout.values()) + 1) * 6
    fig, ax = plt.subplots(figsize=(total_cols * 0.30, (y + 2) * 0.32))
    for ent, grupo in g.groupby("ID_ENTIDAD"):
        col_macro, fila_macro = layout[ent]
        x0, y0 = col_macro * 6, y_base[fila_macro]
        ax.text(x0, y0 - 0.40, f"{NOMBRE_CORTO.get(ent, ent)} ({len(grupo)})",
                fontsize=8, fontweight="bold", ha="left")
        for i, (_, r) in enumerate(grupo.iterrows()):
            x, yy = x0 + (i % 5), y0 + i // 5
            c = _color(r[col_ganador])
            ax.add_patch(Rectangle((x, yy), 0.94, 0.94, facecolor=c,
                                   edgecolor="white", linewidth=0.6))
            ax.text(x + 0.47, yy + 0.47, str(int(r["ID_DISTRITO"])), ha="center",
                    va="center", fontsize=6.5, color=_texto_color(c))
    presentes = sorted({str(v).strip() for v in g[col_ganador].dropna()})
    ax.legend(handles=[Patch(facecolor=_color(p), label=p) for p in presentes],
              loc="lower right", ncol=4, fontsize=8, frameon=False,
              bbox_to_anchor=(1.0, -0.02))
    ax.set_xlim(-0.6, total_cols)
    ax.set_ylim(y + 0.5, -1.3)
    ax.axis("off")
    ax.set_title(f"{titulo}\n{fuente}".strip(), fontsize=12, loc="left")
    ruta_png = Path(ruta_png)
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta_png


def wall_map_html(ganadores: pd.DataFrame, col_ganador: str, titulo: str,
                  ruta_html=None, cols_hover: list[str] | None = None):
    import plotly.graph_objects as go
    g = ganadores.copy()
    g["ID_ENTIDAD"] = g["ID_ENTIDAD"].astype(str).str.zfill(2)
    g["ID_DISTRITO"] = pd.to_numeric(g["ID_DISTRITO"], errors="coerce")
    g = g[g["ID_DISTRITO"].notna()].sort_values(["ID_ENTIDAD", "ID_DISTRITO"])
    g["ID_DISTRITO"] = g["ID_DISTRITO"].astype(int)
    if "ENTIDAD" in g.columns:
        g["ENTIDAD"] = g["ENTIDAD"].fillna(g["ID_ENTIDAD"])
    counts = g.groupby("ID_ENTIDAD")["ID_DISTRITO"].count().to_dict()
    layout, y_base, y = _layout_y_base(g)
    g["IDX"] = g.groupby("ID_ENTIDAD").cumcount()
    g["X"] = g["ID_ENTIDAD"].map(lambda e: layout[e][0] * 6) + g["IDX"] % 5
    g["Y"] = [y_base[layout[e][1]] + i // 5 for e, i in zip(g["ID_ENTIDAD"], g["IDX"])]
    base = [c for c in ("ENTIDAD", "ID_DISTRITO") if c in g.columns] or \
           ["ID_ENTIDAD", "ID_DISTRITO"]
    hover = [c for c in (cols_hover or []) if c in g.columns]
    fig = go.Figure()
    for valor, sub in g.groupby(col_ganador, dropna=False):
        etiqueta = str(valor) if pd.notna(valor) else "s/d"
        tpl = "<b>" + " · ".join(f"%{{customdata[{i}]}}" for i in range(len(base))) + "</b>"
        tpl += "".join(f"<br>{c}: %{{customdata[{len(base) + i}]}}"
                       for i, c in enumerate(hover)) + f"<extra>{etiqueta}</extra>"
        fig.add_trace(go.Scatter(
            x=sub["X"], y=sub["Y"], mode="markers+text", name=etiqueta,
            text=sub["ID_DISTRITO"].astype(str), textposition="middle center",
            textfont=dict(size=9, color=_texto_color(_color(etiqueta))),
            marker=dict(symbol="square", size=24, color=_color(etiqueta),
                        line=dict(color="white", width=1)),
            customdata=sub[base + hover].astype(str).values, hovertemplate=tpl))
    for e, (c, f) in layout.items():
        if e in counts:
            fig.add_annotation(x=c * 6, y=y_base[f] - 0.5,
                               text=f"{NOMBRE_CORTO.get(e, e)} ({counts[e]})",
                               showarrow=False, xanchor="left", font=dict(size=10, color="#444"))
    total_cols = (max(c for c, _ in layout.values()) + 1) * 6
    fig.update_xaxes(range=[-0.8, total_cols], visible=False)
    fig.update_yaxes(range=[y + 0.6, -1.6], visible=False)
    fig.update_layout(title=dict(text=titulo, x=0.01), template="plotly_white",
                      height=max(420, y * 22), margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=-0.01), hoverlabel=dict(font_size=12))
    if ruta_html:
        fig.write_html(ruta_html, include_plotlyjs="cdn")
    return fig
