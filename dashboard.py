"""dashboard.py — Panel interactivo. Detecta módulos opcionales automáticamente."""
from __future__ import annotations

import hashlib, io, os, tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from dataset_ine import DatasetINE, formatear_distrito

try:
    from analisis_comparativo import ComparadorINE
except ImportError:
    ComparadorINE = None
try:
    from proyeccion_2027 import Encuestas, Proyeccion2027
except ImportError:
    Encuestas = Proyeccion2027 = None
try:
    from graficas_ine import wall_map_html
except ImportError:
    wall_map_html = None

st.set_page_config(page_title="Agente Electoral MX", page_icon="🗳️", layout="wide")

try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass


@st.cache_resource(show_spinner="Cargando cómputos…")
def _ds(bytes_: bytes, nombre: str, ciclo: str) -> DatasetINE:
    tmp = Path(tempfile.gettempdir()) / f"ine_{hashlib.md5(bytes_).hexdigest()}_{nombre}"
    if not tmp.exists():
        tmp.write_bytes(bytes_)
    return DatasetINE(tmp, ciclo=ciclo)


def ds_de(fuente, ciclo: str, etiqueta: str) -> DatasetINE | None:
    if fuente is None or fuente == "":
        return None
    try:
        if hasattr(fuente, "getvalue"):
            return _ds(fuente.getvalue(), fuente.name, ciclo)
        return _ds(Path(fuente).read_bytes(), Path(fuente).name, ciclo)
    except Exception as e:
        st.sidebar.error(f"{etiqueta}: {e}")
        return None


# ─────────────── Barra lateral ───────────────
with st.sidebar:
    st.header("⚙️ Datos")
    origen = st.radio("Origen", ["🎮 Demo sintética", "Subir archivo", "Ruta local"],
                      horizontal=True)
    ciclo = st.text_input("Ciclo principal", "2024")
    ds, demo_prev = None, None
    if origen == "🎮 Demo sintética":
        from demo_ine import generar
        r21, r24 = generar(Path(tempfile.gettempdir()) / "demo_ine")
        st.caption("⚠️ Datos **FICTICIOS** para demostración — no son resultados oficiales.")
        ds = ds_de(str(r24 if ciclo == "2024" else r21), ciclo, "Demo")
        demo_prev = str(r21)
    elif origen == "Subir archivo":
        f_main = st.file_uploader(f"Dataset {ciclo} (CSV/XLSX/ZIP)",
                                  type=["csv", "xlsx", "xls", "zip"])
        ds = ds_de(f_main, ciclo, "Dataset principal")
    else:
        ruta = st.text_input("Ruta del CSV", os.environ.get(
            "RUTA_DATASET_INE", "datos_reducidos/Computos_2024.csv"))
        ds = ds_de(ruta, ciclo, "Dataset principal")

    ds_prev = None
    if ComparadorINE is not None:
        st.divider()
        st.subheader("Comparativo (opcional)")
        ciclo_prev = st.text_input("Ciclo anterior", "2021")
        if origen == "🎮 Demo sintética" and demo_prev:
            ds_prev = ds_de(demo_prev, ciclo_prev, "Demo previa")
        else:
            f_prev = st.file_uploader(f"Dataset {ciclo_prev}",
                                      type=["csv", "xlsx", "xls", "zip"], key="prev")
            ds_prev = ds_de(f_prev or os.environ.get("RUTA_DATASET_INE_PREV", ""),
                            ciclo_prev, "Dataset previo")
    st.caption("🗳️ Agente Electoral MX · escenario estadístico, no resultados oficiales.")

if ds is None:
    st.info("👆 Configura el origen de datos en la barra lateral para comenzar.")
    st.stop()

comp = None
if ComparadorINE is not None and ds_prev is not None and ds_prev.ciclo != ds.ciclo:
    try:
        comp = ComparadorINE(ds_prev, ds)
    except Exception as e:
        st.sidebar.warning(f"Comparativo no disponible: {e}")

# ─────────────── Tabs (según módulos presentes) ───────────────
tabs_labels, tiene_comp, tiene_proy = ["📋 Distritos", "🗺️ Mapa"], comp is not None, False
if comp is not None:
    tabs_labels.append("📈 Comparativo")
if Proyeccion2027 is not None:
    tabs_labels.append("🔮 Proyección 2027")
    tiene_proy = True
tabs = st.tabs(tabs_labels)
tab1, tab2 = tabs[0], tabs[1]
tab3 = tabs[2] if tiene_comp else None
tab4 = tabs[3] if tiene_proy else None

# ─────────────── Tab 1: Distritos ───────────────
with tab1:
    df = ds.consultar_todos()
    if "ENTIDAD" not in df or df["ENTIDAD"].isna().all():
        df["ENTIDAD"] = df["ID_ENTIDAD"].astype(str).str.zfill(2)
    c1, c2, c3 = st.columns([2, 2, 3])
    ents = sorted(df["ENTIDAD"].dropna().astype(str).unique())
    ent = c1.selectbox("Entidad", ["(todas)"] + ents)
    gan = c2.multiselect("Ganador", sorted(df["GANADOR"].dropna().unique()))
    q = c3.text_input("Buscar cabecera", "")
    f = df
    if ent != "(todas)":
        f = f[f["ENTIDAD"] == ent]
    if gan:
        f = f[f["GANADOR"].isin(gan)]
    if q:
        f = f[f["CABECERA"].astype(str).str.contains(q, case=False, na=False)]
    st.dataframe(f, use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("🔎 Ficha de un distrito")
    a, b = st.columns(2)
    ent_d = a.selectbox("Entidad", ents, index=ents.index(ent) if ent in ents else 0)
    dists = sorted(df.loc[df["ENTIDAD"] == ent_d, "ID_DISTRITO"].astype(int).unique())
    dis_d = b.selectbox("Distrito", dists)
    if st.button("Mostrar ficha"):
        st.markdown(formatear_distrito(ds.consultar(ent_d, dis_d)))

# ─────────────── Tab 2: Mapa ───────────────
with tab2:
    if wall_map_html is None:
        st.info("El módulo graficas_ine no está presente.")
    else:
        st.caption("Cada cuadro = 1 distrito federal. Pasa el cursor para ver el detalle.")
        fig = wall_map_html(df, "GANADOR", f"Ganador por distrito federal — {ds.ciclo}",
                            cols_hover=["CABECERA", "VOTOS_GANADOR", "PCT_GANADOR", "MARGEN"])
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("⬇️ Descargar mapa (.html)",
                           fig.to_html(include_plotlyjs="cdn"), "wallmap.html")

# ─────────────── Tab 3: Comparativo ───────────────
if tab3 is not None:
    with tab3:
        t = comp.tabla_distritos()
        ca, cb = comp.prev.ciclo, comp.nuevo.ciclo
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Distritos comparados", len(t))
        m2.metric("Cambiaron de bloque", int(t["CAMBIO_GANADOR"].sum()))
        m3.metric("Volatilidad mediana", f"{t['VOLATILIDAD_PP'].median():.1f} pp")
        m4.metric("Swing mediano", f"{t['SWING_PP'].median():+.1f} pp")
        res = comp.resumen_nacional()
        izq, der = st.columns(2)
        izq.subheader("🏛️ Distritos por bloque")
        izq.dataframe(res["bloques"], use_container_width=True)
        der.subheader("📊 Votación por partido")
        der.dataframe(res["partidos"], use_container_width=True)
        st.subheader("🔁 Distritos que cambiaron de bloque")
        st.dataframe(comp.cambios_ganador(), use_container_width=True, hide_index=True)
        st.subheader("🌪️ 15 distritos más volátiles")
        st.dataframe(comp.top_volatiles(15)[["DISTRITO", "VOLATILIDAD_PP", "SWING_PP"]],
                     hide_index=True, use_container_width=True)
        dpp = [c for c in t.columns if c.startswith("DPP_")]
        if dpp:
            st.subheader("Δpp promedio por partido")
            st.bar_chart(t[dpp].mean().rename(lambda c: c[4:]))
        tmp = Path(tempfile.gettempdir()) / "comparativo_dashboard.xlsx"
        comp.exportar(tmp)
        st.download_button("⬇️ Descargar comparativo (.xlsx)", tmp.read_bytes(),
                           "comparativo.xlsx")

# ─────────────── Tab 4: Proyección 2027 ───────────────
if tab4 is not None:
    with tab4:
        st.caption("Escenario estadístico: baseline distrital + swing de encuestas + "
                   "Monte Carlo. No es un pronóstico ni un resultado oficial.")
        c1, c2, c3 = st.columns(3)
        sims = c1.number_input("Simulaciones", 1000, 20000, 5000, 1000)
        fecha_e = c2.date_input("Fecha de la elección", value=date(2027, 6, 6))
        penal = c3.number_input("Penalización al oficialismo (pp)", 0.0, 10.0, 0.0, 0.5)
        f_enc = st.file_uploader("CSV de encuestas (opcional; sin él = status quo)",
                                 type=["csv", "xlsx"])
        if st.button("▶️ Ejecutar proyección", type="primary"):
            try:
                enc = None
                if f_enc is not None:
                    enc = (Encuestas(io.BytesIO(f_enc.getvalue()))
                           if not f_enc.name.lower().endswith((".xlsx", ".xls"))
                           else Encuestas(io.BytesIO(f_enc.getvalue())))
                with st.spinner("Simulando…"):
                    proy = Proyeccion2027(ds, encuestas=enc, ds_prev=ds_prev,
                                          fecha_eleccion=fecha_e,
                                          penalizacion_oficialismo=float(penal))
                    proy.simular(int(sims))
                st.session_state.proy = proy
            except Exception as e:
                st.error(f"Error en la proyección: {e}")
        if "proy" in st.session_state:
            proy = st.session_state.proy
            esc = proy.sim["escanos"]
            lider = esc["media"].idxmax()
            k1, k2, k3 = st.columns(3)
            k1.metric("Bloque líder (escaños esperados)",
                      f"{lider}: {esc.loc[lider, 'media']:.0f}",
                      f"p5–p95: {esc.loc[lider, 'p5']:.0f}–{esc.loc[lider, 'p95']:.0f}")
            k2.metric("Distritos competitivos",
                      f"{int((proy.tabla_proyecciones()['P_GANADOR'] < 65).sum())}"
                      f" de {len(proy.meta)}")
            k3.metric("P(mayoría) del líder", f"{esc.loc[lider, 'p_mayoria']:.0f}%")
            import plotly.graph_objects as go
            fig = go.Figure()
            for bl, r in esc.iterrows():
                fig.add_trace(go.Bar(name=bl, x=[bl], y=[r["media"]],
                                     error_y=dict(type="data", symmetric=False,
                                                  arrayminus=[max(r["media"] - r["p5"], 0)],
                                                  array=[max(r["p95"] - r["media"], 0)])))
            fig.add_hline(y=proy.mayoria, line_dash="dash",
                          annotation_text=f"Mayoría ({proy.mayoria})")
            fig.update_layout(barmode="group", yaxis_title="Distritos ganados",
                              title="Escaños proyectados por bloque (media, banda p5–p95)")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("🔥 Distritos más competitivos")
            st.dataframe(proy.competitivos(20), use_container_width=True, hide_index=True)
            if wall_map_html is not None:
                tproy = proy.tabla_proyecciones()
                figm = wall_map_html(tproy, "GANADOR_PROBABLE",
                                     "Proyección 2027 — ganador probable",
                                     cols_hover=[c for c in tproy.columns
                                                 if c.startswith(("PCT_", "P_"))]
                                     + ["CLASIFICACION"])
                st.plotly_chart(figm, use_container_width=True)
            tmp = Path(tempfile.gettempdir()) / "proyeccion_2027.xlsx"
            proy.exportar(tmp)
            st.download_button("⬇️ Descargar proyección (.xlsx)", tmp.read_bytes(),
                               "proyeccion_2027.xlsx")
