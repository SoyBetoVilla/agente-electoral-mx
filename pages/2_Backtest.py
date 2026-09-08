"""Página de backtest: validación retrospectiva de la metodología."""
import hashlib, io, os, tempfile
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Backtest — Electoral MX", page_icon="🧪", layout="wide")
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

from backtest import Backtest
from dataset_ine import DatasetINE  # noqa: E402


@st.cache_resource(show_spinner="Cargando dataset…")
def _ds(b: bytes, nombre: str, ciclo: str) -> DatasetINE:
    tmp = Path(tempfile.gettempdir()) / f"ine_{hashlib.md5(b).hexdigest()}_{nombre}"
    if not tmp.exists():
        tmp.write_bytes(b)
    return DatasetINE(tmp, ciclo=ciclo)


def _ds_de(fuente, ciclo, etiqueta):
    if fuente is None:
        return None
    try:
        if hasattr(fuente, "getvalue"):
            return _ds(fuente.getvalue(), fuente.name, ciclo)
        return _ds(Path(fuente).read_bytes(), Path(fuente).name, ciclo)
    except Exception as e:
        st.error(f"{etiqueta}: {e}")
        return None


st.title("🧪 Backtest — validación de la metodología")
st.caption("Reconstruye una elección pasada solo con información previa disponible y "
           "compara contra el resultado real (sin fuga de datos).")

c1, c2, c3 = st.columns(3)
ciclo_b = c1.text_input("Ciclo base", "2021")
ciclo_r = c2.text_input("Ciclo real", "2024")
elec = c3.date_input("Fecha de la elección real", value=date(2024, 6, 2))
modo = st.radio("Modo", ["Status quo", "Con encuestas", "Oracle"], horizontal=True)
sims = st.slider("Simulaciones", 500, 20000, 3000, 500)

fb = st.file_uploader(f"Dataset base {ciclo_b} (CSV/XLSX/ZIP)",
                      type=["csv", "xlsx", "xls", "zip"])
ds_b = _ds_de(fb or os.environ.get("RUTA_DATASET_INE_PREV", ""), ciclo_b, "Base")
fr = st.file_uploader(f"Dataset real {ciclo_r}", type=["csv", "xlsx", "xls", "zip"],
                      key="real")
ds_r = _ds_de(fr or os.environ.get("RUTA_DATASET_INE", ""), ciclo_r, "Real")

enc = None
if modo == "Con encuestas":
    fe = st.file_uploader("Encuestas PREVIAS a la elección (CSV)", type=["csv"], key="enc")
    if fe is not None:
        try:
            from proyeccion_2027 import Encuestas
            enc = Encuestas(io.BytesIO(fe.getvalue()),
                            fecha_referencia=elec - timedelta(days=21))
        except Exception as e:
            st.error(f"Encuestas: {e}")

if st.button("▶️ Ejecutar backtest", type="primary", disabled=not (ds_b and ds_r)):
    if ds_b.ciclo == ds_r.ciclo:
        st.warning("El ciclo base y el real deben ser distintos.")
    else:
        with st.spinner("Simulando…"):
            try:
                bt = Backtest(ds_b, ds_r, encuestas=enc, oracle=(modo == "Oracle"),
                              fecha_eleccion=elec, sims=int(sims), seed=42)
                st.session_state.bt = bt
            except Exception as e:
                st.error(f"Error: {e}")

if "bt" in st.session_state:
    bt = st.session_state.bt
    m = bt.metricas
    k = st.columns(5)
    k[0].metric("Acierto", f"{m['acierto']}%", f"naive {m['acierto_naive']}%")
    k[1].metric("Brier", m["brier"], f"skill {m['skill_vs_ingenuo']}")
    k[2].metric("ECE", f"{m['ece_pp']} pp")
    k[3].metric("MAE margen", f"{m['mae_margen_pp']} pp")
    k[4].metric("Flips", f"{m['flip_acertado']}/{m['flips_reales']}",
                f"omitidos {m['flip_omitido']} · FA {m['falsa_alarma']}")
    izq, der = st.columns(2)
    izq.subheader("Confiabilidad del favorito")
    izq.dataframe(bt.tabla_confiabilidad(), use_container_width=True, hide_index=True)
    der.subheader("Escaños: real vs simulación")
    der.dataframe(bt._tabla_escanos(), use_container_width=True, hide_index=True)
    png = Path(tempfile.gettempdir()) / "calibracion_bt.png"
    st.image(str(bt.grafica(png)), use_container_width=True)
    st.subheader("🔁 Clasificación de distritos")
    st.dataframe(bt.tabla["CATEGORIA"].value_counts().rename_frame("Distritos"),
                 use_container_width=True)
    tmp = Path(tempfile.gettempdir()) / "backtest.xlsx"
    st.download_button("⬇️ Descargar backtest (.xlsx)",
                       bt.exportar(tmp).read_bytes(), "backtest.xlsx")
