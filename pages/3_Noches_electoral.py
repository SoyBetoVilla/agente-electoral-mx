"""Página de monitoreo de la noche electoral (lee la carpeta noche/)."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Noche electoral — Electoral MX", page_icon="🌙", layout="wide")
st.title("🌙 Noche electoral — monitoreo en vivo")
d = Path("noche")
if st.button("🔄 Actualizar"):
    st.rerun()
if (d / "estado.json").exists():
    est = json.loads((d / "estado.json").read_text(encoding="utf-8"))
    k = st.columns(4)
    k[0].metric("Distritos contados", f"{est['avance']['distritos_contados']}/300")
    k[1].metric("Actas capturadas", f"≈{est['avance'].get('pct_actas', '—')}%")
    k[2].metric("Sorpresas", est["n_sorpresas"])
    k[3].metric("Cerrados", est["n_cerrados"])
    if est.get("escanos"):
        st.dataframe(pd.DataFrame(est["escanos"]).T.rename(
            columns={"ACTUAL": "Actual", "PROYECCION": "Proyección"}),
            use_container_width=True)
    if est.get("sorpresas_top"):
        st.subheader("⚠️ Últimas sorpresas")
        st.dataframe(pd.DataFrame(est["sorpresas_top"]), hide_index=True,
                     use_container_width=True)
    if (d / "informe_actual.md").exists():
        st.markdown((d / "informe_actual.md").read_text(encoding="utf-8"))
else:
    st.info("Sin monitoreo activo. Lanza el monitor (local o VPS); esta página mostrará "
            "el estado leyendo la carpeta `noche/`.")
    st.code("python noche_electoral.py --config noche.json\n"
            "# ensayo:\n"
            "python noche_electoral.py --fuente-archivo Computos_2024.csv "
            "--base Computos_2024.csv --intervalo 30", language="bash")
