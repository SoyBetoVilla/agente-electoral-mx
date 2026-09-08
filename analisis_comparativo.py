"""analisis_comparativo.py — Comparativo entre ciclos: Δpp, volatilidad (Pedersen),
ganador por bloque, flips y swings."""
from __future__ import annotations

import re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_ine import DatasetINE

PARTIDOS_TOKEN = ["MORENA", "PAN", "PRI", "PRD", "PVEM", "PT", "MC"]
ALIAS = {"MOVIMIENTO CIUDADANO": "MC", "PARTIDO DEL TRABAJO": "PT", "PARTIDO VERDE": "PVEM"}
BLOQUE_DE_PARTIDO = {
    "PAN": "PAN-PRI-PRD", "PRI": "PAN-PRI-PRD", "PRD": "PAN-PRI-PRD",
    "MORENA": "MORENA-PT-PVEM", "PT": "MORENA-PT-PVEM", "PVEM": "MORENA-PT-PVEM",
    "MC": "MC"}


def _sin_acentos(t) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(t))
                   if unicodedata.category(c) != "Mn")


def token_partido(nombre_col: str) -> str | None:
    t = _sin_acentos(nombre_col).upper().replace("_", " ")
    for alias, tok in ALIAS.items():
        if alias in t:
            return tok
    for tok in PARTIDOS_TOKEN:
        if re.search(rf"\b{tok}\b", t):
            return tok
    return None


def matriz_ciclo(ds: DatasetINE) -> pd.DataFrame:
    """Una fila por distrito: votos sumados por token de partido + total."""
    df = ds.df.copy()
    df["ID_ENTIDAD"] = df["ID_ENTIDAD"].astype(str).str.split(".").str[0].str.zfill(2)
    df["ID_DISTRITO"] = pd.to_numeric(df["ID_DISTRITO"], errors="coerce")
    df = df[df["ID_DISTRITO"].notna()].reset_index(drop=True)
    df["ID_DISTRITO"] = df["ID_DISTRITO"].astype(int)
    nombres = (dict(zip(df["ID_ENTIDAD"], df["ENTIDAD"].astype(str)))
               if "ENTIDAD" in df.columns else {})
    salidas = {"ID_ENTIDAD": df["ID_ENTIDAD"], "ID_DISTRITO": df["ID_DISTRITO"]}
    for tok in PARTIDOS_TOKEN:
        cols = [c for c in ds.partidos if token_partido(c) == tok]
        if cols:
            salidas[tok] = df[cols].sum(axis=1)
    m = pd.DataFrame(salidas)
    m["NULOS_ETC"] = df[ds.especiales].sum(axis=1) if ds.especiales else 0
    tokens = [c for c in m.columns if c in PARTIDOS_TOKEN]
    m["TOTAL_CALC"] = m[tokens].sum(axis=1) + m["NULOS_ETC"]
    m = m.groupby(["ID_ENTIDAD", "ID_DISTRITO"], as_index=False).sum(numeric_only=True)
    m["ENTIDAD"] = m["ID_ENTIDAD"].map(nombres)
    m["CLAVE"] = m["ID_ENTIDAD"] + "-" + m["ID_DISTRITO"].astype(str).str.zfill(3)
    return m


class ComparadorINE:
    def __init__(self, prev: DatasetINE, nuevo: DatasetINE,
                 bloques: dict[str, str] | None = None):
        self.prev, self.nuevo = prev, nuevo
        self.bloques = dict(BLOQUE_DE_PARTIDO if bloques is None else bloques)
        self.m_prev, self.m_nuevo = matriz_ciclo(prev), matriz_ciclo(nuevo)
        self.fuera_prev = set(self.m_prev["CLAVE"]) - set(self.m_nuevo["CLAVE"])
        self.fuera_nuevo = set(self.m_nuevo["CLAVE"]) - set(self.m_prev["CLAVE"])
        self.no_mapeados = sorted({c for ds in (prev, nuevo) for c in ds.partidos
                                   if token_partido(c) is None})
        self.nombres = {**{r["ID_ENTIDAD"]: r.get("ENTIDAD") for _, r in self.m_prev.iterrows()},
                        **{r["ID_ENTIDAD"]: r.get("ENTIDAD") for _, r in self.m_nuevo.iterrows()}}
        self._tabla = None

    def _fusionar_matrices(self) -> pd.DataFrame:
        """Empareja distritos. ComparadorMapeado (Etapa 2) sobreescribe esto."""
        self._m_prev_ali = self.m_prev
        return self.m_prev.merge(self.m_nuevo, on="CLAVE", how="outer",
                                 suffixes=("_A", "_B"), indicator=True)

    @staticmethod
    def _rellenar(ambos: pd.DataFrame, tokens: list[str]):
        for t in tokens:
            for s in ("_A", "_B"):
                col = f"{t}{s}"
                ambos[col] = ambos[col].fillna(0) if col in ambos else 0.0

    def _aplicar_bloques(self, df: pd.DataFrame, sufijo: str, tokens: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for t in tokens:
            b = self.bloques.get(t, "OTROS")
            v = df[f"{t}{sufijo}"].fillna(0)
            out[b] = out[b] + v if b in out.columns else v
        return out

    @staticmethod
    def _ganadores(bloques_df: pd.DataFrame, total: pd.Series) -> pd.DataFrame:
        gan, p1, p2 = [], [], []
        for idx, r in bloques_df.iterrows():
            o = r.sort_values(ascending=False)
            t = total.loc[idx]
            gan.append(o.index[0] if o.iloc[0] > 0 else None)
            p1.append(100 * o.iloc[0] / t if t else np.nan)
            p2.append(100 * o.iloc[1] / t if t else np.nan)
        g = pd.DataFrame({"GANADOR": gan, "PCT_GAN": p1, "PCT_SEG": p2})
        g["MARGEN_PP"] = g["PCT_GAN"] - g["PCT_SEG"]
        return g

    def tabla_distritos(self) -> pd.DataFrame:
        if self._tabla is not None:
            return self._tabla
        ca, cb = self.prev.ciclo, self.nuevo.ciclo
        fusion = self._fusionar_matrices()
        ambos = fusion[fusion["_merge"] == "both"].reset_index(drop=True).drop(columns="_merge")
        tokens = sorted({t for t in PARTIDOS_TOKEN
                         if f"{t}_A" in ambos.columns or f"{t}_B" in ambos.columns})
        self._rellenar(ambos, tokens)
        partes = ambos["CLAVE"].str.split("-", expand=True)
        tabla = pd.DataFrame({"ID_ENTIDAD": partes[0], "ID_DISTRITO": partes[1].astype(int)})
        tabla["ENTIDAD"] = tabla["ID_ENTIDAD"].map(self.nombres)
        totA = ambos["TOTAL_CALC_A"].replace(0, np.nan)
        totB = ambos["TOTAL_CALC_B"].replace(0, np.nan)
        vol = pd.Series(0.0, index=ambos.index)
        for t in tokens:
            dpp = 100 * ambos[f"{t}_B"] / totB - 100 * ambos[f"{t}_A"] / totA
            dpp = dpp.replace([np.inf, -np.inf], np.nan)
            tabla[f"DPP_{t}"] = dpp.round(2)
            vol = vol + dpp.abs().fillna(0)
        tabla["VOLATILIDAD_PP"] = (0.5 * vol).round(2)
        gA = self._ganadores(self._aplicar_bloques(ambos, "_A", tokens), ambos["TOTAL_CALC_A"])
        gB = self._ganadores(self._aplicar_bloques(ambos, "_B", tokens), ambos["TOTAL_CALC_B"])
        tabla[f"GANADOR_BLOQUE_{ca}"] = gA["GANADOR"].values
        tabla[f"PCT_GANADOR_{ca}"] = gA["PCT_GAN"].round(2).values
        tabla[f"GANADOR_BLOQUE_{cb}"] = gB["GANADOR"].values
        tabla[f"PCT_GANADOR_{cb}"] = gB["PCT_GAN"].round(2).values
        tabla["CAMBIO_GANADOR"] = (gA["GANADOR"].fillna("") != gB["GANADOR"].fillna("")).values
        swing = []
        for i in range(len(ambos)):
            g = gB.iloc[i]["GANADOR"]
            pA = 100 * (self._aplicar_bloques(ambos, "_A", tokens).iloc[i].get(g, 0)
                        / ambos.iloc[i]["TOTAL_CALC_A"]) if g else np.nan
            swing.append(gB.iloc[i]["PCT_GAN"] - pA)
        tabla["SWING_PP"] = np.round(swing, 2)
        self._tokens, self._tabla = tokens, tabla
        return tabla

    def cambios_ganador(self) -> pd.DataFrame:
        ca, cb = self.prev.ciclo, self.nuevo.ciclo
        t = self.tabla_distritos()
        cols = ["ID_ENTIDAD", "ID_DISTRITO", "ENTIDAD", f"GANADOR_BLOQUE_{ca}",
                f"GANADOR_BLOQUE_{cb}", "SWING_PP"]
        return t[t["CAMBIO_GANADOR"]][cols].sort_values(["ID_ENTIDAD", "ID_DISTRITO"])

    def cambio_por_partido(self) -> pd.DataFrame:
        ca, cb = self.prev.ciclo, self.nuevo.ciclo
        fusion = getattr(self, "_m_prev_ali", self.m_prev).merge(
            self.m_nuevo, on="CLAVE", how="inner", suffixes=("_A", "_B"))
        tokens = self._tokens or PARTIDOS_TOKEN
        self._rellenar(fusion, tokens)
        piezas = []
        for t in tokens:
            tmp = fusion[["CLAVE", "ID_ENTIDAD_A", "ID_DISTRITO_A"]].copy()
            tmp["PARTIDO"] = t
            tmp[f"VOTOS_{ca}"] = fusion[f"{t}_A"]
            tmp[f"VOTOS_{cb}"] = fusion[f"{t}_B"]
            tmp[f"PCT_{ca}"] = (100 * fusion[f"{t}_A"] / fusion["TOTAL_CALC_A"]).round(2)
            tmp[f"PCT_{cb}"] = (100 * fusion[f"{t}_B"] / fusion["TOTAL_CALC_B"]).round(2)
            tmp[f"DPP_{ca}_{cb}"] = (tmp[f"PCT_{cb}"] - tmp[f"PCT_{ca}"]).round(2)
            piezas.append(tmp)
        return (pd.concat(piezas, ignore_index=True)
                .rename(columns={"ID_ENTIDAD_A": "ID_ENTIDAD", "ID_DISTRITO_A": "ID_DISTRITO"}))

    def resumen_nacional(self) -> dict[str, pd.DataFrame]:
        ca, cb = self.prev.ciclo, self.nuevo.ciclo
        salidas = {}
        for clave, agrupar_bloques in (("partidos", False), ("bloques", True)):
            idx = sorted({t for t in PARTIDOS_TOKEN
                          if t in self.m_prev.columns or t in self.m_nuevo.columns})
            if agrupar_bloques:
                idx = sorted({self.bloques.get(t, "OTROS") for t in idx})
            salidas[clave] = pd.DataFrame(index=idx)
            for ciclo, m in ((ca, self.m_prev), (cb, self.m_nuevo)):
                tokens_c = [t for t in PARTIDOS_TOKEN if t in m.columns]
                if agrupar_bloques and tokens_c:
                    agr = pd.concat([m[t].rename(self.bloques.get(t, "OTROS"))
                                     for t in tokens_c], axis=1)
                    agr = agr.T.groupby(level=0).sum().T
                else:
                    agr = m[tokens_c]
                total_nac = m["TOTAL_CALC"].sum()
                salidas[clave][f"VOTOS_{ciclo}"] = agr.sum().reindex(idx).fillna(0).astype("int64")
                salidas[clave][f"PCT_{ciclo}"] = (100 * salidas[clave][f"VOTOS_{ciclo}"]
                                                  / total_nac).round(2)
                sub = agr[agr.sum(axis=1) > 0]
                esc = sub.idxmax(axis=1).value_counts()
                salidas[clave][f"ESCANOS_{ciclo}"] = esc.reindex(idx).fillna(0).astype(int)
            salidas[clave] = salidas[clave].sort_values(f"VOTOS_{cb}", ascending=False)
        return salidas

    def top_volatiles(self, n: int = 15) -> pd.DataFrame:
        t = self.tabla_distritos()
        top = t.nlargest(n, "VOLATILIDAD_PP").copy()
        top["DISTRITO"] = (top["ENTIDAD"].fillna(top["ID_ENTIDAD"]) + " D"
                           + top["ID_DISTRITO"].astype(str).str.zfill(3))
        return top

    def exportar(self, ruta) -> Path:
        ruta = Path(ruta)
        ca, cb = self.prev.ciclo, self.nuevo.ciclo
        notas = pd.DataFrame({"Nota": [
            "GANADOR por BLOQUE de coalición (el INE reparte votos de coalición entre miembros).",
            f"VOLATILIDAD_PP = índice de Pedersen: 0.5·Σ|Δpp| ({ca}→{cb}).",
            "SWING_PP = variación del bloque ganador del ciclo nuevo vs su voto previo.",
            f"Partidos no reconocidos: {self.no_mapeados or 'ninguno'}",
            f"Fuentes: {self.prev.ruta.name}, {self.nuevo.ruta.name}."]})
        with pd.ExcelWriter(ruta, engine="openpyxl") as xl:
            self.tabla_distritos().to_excel(xl, sheet_name="Distritos", index=False)
            self.cambio_por_partido().to_excel(xl, sheet_name="Cambio por partido", index=False)
            self.cambios_ganador().to_excel(xl, sheet_name="Cambios de ganador", index=False)
            res = self.resumen_nacional()
            res["partidos"].to_excel(xl, sheet_name="Nacional partidos")
            res["bloques"].to_excel(xl, sheet_name="Nacional bloques")
            notas.to_excel(xl, sheet_name="Notas", index=False)
        return ruta
        if __name__ == "__main__":  # python analisis_comparativo.py 2021.csv 2024.csv
    import argparse
    from pathlib import Path

    from dataset_ine import DatasetINE

    ap = argparse.ArgumentParser(description="Comparativo INE entre ciclos")
    ap.add_argument("prev")
    ap.add_argument("nuevo")
    ap.add_argument("--ciclo-prev", default="2021")
    ap.add_argument("--ciclo-nuevo", default="2024")
    ap.add_argument("--salida", default="comparativo.xlsx")
    ap.add_argument("--graficas", nargs="?", const="graficas", default=None)
    a = ap.parse_args()
    comp = ComparadorINE(DatasetINE(a.prev, ciclo=a.ciclo_prev),
                         DatasetINE(a.nuevo, ciclo=a.ciclo_nuevo))
    res = comp.resumen_nacional()
    print(f"📊 Nacional por partido\n{res['partidos'].to_string()}")
    print(f"\n🏛️  Distritos por bloque\n{res['bloques'].to_string()}")
    t = comp.tabla_distritos()
    print(f"\n🔁 Cambios de bloque: {int(t['CAMBIO_GANADOR'].sum())} de {len(t)}")
    print(f"\n✅ Exportado → {comp.exportar(a.salida).resolve()}")
    if a.graficas:
        try:
            from graficas_ine import wall_map, wall_map_html
            Path(a.graficas).mkdir(exist_ok=True)
            cb = a.ciclo_nuevo
            print("🖼️ ", wall_map(t, f"GANADOR_BLOQUE_{cb}", f"Ganador por bloque — {cb}",
                                  f"{a.graficas}/wallmap_{cb}.png", "Fuente: cómputos INE"))
            wall_map_html(t, f"GANADOR_BLOQUE_{cb}", f"Ganador por bloque — {cb}",
                          f"{a.graficas}/wallmap_{cb}.html")
            print(f"🌐 {a.graficas}/wallmap_{cb}.html")
        except ImportError:
            print("⚠️ pip install matplotlib plotly")
