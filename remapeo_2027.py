"""remapeo_2027.py — Empareja distritos entre distritaciones (DIRECTO/CABECERA/MANUAL/SIN_PAR)."""
from __future__ import annotations

import argparse
import difflib, re, unicodedata
from pathlib import Path

import pandas as pd

from analisis_comparativo import ComparadorINE
from dataset_ine import DatasetINE

UMBRAL = 0.72


def _norm(t) -> str:
    t = unicodedata.normalize("NFD", str(t or ""))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").upper()
    t = re.sub(r"CABECERA|DISTRITO\s*(FEDERAL)?|NUM\.?|\bDE\b|\bDEL\b|\bLA\b|\bEL\b|"
               r"CIUDAD|MUNICIPIO|[.,:;/\-_]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _similitud(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()


def _col_cabecera(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "CABECERA" in str(c).upper():
            return c
    for c in ("NOMBRE_DISTRITO", "DISTRITO"):
        if c in df.columns and df[c].astype(str).str.contains(r"[A-ZÁÉÍÓÚÑ]", na=False).any():
            return c
    return None


def _pares(ds: DatasetINE) -> pd.DataFrame:
    df = ds.df.copy()
    df["ID_ENTIDAD"] = df["ID_ENTIDAD"].astype(str).str.split(".").str[0].str.zfill(2)
    df["ID_DISTRITO"] = pd.to_numeric(df["ID_DISTRITO"], errors="coerce")
    df = df[df["ID_DISTRITO"].notna()].copy()
    df["ID_DISTRITO"] = df["ID_DISTRITO"].astype(int)
    col = _col_cabecera(df)
    if col is None:
        raise ValueError(f"{ds.ruta.name}: sin columna de cabecera. Usa un mapeo manual (--manual).")
    df["CABECERA"] = df[col].astype(str)
    cols = ["ID_ENTIDAD", "ID_DISTRITO", "CABECERA"] + (
        ["ENTIDAD"] if "ENTIDAD" in df.columns else [])
    return df[cols]


def _leer_manual(ruta):
    if not ruta:
        return pd.DataFrame(columns=["ID_ENTIDAD", "ID_DISTRITO_NUEVO", "ID_DISTRITO_PREV"])
    m = pd.read_csv(ruta, dtype=str)
    m.columns = [c.strip().upper() for c in m.columns]
    if not {"ID_ENTIDAD", "ID_DISTRITO_NUEVO"}.issubset(m.columns):
        raise ValueError("El mapeo manual requiere: ID_ENTIDAD, ID_DISTRITO_NUEVO "
                         "(+ ID_DISTRITO_PREV; vacío = SIN_PAR explícito).")
    return m


def construir_mapeo(ds_prev: DatasetINE, ds_nuevo: DatasetINE,
                    ruta_manual=None, umbral: float = UMBRAL) -> pd.DataFrame:
    nn = _pares(ds_nuevo).rename(columns={"ID_DISTRITO": "ID_DISTRITO_NUEVO",
                                          "CABECERA": "CABECERA_NUEVA"})
    pp = _pares(ds_prev).rename(columns={"ID_DISTRITO": "ID_DISTRITO_PREV",
                                         "CABECERA": "CABECERA_PREV"})
    manual = _leer_manual(ruta_manual)
    pp_x = pp.drop(columns=["ENTIDAD"], errors="ignore")

    cruz = nn.merge(pp_x, on="ID_ENTIDAD", how="inner")
    cruz["SCORE"] = [_similitud(a, b) for a, b in
                     zip(cruz["CABECERA_NUEVA"], cruz["CABECERA_PREV"])]

    asign: dict[tuple, tuple] = {}
    usados_prev: set[tuple] = set()

    for _, r in cruz[(cruz["ID_DISTRITO_NUEVO"] == cruz["ID_DISTRITO_PREV"])
                     & (cruz["SCORE"] >= 0.90)].iterrows():
        k = (r["ID_ENTIDAD"], int(r["ID_DISTRITO_NUEVO"]))
        asign[k] = (int(r["ID_DISTRITO_PREV"]), "DIRECTO", float(r["SCORE"]))
        usados_prev.add((r["ID_ENTIDAD"], int(r["ID_DISTRITO_PREV"])))

    for _, r in cruz[cruz["SCORE"] >= umbral].sort_values("SCORE", ascending=False).iterrows():
        k = (r["ID_ENTIDAD"], int(r["ID_DISTRITO_NUEVO"]))
        kp = (r["ID_ENTIDAD"], int(r["ID_DISTRITO_PREV"]))
        if k not in asign and kp not in usados_prev:
            asign[k] = (kp[1], "CABECERA", float(r["SCORE"]))
            usados_prev.add(kp)

    for _, r in manual.iterrows():
        k = (str(r["ID_ENTIDAD"]).zfill(2), int(r["ID_DISTRITO_NUEVO"]))
        if pd.notna(r.get("ID_DISTRITO_PREV")):
            pv = int(r["ID_DISTRITO_PREV"])
            for kk in [kk for kk, (v, _, _) in asign.items()
                       if kk[0] == k[0] and v == pv and kk != k]:
                del asign[kk]
            asign[k] = (pv, "MANUAL", 1.0)
            usados_prev.add((k[0], pv))
        else:
            asign.pop(k, None)

    cab_prev = {(r["ID_ENTIDAD"], int(r["ID_DISTRITO_PREV"])): r["CABECERA_PREV"]
                for _, r in pp.iterrows()}
    filas = []
    for _, r in nn.iterrows():
        k = (r["ID_ENTIDAD"], int(r["ID_DISTRITO_NUEVO"]))
        pv, met, sc = asign.get(k, (None, "SIN_PAR", 0.0))
        filas.append({"ID_ENTIDAD": r["ID_ENTIDAD"], "ENTIDAD": r.get("ENTIDAD"),
                      "ID_DISTRITO": int(r["ID_DISTRITO_NUEVO"]),
                      "CABECERA_NUEVA": r["CABECERA_NUEVA"], "ID_DISTRITO_PREV": pv,
                      "CABECERA_PREV": cab_prev.get((r["ID_ENTIDAD"], pv)) if pv is not None else None,
                      "METODO": met, "SCORE": round(sc, 3)})
    mapeo = pd.DataFrame(filas)
    d = mapeo.dropna(subset=["ID_DISTRITO_PREV"]).groupby(
        ["ID_ENTIDAD", "ID_DISTRITO_PREV"]).size()
    if (d > 1).any():
        print("   ⚠️ Varios distritos nuevos apuntan al mismo previo (fusión): revisa el mapeo.")
    return mapeo


class ComparadorMapeado(ComparadorINE):
    """Comparador que empareja distritos mediante un mapeo de redistritación."""

    def __init__(self, prev: DatasetINE, nuevo: DatasetINE, mapeo: pd.DataFrame,
                 bloques: dict | None = None):
        super().__init__(prev, nuevo, bloques)
        m = mapeo.reset_index(drop=True).copy()
        m["ID_ENTIDAD"] = m["ID_ENTIDAD"].astype(str).str.split(".").str[0].str.zfill(2)
        m["ID_DISTRITO"] = pd.to_numeric(m["ID_DISTRITO"], errors="coerce").astype("Int64")
        m["ID_DISTRITO_PREV"] = pd.to_numeric(m.get("ID_DISTRITO_PREV"),
                                              errors="coerce").astype("Int64")
        self.mapeo = m
        sinpar = m[m["METODO"] == "SIN_PAR"]
        self.fuera_nuevo = {(r["ID_ENTIDAD"], int(r["ID_DISTRITO"]))
                            for _, r in sinpar.iterrows()}
        usados = {(r["ID_ENTIDAD"], int(r["ID_DISTRITO_PREV"]))
                  for _, r in m.dropna(subset=["ID_DISTRITO_PREV"]).iterrows()}
        self.fuera_prev = {c for c in self.m_prev["CLAVE"]
                           if (c.split("-")[0], int(c.split("-")[1])) not in usados}

    def _fusionar_matrices(self) -> pd.DataFrame:
        m = self.mapeo[self.mapeo["ID_DISTRITO_PREV"].notna()].copy()
        m["CLAVE"] = (m["ID_ENTIDAD"] + "-" + m["ID_DISTRITO_PREV"].astype(int)
                      .astype(str).str.zfill(3))
        m["CLAVE_NUEVA"] = (m["ID_ENTIDAD"] + "-" + m["ID_DISTRITO"].astype(int)
                            .astype(str).str.zfill(3))
        prev = self.m_prev.merge(m[["CLAVE", "CLAVE_NUEVA"]], on="CLAVE", how="left")
        prev["CLAVE"] = prev["CLAVE_NUEVA"].fillna(prev["CLAVE"])
        self._m_prev_ali = prev.drop(columns="CLAVE_NUEVA")
        return self._m_prev_ali.merge(self.m_nuevo, on="CLAVE", how="outer",
                                      suffixes=("_A", "_B"), indicator=True)


def previos_sin_uso(mapeo: pd.DataFrame, ds_prev: DatasetINE) -> list[str]:
    p = _pares(ds_prev)
    usados = {(r["ID_ENTIDAD"], int(r["ID_DISTRITO_PREV"]))
              for _, r in mapeo.dropna(subset=["ID_DISTRITO_PREV"]).iterrows()}
    return [f"{r['ID_ENTIDAD']} D{int(r['ID_DISTRITO']):03d} ({r['CABECERA']})"
            for _, r in p.iterrows()
            if (r["ID_ENTIDAD"], int(r["ID_DISTRITO"])) not in usados]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mapeo de redistritación entre ciclos INE")
    ap.add_argument("dataset_nuevo")
    ap.add_argument("--prev", required=True)
    ap.add_argument("--ciclo-nuevo", default="2027")
    ap.add_argument("--ciclo-prev", default="2024")
    ap.add_argument("--manual")
    ap.add_argument("--umbral", type=float, default=UMBRAL)
    ap.add_argument("--salida", default="mapeo_distritos.csv")
    ap.add_argument("--comparar")
    a = ap.parse_args()
    ds_n = DatasetINE(a.dataset_nuevo, ciclo=a.ciclo_nuevo)
    ds_p = DatasetINE(a.prev, ciclo=a.ciclo_prev)
    mapeo = construir_mapeo(ds_p, ds_n, a.manual, a.umbral)
    mapeo.to_csv(a.salida, index=False, encoding="utf-8-sig")
    c = mapeo["METODO"].value_counts()
    print(f"🗺️  Mapeo {a.ciclo_prev} → {a.ciclo_nuevo}: {len(mapeo)} distritos nuevos")
    print(f"   DIRECTO={c.get('DIRECTO', 0)}  CABECERA={c.get('CABECERA', 0)}  "
          f"MANUAL={c.get('MANUAL', 0)}  SIN_PAR={c.get('SIN_PAR', 0)}")
    huerfanos = previos_sin_uso(mapeo, ds_p)
    if huerfanos:
        print(f"   ℹ️  {len(huerfanos)} distritos de {a.ciclo_prev} sin sucesor: "
              + ", ".join(huerfanos[:10]))
    print(f"✅ Mapeo → {Path(a.salida).resolve()} (revísalo; corrige y re-ejecuta con --manual)")
    if a.comparar:
        ComparadorMapeado(ds_p, ds_n, mapeo).exportar(a.comparar)
        print(f"✅ Comparativo → {Path(a.comparar).resolve()}")
