"""dataset_ine.py — Carga y consulta de cómputos distritales del INE (CSV/XLSX/ZIP)."""
from __future__ import annotations

import csv, io, re, unicodedata, zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

NO_PARTIDO = re.compile(
    r"^(UNNAMED|_COL|NAN)$|ID_|ENTIDAD|DISTRITO|CABECERA|SECCION|CASILLA|"
    r"CONTABILIZ|ACTA|EMISION|FECHA|HORA|OBSERVAC|ESTATUS|TOTAL|LISTA|"
    r"PORCENT|PCT|UNIDAD|MUNICIPIO|SEDE|%", re.I)
ESPECIALES = re.compile(r"CIUDADANOS|NO_REGISTRAD|NULOS", re.I)
COALICION = re.compile(r"^C_")


def _quitar_acentos(t) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(t))
                   if unicodedata.category(c) != "Mn")


class DatasetINE:
    def __init__(self, ruta, ciclo: str = "2024", incluir_coaliciones: bool = False):
        self.ruta, self.ciclo, self.incluir_coaliciones = Path(ruta), ciclo, incluir_coaliciones
        crudo = self._leer(self.ruta)
        i_enc = self._localizar_encabezado(crudo)
        self.df = self._agregar_distritos(self._normalizar(crudo, i_enc))
        self.col_total = self._detectar_total()
        self.especiales = [c for c in self.df.columns if ESPECIALES.search(c)]
        self.partidos = self._detectar_partidos()

    def _leer(self, ruta: Path) -> pd.DataFrame:
        sufijo = ruta.suffix.lower()
        if sufijo == ".zip":
            with zipfile.ZipFile(ruta) as z:
                miembros = [n for n in z.namelist()
                            if n.lower().endswith((".csv", ".xlsx", ".xls"))]
                if not miembros:
                    raise ValueError("El ZIP no contiene CSV/XLSX.")
                miembro = max(miembros, key=lambda n: z.getinfo(n).file_size)
                sufijo, data = Path(miembro).suffix.lower(), z.read(miembro)
        else:
            data = ruta.read_bytes()
        if sufijo in (".xlsx", ".xls"):
            return pd.read_excel(io.BytesIO(data), header=None, dtype=str)
        texto = None
        for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                texto = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        primera = next(l for l in texto.splitlines() if l.strip())
        try:
            delim = csv.Sniffer().sniff(primera, delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ";" if primera.count(";") > primera.count(",") else ","
        return pd.read_csv(io.StringIO(texto), sep=delim, header=None,
                           dtype=str, low_memory=False)

    def _localizar_encabezado(self, df: pd.DataFrame) -> int:
        for i in range(min(15, len(df))):
            fila = df.iloc[i].astype(str).map(_quitar_acentos).str.upper().str.strip()
            unidos = " ".join(fila)
            if "ENTIDAD" in unidos and "DISTRITO" in unidos:
                return i
        raise ValueError("No se encontró el encabezado (columnas ENTIDAD/DISTRITO).")

    def _normalizar(self, crudo: pd.DataFrame, i_enc: int) -> pd.DataFrame:
        encabezado = crudo.iloc[i_enc].map(
            lambda x: re.sub(r"\s+", "_", _quitar_acentos(x).strip().upper()))
        df = crudo.iloc[i_enc + 1:].copy()
        df.columns = [c if c and c.upper() != "NAN" and not c.upper().startswith("UNNAMED")
                      else f"_COL{j}" for j, c in enumerate(encabezado)]
        df = df.dropna(how="all").reset_index(drop=True)
        cols = pd.Series(df.columns, dtype=object)
        for c in cols[cols.duplicated()].unique():
            mascara = (cols == c).to_numpy()
            cols[mascara] = [f"{c}_{n}" for n in range(1, mascara.sum() + 1)]
        df.columns = cols
        # Texto: identificadores/nombres. Numérico: votos, especiales y TOTAL.
        es_texto = lambda c: (bool(NO_PARTIDO.search(c)) and not ESPECIALES.search(c)
                              and "TOTAL" not in c.upper())
        for c in df.columns:
            if es_texto(c):
                df[c] = df[c].astype(str).str.strip()
            else:
                limpio = (df[c].astype(str).str.replace(",", "", regex=False).str.strip()
                          .replace({"": None, "nan": None, "None": None}))
                df[c] = pd.to_numeric(limpio, errors="coerce").fillna(0)
        for col_id in ("ID_DISTRITO", "ID_ENTIDAD"):
            if col_id in df.columns:
                df = df[pd.to_numeric(df[col_id], errors="coerce").notna()]
        return df.reset_index(drop=True)

    def _agregar_distritos(self, df: pd.DataFrame) -> pd.DataFrame:
        claves = [c for c in ("ID_ENTIDAD", "ENTIDAD", "ID_DISTRITO", "DISTRITO",
                              "CABECERA_DISTRITAL", "CABECERA") if c in df.columns]
        numericas = [c for c in df.columns
                     if c not in claves and pd.api.types.is_numeric_dtype(df[c])]
        return df.groupby(claves, dropna=False, as_index=False)[numericas].sum()

    def _detectar_total(self) -> str | None:
        return next((c for c in self.df.columns
                     if re.search(r"TOTAL_VOTOS|TOTAL_VOTACION", c, re.I)), None)

    def _detectar_partidos(self) -> list[str]:
        return [c for c in self.df.columns
                if not NO_PARTIDO.search(c) and not ESPECIALES.search(c)
                and not (COALICION.match(c) and not self.incluir_coaliciones)
                and self.df[c].sum() > 0]

    def _fila_ganador(self, fila) -> dict:
        votos = {p: int(fila[p]) for p in self.partidos}
        orden = sorted(votos.items(), key=lambda kv: kv[1], reverse=True)
        especiales = {e: int(fila[e]) for e in self.especiales}
        total = (int(fila[self.col_total]) if self.col_total
                 else int(sum(votos.values()) + sum(especiales.values())))
        pct = lambda v: round(100 * v / total, 2) if total else 0.0
        return {"ciclo": self.ciclo,
                "id_entidad": str(fila.get("ID_ENTIDAD", "")).split(".")[0].zfill(2) or None,
                "entidad": str(fila.get("ENTIDAD", "")) or None,
                "distrito": int(fila["ID_DISTRITO"]) if pd.notna(fila.get("ID_DISTRITO")) else None,
                "cabecera": str(fila.get("CABECERA_DISTRITAL", "")) or None,
                "votos_partidos": orden, "especiales": especiales, "total_votos": total,
                "nulos": especiales.get("NULOS", 0),
                "ganador": orden[0][0], "votos_ganador": orden[0][1],
                "pct_ganador": pct(orden[0][1]),
                "segundo": orden[1][0] if len(orden) > 1 else None,
                "margen": orden[0][1] - orden[1][1] if len(orden) > 1 else None,
                "fuente": f"Dataset oficial INE ({self.ruta.name})"}

    def consultar(self, entidad, distrito: int | None = None) -> dict:
        df = self.df
        if str(entidad).isdigit():
            objetivo = str(int(entidad)).zfill(2)
            sub = df[df["ID_ENTIDAD"].map(lambda v: str(v).split(".")[0].zfill(2) == objetivo)]
        else:
            objetivo = _quitar_acentos(entidad).upper()
            sub = df[df["ENTIDAD"].map(lambda x: objetivo in _quitar_acentos(x).upper())]
        if sub.empty:
            raise ValueError(f"Entidad no encontrada: {entidad!r}")
        disponibles = sorted(pd.to_numeric(sub["ID_DISTRITO"], errors="coerce")
                             .dropna().astype(int).unique()) if "ID_DISTRITO" in sub else []
        if distrito is not None:
            sub = sub[pd.to_numeric(sub["ID_DISTRITO"], errors="coerce") == int(distrito)]
            if sub.empty:
                raise ValueError(f"Distrito {distrito} no existe en {entidad}. "
                                 f"Disponibles: {disponibles}")
        elif len(sub) > 1:
            return {"entidad": str(sub.iloc[0]["ENTIDAD"]),
                    "aviso": f"{len(sub)} distritos; indica cuál. Disponibles: {disponibles}"}
        return self._fila_ganador(sub.iloc[0])

    def consultar_todos(self) -> pd.DataFrame:
        registros = []
        for _, fila in self.df.iterrows():
            r = self._fila_ganador(fila)
            reg = {"CICLO": r["ciclo"], "ID_ENTIDAD": r["id_entidad"], "ENTIDAD": r["entidad"],
                   "ID_DISTRITO": r["distrito"], "CABECERA": r["cabecera"],
                   "GANADOR": r["ganador"], "VOTOS_GANADOR": r["votos_ganador"],
                   "PCT_GANADOR": r["pct_ganador"], "SEGUNDO": r["segundo"],
                   "MARGEN": r["margen"], "TOTAL_VOTOS": r["total_votos"]}
            reg.update({k: v for k, v in r["especiales"].items()})
            reg.update(dict(r["votos_partidos"]))
            registros.append(reg)
        return (pd.DataFrame(registros)
                .sort_values(["ID_ENTIDAD", "ID_DISTRITO"]).reset_index(drop=True))

    def exportar(self, ruta) -> Path:
        ruta = Path(ruta)
        df = self.consultar_todos()
        if ruta.suffix.lower() == ".csv":
            df.to_csv(ruta, index=False, encoding="utf-8-sig")
        else:
            leyenda = pd.DataFrame({"Campo": ["GANADOR", "VOTOS_GANADOR", "PCT_GANADOR",
                                              "SEGUNDO", "MARGEN", "TOTAL_VOTOS", "<PARTIDO>"],
                                    "Descripción": ["Partido/coalición con más votos",
                                        "Votos del ganador", "% sobre el total del distrito",
                                        "Segundo lugar", "Diferencia 1º vs 2º",
                                        "Total de votos del distrito",
                                        "Una columna por partido detectado"]})
            with pd.ExcelWriter(ruta, engine="openpyxl") as xl:
                df.to_excel(xl, sheet_name="Distritos", index=False)
                leyenda.to_excel(xl, sheet_name="Leyenda", index=False)
        return ruta

    def resumen(self) -> str:
        return "\n".join([
            f"📊 Dataset INE — ciclo {self.ciclo} | {self.ruta.name}",
            f"   Distritos: {len(self.df)} | Entidades: {self.df['ENTIDAD'].nunique()}",
            f"   Partidos detectados: {', '.join(self.partidos) or '(ninguno)'}",
            f"   Votos especiales: {', '.join(self.especiales) or '(ninguno)'}"])


def descargar(url: str, destino=".") -> Path:
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    nombre = url.split("?")[0].rstrip("/").split("/")[-1] or "dataset_ine.zip"
    ruta = destino / nombre
    with requests.get(url, stream=True, timeout=120,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(ruta, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    if ruta.suffix.lower() == ".zip":
        with zipfile.ZipFile(ruta) as z:
            z.extractall(destino)
    return ruta


def formatear_distrito(res: dict) -> str:
    if "aviso" in res:
        return f"⚠️ {res['aviso']}"
    total = res["total_votos"]
    pct = lambda v: f"{100 * v / total:.1f}%" if total else "—"
    lineas = [f"## Distrito {res['distrito']} de {res['entidad']} — {res['ciclo']}",
              "| Partido / Coalición | Votos | % |", "|---|---:|---:|"]
    for nombre, votos in res["votos_partidos"]:
        lineas.append(f"| {nombre} | {votos:,.0f} | {pct(votos)} |")
    for nombre, votos in res["especiales"].items():
        lineas.append(f"| {nombre} | {votos:,.0f} | {pct(votos)} |")
    lineas += [f"- 🏆 Ganador: **{res['ganador']}** con {res['votos_ganador']:,.0f} votos "
               f"({res['pct_ganador']:.1f}%)",
               f"- 📚 Fuente: {res['fuente']} — consultado el {date.today():%d/%m/%Y}",
               "- ⚠️ Nota: los votos de coalición ya vienen repartidos entre partidos miembros."]
    return "\n".join(lineas)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Consulta cómputos distritales INE")
    ap.add_argument("ruta")
    ap.add_argument("--ciclo", default="2024")
    ap.add_argument("--entidad")
    ap.add_argument("--distrito", type=int)
    ap.add_argument("--salida")
    a = ap.parse_args()
    ds = DatasetINE(a.ruta, ciclo=a.ciclo)
    print(ds.resumen())
    if a.entidad:
        print("\n" + formatear_distrito(ds.consultar(a.entidad, a.distrito)))
    if a.salida:
        print(f"\n✅ Exportado → {ds.exportar(a.salida).resolve()}")
