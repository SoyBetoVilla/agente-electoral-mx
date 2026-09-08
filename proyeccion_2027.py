"""proyeccion_2027.py — Proyección probabilística: baseline INE + swing de encuestas
+ Monte Carlo con shocks jerárquicos. ESCENARIO estadístico, no pronóstico."""
from __future__ import annotations

import math, re, unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analisis_comparativo import BLOQUE_DE_PARTIDO, PARTIDOS_TOKEN, matriz_ciclo
from dataset_ine import DatasetINE

FECHA_ELECCION_DEFAULT = date(2027, 6, 6)
ENTIDADES = {"01": "Aguascalientes", "02": "Baja California", "03": "Baja California Sur",
    "04": "Campeche", "05": "Chiapas", "06": "Chihuahua", "07": "Ciudad de México",
    "08": "Coahuila", "09": "Colima", "10": "Durango", "11": "Guanajuato",
    "12": "Guerrero", "13": "Hidalgo", "14": "Jalisco", "15": "Estado de México",
    "16": "Michoacán", "17": "Morelos", "18": "Nayarit", "19": "Nuevo León",
    "20": "Oaxaca", "21": "Puebla", "22": "Querétaro", "23": "Quintana Roo",
    "24": "San Luis Potosí", "25": "Sinaloa", "26": "Sonora", "27": "Tabasco",
    "28": "Tamaulipas", "29": "Tlaxcala", "30": "Veracruz", "31": "Yucatán",
    "32": "Zacatecas"}


def _na(t) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(t or ""))
                   if unicodedata.category(c) != "Mn").upper().strip()


def _codigo_entidad(v) -> str | None:
    s = _na(v).replace(".0", "")
    if s.isdigit():
        return s.zfill(2) if 0 < int(s) <= 32 else None
    if not s:
        return None
    for cod, nombre in ENTIDADES.items():
        if _na(nombre) == s:
            return cod
    for cod, nombre in ENTIDADES.items():
        if _na(nombre) in s or s in _na(nombre):
            return cod
    return None


class Encuestas:
    def __init__(self, ruta, fecha_referencia: date | None = None,
                 half_life_dias: float = 45.0):
        self.fecha_ref = fecha_referencia or date.today()
        self.half_life = float(half_life_dias)
        self.df = self._leer(ruta)
        self._calcular_pesos()

    def _leer(self, ruta) -> pd.DataFrame:
        suf = getattr(ruta, "suffix", "")
        es_xlsx = bool(suf) and suf.lower() in (".xlsx", ".xls")
        if hasattr(ruta, "name") and not es_xlsx:
            es_xlsx = str(ruta.name).lower().endswith((".xlsx", ".xls"))
        df = pd.read_excel(ruta) if es_xlsx else pd.read_csv(ruta)
        df.columns = [re.sub(r"\s+", "_", _na(c)) for c in df.columns]
        ren = {}
        for c in df.columns:
            if "ID_DISTRITO" in c:
                ren[c] = "ID_DISTRITO"
            elif "ENTIDAD" in c:
                ren[c] = "ID_ENTIDAD"
            elif "FECHA" in c:
                ren[c] = "FECHA"
            elif "CASA" in c or "ENCUESTADORA" in c:
                ren[c] = "CASA"
            elif "AMBITO" in c or "NIVEL" in c:
                ren[c] = "AMBITO"
            elif c in ("MUESTRA", "N", "N_MUESTRA"):
                ren[c] = "MUESTRA"
            elif c in PARTIDOS_TOKEN or c in ("OTROS", "OTRAS", "INDEPENDIENTES"):
                ren[c] = c
        df = df.rename(columns=ren)
        df["FECHA"] = pd.to_datetime(df.get("FECHA"), errors="coerce", dayfirst=True)
        df["AMBITO"] = (df.get("AMBITO", "NACIONAL").astype(str).str.upper().str.strip()
                        .replace({"NACION": "NACIONAL", "ESTADO": "ESTATAL",
                                  "EDO": "ESTATAL", "DISTRITO": "DISTRITAL"}))
        if "ID_ENTIDAD" in df.columns:
            df["ID_ENTIDAD"] = df["ID_ENTIDAD"].map(_codigo_entidad)
        else:
            df["ID_ENTIDAD"] = None
        df["ID_DISTRITO"] = pd.to_numeric(df.get("ID_DISTRITO"), errors="coerce")
        df["MUESTRA"] = pd.to_numeric(df.get("MUESTRA"), errors="coerce").fillna(800)
        partidos = [c for c in df.columns if c in PARTIDOS_TOKEN or c == "OTROS"]
        if not partidos:
            raise ValueError("El CSV de encuestas no tiene columnas de partido.")
        df[partidos] = df[partidos].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
        suma = df[partidos].sum(axis=1)
        df = df[suma > 0].copy()
        df[partidos] = df[partidos].div(suma, axis=0) * 100
        self.partidos = partidos
        return df.reset_index(drop=True)

    def _calcular_pesos(self):
        delta = (pd.Timestamp(self.fecha_ref) - self.df["FECHA"]).dt.days.clip(lower=0)
        self.df["PESO"] = (np.sqrt(self.df["MUESTRA"])
                           * np.exp(-math.log(2) * delta / self.half_life))

    @staticmethod
    def _mediana_ponderada(v: np.ndarray, w: np.ndarray) -> float:
        orden = np.argsort(v)
        v, w = v[orden], w[orden]
        acum = np.cumsum(w)
        return float(v[np.searchsorted(acum, 0.5 * acum[-1])]) if acum[-1] > 0 else 0.0

    def agregado(self, ambito="NACIONAL", id_entidad=None, id_distrito=None):
        meta = {"n_encuestas": 0, "muestra_total": 0, "casas": []}
        sub = self.df[self.df["AMBITO"] == ambito]
        if ambito in ("ESTATAL", "DISTRITAL"):
            sub = sub[sub["ID_ENTIDAD"] == id_entidad]
        if ambito == "DISTRITAL":
            sub = sub[pd.to_numeric(sub["ID_DISTRITO"], errors="coerce") == int(id_distrito)]
        sub = sub[sub["PESO"] > 0]
        if sub.empty:
            return None, None, meta
        bloques = sorted({b for b in BLOQUE_DE_PARTIDO.values() if b != "OTROS"})
        v, w = [], []
        for _, fila in sub.iterrows():
            v.append({b: sum(fila.get(p, 0.0) for p in self.partidos
                             if BLOQUE_DE_PARTIDO.get(p) == b) for b in bloques})
            w.append(fila["PESO"])
        w = np.asarray(w, dtype=float)
        medias, sigmas = {}, {}
        n_ef = float(w.sum() ** 2 / (w ** 2).sum())
        for b in bloques:
            vals = np.asarray([r[b] for r in v], dtype=float)
            vivos = vals[vals > 0]
            if len(vivos) == 0:
                medias[b], sigmas[b] = 0.0, 0.0
                continue
            wv = w[vals > 0]
            mu = self._mediana_ponderada(vals, w)
            media_pond = (float(np.average(vivos, weights=wv))
                          if wv.sum() > 0 else float(vivos.mean()))
            disp = (float(np.sqrt(np.average((vivos - media_pond) ** 2, weights=wv)))
                    if len(vivos) > 1 and wv.sum() > 0 else 0.0)
            err = math.sqrt(max(media_pond * (100 - media_pond), 1) / max(n_ef, 50))
            medias[b] = round(mu, 2)
            sigmas[b] = round(max(math.hypot(disp, err), 1.5), 2)
        meta = {"n_encuestas": len(sub), "muestra_total": int(sub["MUESTRA"].sum()),
                "casas": sorted(sub["CASA"].dropna().astype(str).unique().tolist())}
        return medias, sigmas, meta

    def estatales(self) -> dict:
        sal = {}
        for cod in sorted(self.df.loc[self.df["AMBITO"] == "ESTATAL",
                                      "ID_ENTIDAD"].dropna().unique()):
            m, s, _ = self.agregado("ESTATAL", cod)
            if m:
                sal[cod] = (m, s)
        return sal

    def distritales(self) -> dict:
        sal = {}
        for _, r in self.df[self.df["AMBITO"] == "DISTRITAL"].iterrows():
            if pd.notna(r.get("ID_ENTIDAD")) and pd.notna(r.get("ID_DISTRITO")):
                clave = (r["ID_ENTIDAD"], int(r["ID_DISTRITO"]))
                m, s, _ = self.agregado("DISTRITAL", clave[0], clave[1])
                if m:
                    sal[clave] = (m, s)
        return sal


class Proyeccion2027:
    def __init__(self, ds_base: DatasetINE, encuestas=None, ds_prev: DatasetINE | None = None,
                 fecha_eleccion: date = FECHA_ELECCION_DEFAULT,
                 penalizacion_oficialismo: float = 0.0, bloque_gobierno="MORENA-PT-PVEM",
                 sigma_het_default: float = 4.0, sigma_min: float = 2.5,
                 escanos_en_juego: int = 300):
        self.ds_base, self.ds_prev, self.enc = ds_base, ds_prev, encuestas
        self.fecha_eleccion = fecha_eleccion
        self.penal, self.bloque_gob = float(penalizacion_oficialismo), bloque_gobierno
        self.sigma_het_default, self.sigma_min = sigma_het_default, sigma_min
        self.n_escanos, self.mayoria = escanos_en_juego, escanos_en_juego // 2 + 1
        self.bloques = sorted({b for b in BLOQUE_DE_PARTIDO.values() if b != "OTROS"})
        self._construir_base()
        self._construir_swings()
        self._construir_mu_sigma()

    def _construir_base(self):
        m = matriz_ciclo(self.ds_base)
        self._m_base = m
        V = pd.DataFrame(index=m.index)
        for b in self.bloques:
            toks = [t for t in PARTIDOS_TOKEN
                    if BLOQUE_DE_PARTIDO.get(t) == b and t in m.columns]
            V[b] = m[toks].sum(axis=1) if toks else 0.0
        den = V.sum(axis=1).replace(0, np.nan)
        self.pct_base = V.div(den, axis=0).fillna(0) * 100
        self.votos_bloque, self.den = V, den
        self.meta = m[["ID_ENTIDAD", "ID_DISTRITO", "ENTIDAD"]].copy()
        cb = self.ds_base.df.copy()
        cb["ID_ENTIDAD"] = cb["ID_ENTIDAD"].astype(str).str.split(".").str[0].str.zfill(2)
        cb["ID_DISTRITO"] = pd.to_numeric(cb["ID_DISTRITO"], errors="coerce")
        col_cb = next((c for c in cb.columns if "CABECERA" in str(c).upper()), None)
        if col_cb:
            mapa = dict(zip(zip(cb["ID_ENTIDAD"], cb["ID_DISTRITO"].dropna().astype(int)),
                            cb[col_cb].astype(str)))
            self.meta["CABECERA"] = [mapa.get((r["ID_ENTIDAD"], int(r["ID_DISTRITO"])), "")
                                     for _, r in self.meta.iterrows()]
        else:
            self.meta["CABECERA"] = ""
        self.meta["TOTAL_VOTOS_BASE"] = m["TOTAL_CALC"]

    def pct_bloque_de_local(self, V_local: pd.DataFrame) -> pd.Series:
        den = V_local.sum(axis=1).replace(0, np.nan)
        return (V_local.sum() / den.sum() * 100).fillna(0)

    def _construir_swings(self):
        self.swing = pd.DataFrame(0.0, index=self.meta.index, columns=self.bloques)
        self.sigma_enc = pd.DataFrame(0.0, index=self.meta.index, columns=self.bloques)
        self.fuente_swing = pd.Series("sin encuestas (status quo)", index=self.meta.index)
        if self.enc is None:
            return
        med_nac, sig_nac, meta_nac = self.enc.agregado("NACIONAL")
        self.meta_nac = meta_nac
        if med_nac:
            base_nac = self.pct_bloque_de_local(self.votos_bloque)
            for b in self.bloques:
                self.swing[b] = med_nac.get(b, 0.0) - base_nac.get(b, 0.0)
                self.sigma_enc[b] = sig_nac.get(b, 3.0)
            self.fuente_swing[:] = (f"nacional ({meta_nac['n_encuestas']} enc., "
                                    f"n={meta_nac['muestra_total']:,})")
        for cod, idx in self.meta.groupby("ID_ENTIDAD").indices.items():
            base_est = self.pct_bloque_de_local(self.votos_bloque.iloc[idx])
            for cod2, (med, sig) in self.enc.estatales().items():
                if cod2 == cod:
                    for b in self.bloques:
                        self.swing.loc[idx, b] = med.get(b, 0.0) - base_est.get(b, 0.0)
                        self.sigma_enc.loc[idx, b] = sig.get(b, 3.0)
                    self.fuente_swing.loc[idx] = f"estatal {cod} ({ENTIDADES.get(cod, cod)})"
        for (cod, dis), (med, sig) in self.enc.distritales().items():
            idx = self.meta.index[(self.meta["ID_ENTIDAD"] == cod)
                                  & (self.meta["ID_DISTRITO"] == dis)]
            if len(idx):
                base_f = self.pct_base.loc[idx].iloc[0]
                for b in self.bloques:
                    self.swing.loc[idx, b] = med.get(b, 0.0) - base_f.get(b, 0.0)
                    self.sigma_enc.loc[idx, b] = sig.get(b, 3.0)
                self.fuente_swing.loc[idx] = f"distrital {cod}-D{dis:03d}"

    def _construir_mu_sigma(self):
        mu = self.pct_base + self.swing
        if self.penal and self.bloque_gob in mu.columns:
            mu[self.bloque_gob] -= self.penal
        mu = mu.clip(lower=0.5, upper=99)
        self.medias = mu.div(mu.sum(axis=1), axis=0) * 100
        sigma_het = pd.Series(self.sigma_het_default, index=self.bloques)
        if self.ds_prev is not None:
            try:
                m_prev = matriz_ciclo(self.ds_prev)
                Vp = pd.DataFrame(index=m_prev.index)
                for b in self.bloques:
                    toks = [t for t in PARTIDOS_TOKEN
                            if BLOQUE_DE_PARTIDO.get(t) == b and t in m_prev.columns]
                    Vp[b] = m_prev[toks].sum(axis=1) if toks else 0.0
                pct_prev = Vp.div(Vp.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
                comunes = sorted(set(m_prev["CLAVE"]) & set(self._m_base["CLAVE"]))
                if comunes:
                    a_ = self.pct_base.set_index(self._m_base["CLAVE"]).loc[comunes]
                    b_ = pct_prev.set_index(m_prev["CLAVE"]).loc[comunes]
                    sw = a_ - b_
                    for b in self.bloques:
                        s = float(sw[b].std())
                        if np.isfinite(s) and s > 0:
                            sigma_het[b] = min(max(s, 2.0), 8.0)
            except Exception:
                pass
        self.sigma_het = sigma_het
        meses = max((self.fecha_eleccion - date.today()).days, 0) / 30.44
        self.sigma_tiempo = min(0.55 * math.sqrt(meses), 4.0)
        self.sigma = ((self.sigma_enc ** 2).add(
            sigma_het ** 2 + self.sigma_tiempo ** 2, axis=1)).pow(0.5)
        self.sigma = self.sigma.clip(lower=self.sigma_min)

    def simular(self, n: int = 10000, seed: int = 42,
                sigma_nacional: float = 0.0, sigma_estatal: float = 0.0,
                rho_shock: float = 0.0) -> dict:
        rng = np.random.default_rng(seed)
        mu = self.medias.to_numpy(dtype=np.float32)
        sg = self.sigma.to_numpy(dtype=np.float32)
        nD, B = mu.shape
        L = None
        if sigma_nacional > 0:
            if rho_shock > 0:
                R = np.full((B, B), rho_shock, dtype=np.float64)
                np.fill_diagonal(R, 1.0)
                L = np.linalg.cholesky(R).astype(np.float32)
            z = rng.standard_normal((n, B), dtype=np.float32)
            eps_nac = (z @ L.T if L is not None else z) * np.float32(sigma_nacional)
        else:
            eps_nac = 0.0
        if sigma_estatal > 0:
            cods = self.meta["ID_ENTIDAD"].astype(str).to_numpy()
            ents = np.unique(cods)
            idx = np.searchsorted(ents, cods)
            z = rng.standard_normal((n, len(ents), B), dtype=np.float32)
            if L is not None:
                z = z @ L.T
            eps_est = (z * np.float32(sigma_estatal))[:, idx, :]
        else:
            eps_est = 0.0
        var_shock = sigma_nacional ** 2 + sigma_estatal ** 2
        var_idio = np.clip(sg.astype(np.float64) ** 2 - var_shock, 0.25, None)
        eps_dis = (rng.standard_normal((n, nD, B), dtype=np.float32)
                   * np.sqrt(var_idio).astype(np.float32))
        X = mu[None, :, :] + eps_nac[:, None, :] + eps_est + eps_dis
        X = np.clip(X, 0, None)
        X /= X.sum(axis=2, keepdims=True)
        ganador = X.argmax(axis=2)
        p_win = (ganador[:, :, None] == np.arange(B)[None, None, :]).mean(axis=0)
        pct_esp = X.mean(axis=0)
        esc = np.zeros((n, B), dtype=np.int32)
        for b in range(B):
            esc[:, b] = (ganador == b).sum(axis=1)
        self.shock_params = {"sigma_nacional": sigma_nacional,
                             "sigma_estatal": sigma_estatal, "rho_shock": rho_shock}
        self.sim = {"n": n, "seed": seed,
            "p_win": pd.DataFrame(p_win * 100, index=self.medias.index, columns=self.bloques),
            "pct_esperado": pd.DataFrame(pct_esp * 100, index=self.medias.index,
                                         columns=self.bloques),
            "esc_matrix": esc,
            "escanos": pd.DataFrame({
                b: {"media": esc[:, i].mean(), "p5": np.percentile(esc[:, i], 5),
                    "p95": np.percentile(esc[:, i], 95),
                    "p_mayoria": (esc[:, i] >= self.mayoria).mean() * 100}
                for i, b in enumerate(self.bloques)}).T}
        return self.sim

    def tabla_proyecciones(self) -> pd.DataFrame:
        if not hasattr(self, "sim"):
            self.simular()
        pw, pe = self.sim["p_win"], self.sim["pct_esperado"]
        tabla = self.meta.copy()
        for b in self.bloques:
            tabla[f"PCT_{b}"] = pe[b].round(1)
            tabla[f"P_{b}"] = pw[b].round(1)
        tabla["GANADOR_PROBABLE"] = pw.idxmax(axis=1)
        tabla["P_GANADOR"] = pw.max(axis=1).round(1)
        tabla["CLASIFICACION"] = tabla["P_GANADOR"].map(self._categoria)
        tabla["FUENTE_SWING"] = self.fuente_swing
        return tabla.sort_values(["ID_ENTIDAD", "ID_DISTRITO"])

    @staticmethod
    def _categoria(p: float) -> str:
        if p >= 95: return "Seguro"
        if p >= 80: return "Inclinado"
        if p >= 65: return "Ligeramente inclinado"
        if p >= 55: return "Competitivo"
        return "Muy competitivo"

    def competitivos(self, n: int = 20) -> pd.DataFrame:
        t = self.tabla_proyecciones()
        return t.nsmallest(n, "P_GANADOR")[["ID_ENTIDAD", "ENTIDAD", "ID_DISTRITO",
                                            "CABECERA", "GANADOR_PROBABLE", "P_GANADOR",
                                            "CLASIFICACION"]]

    def formatear_proyeccion(self, entidad, distrito: int) -> str:
        t = self.tabla_proyecciones()
        cod = str(entidad).zfill(2) if str(entidad).isdigit() else _codigo_entidad(entidad)
        f = t[(t["ID_ENTIDAD"] == cod) & (t["ID_DISTRITO"] == int(distrito))]
        if f.empty:
            return f"⚠️ Sin proyección para distrito {distrito} de {entidad}."
        r, label = f.iloc[0], f.index[0]
        lineas = [f"## 🔮 Proyección 2027 — Distrito {int(r['ID_DISTRITO'])} de "
                  f"{r['ENTIDAD']} ({r['CABECERA'] or 's/ cabecera'})",
                  "| Bloque | % esperado | P(victoria) |", "|---|---:|---:|"]
        for b in sorted(self.bloques, key=lambda x: -r[f"PCT_{x}"]):
            lineas.append(f"| {b} | {r[f'PCT_{b}']}% | {r[f'P_{b}']}% |")
        lineas += [f"- 🏆 **Ganador probable: {r['GANADOR_PROBABLE']}** "
                   f"({r['P_GANADOR']}%) — {r['CLASIFICACION']}",
                   f"- 📡 Swing aplicado: {r['FUENTE_SWING']}",
                   f"- 🌪️ Incertidumbre (σ): "
                   f"{', '.join(f'{b}: {self.sigma.loc[label, b]:.1f}pp' for b in self.bloques)}",
                   "- ⚠️ Escenario estadístico (Monte Carlo), no pronóstico ni resultado oficial."]
        return "\n".join(lineas)

    def exportar(self, ruta) -> Path:
        if not hasattr(self, "sim"):
            self.simular()
        t = self.tabla_proyecciones()
        notas = pd.DataFrame({"Nota": [
            "ESCENARIO ESTADÍSTICO, no pronóstico ni resultado oficial.",
            "μ = %_base(distrito) + swing(encuesta); normalizado a 100% por bloques.",
            f"σ = √(σ_enc² + σ_het² + σ_tiempo²); piso {self.sigma_min} pp. "
            f"σ_het={dict(self.sigma_het.round(1))}; σ_tiempo={self.sigma_tiempo:.1f} pp.",
            f"P(b) = P de que el bloque b gane el distrito en {self.sim['n']:,} "
            f"simulaciones (seed {self.sim['seed']}).",
            "CLASIFICACION: Seguro ≥95 · Inclinado ≥80 · Lig. inclinado ≥65 · "
            "Competitivo ≥55 · Muy competitivo <55.",
            f"Elección objetivo: {self.fecha_eleccion:%d/%m/%Y}. "
            f"Mayoría = {self.mayoria} de {self.n_escanos}."]})
        with pd.ExcelWriter(ruta, engine="openpyxl") as xl:
            t.to_excel(xl, sheet_name="Proyeccion")
            self.sim["escanos"].round(1).to_excel(xl, sheet_name="Escanos")
            self.competitivos(50).to_excel(xl, sheet_name="Top competitivos", index=False)
            notas.to_excel(xl, sheet_name="Notas", index=False)
        return Path(ruta)

    def resumen(self) -> str:
        if not hasattr(self, "sim"):
            self.simular()
        esc = self.sim["escanos"]
        lineas = [f"🔮 Proyección 2027 — {self.sim['n']:,} simulaciones "
                  f"(elección {self.fecha_eleccion:%d/%m/%Y})"]
        sp = getattr(self, "shock_params", {})
        if sp.get("sigma_nacional", 0):
            lineas.insert(1, f"   🌊 Shock: nacional σ={sp['sigma_nacional']}pp · "
                             f"estatal σ={sp['sigma_estatal']}pp · ρ={sp['rho_shock']}")
        lineas.append("   🏛️  Escaños por bloque (media [p5–p95] · P(mayoría)):")
        for b, r in esc.iterrows():
            lineas.append(f"      {b:<16} {r['media']:6.1f}  [{r['p5']:.0f}–{r['p95']:.0f}]"
                          f"  P(mayoría)={r['p_mayoria']:.1f}%")
        return "\n".join(lineas)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Proyección probabilística 2027")
    ap.add_argument("--base", required=True)
    ap.add_argument("--encuestas")
    ap.add_argument("--prev")
    ap.add_argument("--ciclo-base", default="2024")
    ap.add_argument("--ciclo-prev", default="2021")
    ap.add_argument("--eleccion", default="2027-06-06")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--salida", default="proyeccion_2027.xlsx")
    ap.add_argument("--html")
    a = ap.parse_args()
    enc = Encuestas(a.encuestas) if a.encuestas else None
    prev = DatasetINE(a.prev, ciclo=a.ciclo_prev) if a.prev else None
    p = Proyeccion2027(DatasetINE(a.base, ciclo=a.ciclo_base), encuestas=enc,
                       ds_prev=prev, fecha_eleccion=date.fromisoformat(a.eleccion))
    p.simular(a.sims, a.seed)
    print(p.resumen())
    print("\n🔥 Top 15 más competitivos:\n" + p.competitivos(15).to_string(index=False))
    print(f"\n✅ Exportado → {p.exportar(a.salida).resolve()}")
    if a.html:
        from graficas_ine import wall_map_html
        wall_map_html(p.tabla_proyecciones(), "GANADOR_PROBABLE",
                      "Proyección 2027 — ganador probable (escenario estadístico)", a.html)
        print(f"🌐 {Path(a.html).resolve()}")
