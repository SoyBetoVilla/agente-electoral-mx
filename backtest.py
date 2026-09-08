"""backtest.py — Validación retrospectiva de proyeccion_2027 (sin fuga de información)."""
from __future__ import annotations

import argparse, json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from analisis_comparativo import BLOQUE_DE_PARTIDO, PARTIDOS_TOKEN, matriz_ciclo
from dataset_ine import DatasetINE
from proyeccion_2027 import Encuestas, Proyeccion2027

ELECCION_2024 = date(2024, 6, 2)
BINS_CONF = [0, 55, 65, 80, 95, 100.01]
ETIQ_CONF = ["<55 Muy competitivo", "55–65 Competitivo", "65–80 Lig. inclinado",
             "80–95 Inclinado", "≥95 Seguro"]


class EncuestasOracle:
    """Encuestas sintéticas con el resultado real nacional: valida el motor Monte Carlo,
    no la habilidad predictiva."""

    def __init__(self, ds_real: DatasetINE, bloques: list[str], sigma: float = 0.5):
        m = matriz_ciclo(ds_real)
        V = pd.DataFrame(index=m.index)
        for b in bloques:
            toks = [t for t in PARTIDOS_TOKEN
                    if BLOQUE_DE_PARTIDO.get(t) == b and t in m.columns]
            V[b] = m[toks].sum(axis=1) if toks else 0.0
        tot = float(V.values.sum())
        self.medias = {b: round(100 * float(V[b].sum()) / tot, 3) for b in bloques}
        self.sigma, self.half_life = float(sigma), 0.0
        self.meta_info = {"n_encuestas": 1, "muestra_total": 1_000_000,
                          "casas": ["ORACLE (resultado real)"]}

    def agregado(self, ambito="NACIONAL", id_entidad=None, id_distrito=None):
        if ambito != "NACIONAL":
            return None, None, {"n_encuestas": 0, "muestra_total": 0, "casas": []}
        return dict(self.medias), {b: self.sigma for b in self.medias}, dict(self.meta_info)

    def estatales(self):
        return {}

    def distritales(self):
        return {}


class Backtest:
    def __init__(self, ds_base: DatasetINE, ds_real: DatasetINE,
                 encuestas: Encuestas | None = None, oracle: bool = False,
                 fecha_eleccion: date = ELECCION_2024, penal_oficialismo: float = 0.0,
                 sims: int = 10_000, seed: int = 42, sigma_het_default: float = 4.0,
                 sigma_min: float = 2.5, mapeo: pd.DataFrame | None = None,
                 bloques_map: dict | None = None, shock: dict | None = None):
        if ds_base.ciclo == ds_real.ciclo:
            raise ValueError("El ciclo base y el real deben ser distintos (ej. 2021 → 2024).")
        if bloques_map:
            BLOQUE_DE_PARTIDO.update(bloques_map)
        self.ds_base, self.ds_real = ds_base, ds_real
        self.fecha_eleccion = fecha_eleccion
        self.bloques = sorted({b for b in BLOQUE_DE_PARTIDO.values() if b != "OTROS"})
        self.modo = ("oracle" if oracle else
                     "encuestas" if encuestas is not None else "status_quo")
        self._nota: list[str] = []

        enc = encuestas
        if self.modo == "oracle":
            enc = EncuestasOracle(ds_real, self.bloques)
            self._nota.append("Modo ORACLE: valida el motor Monte Carlo, no la habilidad.")
        elif self.modo == "encuestas":
            enc = self._depurar_fugas(enc)

        self.mapa = self._mapeo_claves(mapeo)
        self.proy = Proyeccion2027(ds_base, encuestas=enc, ds_prev=None,
                                   fecha_eleccion=fecha_eleccion,
                                   penalizacion_oficialismo=penal_oficialismo,
                                   sigma_het_default=sigma_het_default,
                                   sigma_min=sigma_min, escanos_en_juego=300)
        self.proy.simular(sims, seed, **(shock or {}))
        self.tabla, self.excluidos = self._emparejar()
        if self.tabla.empty:
            raise ValueError("Ningún distrito pudo emparejarse entre ciclos; revisa los "
                             "datasets o provee un mapeo.")
        self.metricas = self._metricas()

    # ─────────────── guardas y emparejamiento ───────────────
    def _depurar_fugas(self, enc: Encuestas) -> Encuestas:
        elec = pd.Timestamp(self.fecha_eleccion)
        despues = enc.df[enc.df["FECHA"] > elec]
        if len(despues):
            enc.df = enc.df[enc.df["FECHA"] <= elec].reset_index(drop=True)
            enc._calcular_pesos()
            self._nota.append(f"Anti-fuga: {len(despues)} encuesta(s) fechada(s) después "
                              f"de la elección fueron excluidas.")
        if getattr(enc, "fecha_ref", None) and enc.fecha_ref > self.fecha_eleccion:
            self._nota.append("⚠️ fecha_ref posterior a la elección: no reproduce "
                              "condiciones ex ante.")
        return enc

    def _mapeo_claves(self, mapeo) -> dict | None:
        if mapeo is None:
            return None
        m = mapeo.copy()
        m["ID_ENTIDAD"] = m["ID_ENTIDAD"].astype(str).str.split(".").str[0].str.zfill(2)
        m["ID_DISTRITO"] = pd.to_numeric(m["ID_DISTRITO"], errors="coerce").astype("Int64")
        m["ID_DISTRITO_PREV"] = pd.to_numeric(m.get("ID_DISTRITO_PREV"),
                                              errors="coerce").astype("Int64")
        m = m.dropna(subset=["ID_DISTRITO", "ID_DISTRITO_PREV"])
        par = pd.DataFrame({
            "CLAVE_BASE": m["ID_ENTIDAD"] + "-" + m["ID_DISTRITO_PREV"].astype(int)
                          .astype(str).str.zfill(3),
            "CLAVE_REAL": m["ID_ENTIDAD"] + "-" + m["ID_DISTRITO"].astype(int)
                          .astype(str).str.zfill(3)})
        dup = par["CLAVE_BASE"].duplicated() | par["CLAVE_REAL"].duplicated()
        if dup.any():
            print(f"   ⚠️ {int(dup.sum())} pares no 1:1 del mapeo quedan fuera de la evaluación.")
        par = par[~dup]
        self._nota.append(f"Evaluación restringida a {len(par)} distritos emparejados 1:1 "
                          f"vía mapeo de redistritación.")
        return dict(zip(par["CLAVE_BASE"], par["CLAVE_REAL"]))

    def _matriz_bloques(self, ds: DatasetINE):
        m = matriz_ciclo(ds)
        V = pd.DataFrame(index=m.index)
        for b in self.bloques:
            toks = [t for t in PARTIDOS_TOKEN
                    if BLOQUE_DE_PARTIDO.get(t) == b and t in m.columns]
            V[b] = m[toks].sum(axis=1) if toks else 0.0
        return m, V

    def _emparejar(self):
        tabla = self.proy.tabla_proyecciones().copy()
        tabla["CLAVE_BASE"] = (tabla["ID_ENTIDAD"].astype(str).str.zfill(2) + "-"
                               + tabla["ID_DISTRITO"].astype(int).astype(str).str.zfill(3))

        m_r, V_r = self._matriz_bloques(self.ds_real)
        pct_r = 100 * V_r.div(V_r.sum(axis=1).replace(0, np.nan), axis=0)
        real = pd.DataFrame({"CLAVE_REAL": m_r["CLAVE"], "GANADOR_REAL": V_r.idxmax(axis=1)})
        for b in self.bloques:
            real[f"REAL_PCT_{b}"] = pct_r[b].round(2).values
        s = np.sort(pct_r.values, axis=1)
        with np.errstate(invalid="ignore"):
            real["REAL_MARGEN"] = np.round(s[:, -1] - s[:, -2], 2)

        m_b, V_b = self._matriz_bloques(self.ds_base)
        base = pd.DataFrame({"CLAVE_BASE": m_b["CLAVE"], "GANADOR_BASE": V_b.idxmax(axis=1)})

        tabla["CLAVE_REAL"] = (tabla["CLAVE_BASE"] if self.mapa is None
                               else tabla["CLAVE_BASE"].map(self.mapa))
        excluidos = int(tabla["CLAVE_REAL"].isna().sum())
        f = (tabla.merge(real, on="CLAVE_REAL", how="inner")
                  .merge(base, on="CLAVE_BASE", how="left"))
        f["ACIERTO"] = f["GANADOR_PROBABLE"] == f["GANADOR_REAL"]
        f["ACIERTO_NAIVE"] = f["GANADOR_BASE"] == f["GANADOR_REAL"]
        flip = f["GANADOR_BASE"] != f["GANADOR_REAL"]
        f["FLIP_REAL"] = flip
        f["CATEGORIA"] = np.select(
            [(~flip) & (f["GANADOR_PROBABLE"] == f["GANADOR_BASE"]),
             flip & f["ACIERTO"],
             (~flip) & (~f["ACIERTO"]),
             flip & (f["GANADOR_PROBABLE"] == f["GANADOR_BASE"]),
             flip & (~f["ACIERTO"]) & (f["GANADOR_PROBABLE"] != f["GANADOR_BASE"])],
            ["Mantenido ✔", "Flip acertado ✔", "Falsa alarma ✘",
             "Flip omitido ✘", "Flip a otro bloque ✘"], default="s/d")
        return (f.sort_values(["ID_ENTIDAD", "ID_DISTRITO"]).reset_index(drop=True),
                excluidos)

    # ─────────────── métricas ───────────────
    def tabla_confiabilidad(self) -> pd.DataFrame:
        rango = pd.cut(self.tabla["P_GANADOR"], bins=BINS_CONF, labels=ETIQ_CONF,
                       right=False, include_lowest=True)
        g = (self.tabla.assign(RANGO=rango).groupby("RANGO", observed=False)
             .agg(n=("ACIERTO", "size"), P_MEDIA=("P_GANADOR", "mean"),
                  OBSERVADO=("ACIERTO", "mean")).reset_index())
        g["P_MEDIA"] = g["P_MEDIA"].astype(float).round(1)
        g["OBSERVADO"] = (g["OBSERVADO"].astype(float) * 100).round(1)
        return g

    def _metricas(self) -> dict:
        f = self.tabla
        valid = f[[f"REAL_PCT_{b}" for b in self.bloques]].notna().all(axis=1).values
        n = int(valid.sum())
        P = f.loc[valid, [f"P_{b}" for b in self.bloques]].to_numpy() / 100
        Y = (f.loc[valid, "GANADOR_REAL"].to_numpy()[:, None]
             == np.array(self.bloques)[None, :]).astype(float)
        brier = float(((P - Y) ** 2).sum(axis=1).mean())
        hit_naive = float(f.loc[valid, "ACIERTO_NAIVE"].mean())
        brier_naive = 2 * (1 - hit_naive)
        skill = 1 - brier / brier_naive if brier_naive > 0 else float("nan")
        conf = self.tabla_confiabilidad()
        ece = float((conf["n"] / n * (conf["P_MEDIA"] - conf["OBSERVADO"]).abs())
                    .fillna(0).sum())
        pcols = f.loc[valid, [f"PCT_{b}" for b in self.bloques]].to_numpy()
        rcols = f.loc[valid, [f"REAL_PCT_{b}" for b in self.bloques]].to_numpy()
        sp, sr = np.sort(pcols, 1), np.sort(rcols, 1)
        mae_margen = float(np.nanmean(
            np.abs((sp[:, -1] - sp[:, -2]) - (sr[:, -1] - sr[:, -2]))))
        cat = f["CATEGORIA"].value_counts()
        return {"modo": self.modo, "n": n, "excluidos": self.excluidos,
                "acierto": round(100 * float(f.loc[valid, "ACIERTO"].mean()), 1),
                "acierto_naive": round(100 * hit_naive, 1),
                "brier": round(brier, 4), "brier_naive": round(brier_naive, 4),
                "skill_vs_ingenuo": round(skill, 3) if np.isfinite(skill) else None,
                "ece_pp": round(ece, 2), "mae_margen_pp": round(mae_margen, 2),
                "mae_pct_bloque": {b: round(float((f[f"PCT_{b}"] - f[f"REAL_PCT_{b}"])
                                                  .abs().mean()), 2) for b in self.bloques},
                "sesgo_pct_bloque": {b: round(float((f[f"PCT_{b}"] - f[f"REAL_PCT_{b}"])
                                                    .mean()), 2) for b in self.bloques},
                "flips_reales": int(f.loc[valid, "FLIP_REAL"].sum()),
                "flip_acertado": int(cat.get("Flip acertado ✔", 0)),
                "flip_omitido": int(cat.get("Flip omitido ✘", 0)),
                "falsa_alarma": int(cat.get("Falsa alarma ✘", 0)),
                "flip_otro_bloque": int(cat.get("Flip a otro bloque ✘", 0))}

    def _tabla_escanos(self) -> pd.DataFrame:
        esc = self.proy.sim["escanos"]
        real = self.tabla["GANADOR_REAL"].value_counts()
        filas = []
        for i, b in enumerate(self.bloques):
            r = esc.loc[b]
            re = int(real.get(b, 0))
            col = np.sort(self.proy.sim["esc_matrix"][:, i])
            pctil = 100 * np.searchsorted(col, re, side="right") / len(col)
            filas.append({"BLOQUE": b, "REAL": re, "MEDIA_SIM": round(float(r["media"]), 1),
                          "P5": float(r["p5"]), "P95": float(r["p95"]),
                          "PCTIL_REAL": round(pctil, 1),
                          "EN_P5_P95": bool(r["p5"] <= re <= r["p95"])})
        return pd.DataFrame(filas)

    # ─────────────── salidas ───────────────
    def resumen(self) -> str:
        m = self.metricas
        L = [f"🧪 Backtest {self.ds_base.ciclo} → {self.ds_real.ciclo} · modo {self.modo.upper()}",
             f"   Distritos evaluados: {m['n']}"
             + (f" (sin par, excluidos: {m['excluidos']})" if m["excluidos"] else "")]
        L += [f"   ℹ️  {x}" for x in self._nota]
        L += ["", f"   ✔️  Acierto de ganador   {m['acierto']:>5}%   "
                  f"(ingenuo: {m['acierto_naive']}%)",
              f"   🎯 Brier multiclass     {m['brier']:>5}    (ingenuo: {m['brier_naive']})"
              f"  → skill {m['skill_vs_ingenuo']}",
              f"   📐 ECE (calibración)    {m['ece_pp']:>5} pp",
              f"   📏 MAE margen 1º–2º     {m['mae_margen_pp']:>5} pp",
              f"   🔁 Flips reales {m['flips_reales']}: acertados {m['flip_acertado']} · "
              f"omitidos {m['flip_omitido']} · falsas alarmas {m['falsa_alarma']}",
              "", "   📊 Confiabilidad (favorito):"]
        for _, r in self.tabla_confiabilidad().iterrows():
            pm = "—" if pd.isna(r["P_MEDIA"]) else f"{r['P_MEDIA']:.1f}%"
            ob = "—" if pd.isna(r["OBSERVADO"]) else f"{r['OBSERVADO']:.1f}%"
            L.append(f"      {r['RANGO']:<20} n={int(r['n']):>3}   P̄={pm:>7}   obs={ob:>7}")
        L += ["", "   🏛️  Escaños: real vs simulación:",
              self._tabla_escanos().to_string(index=False)]
        return "\n".join(L)

    def grafica(self, ruta) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conf = self.tabla_confiabilidad()
        conf = conf[(conf["n"] > 0) & conf["P_MEDIA"].notna()]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
        a1.plot([0, 100], [0, 100], "--", color="grey", lw=1, label="calibración perfecta")
        a1.scatter(conf["P_MEDIA"], conf["OBSERVADO"], s=conf["n"] * 4, color="#2E6DB4", zorder=3)
        for _, r in conf.iterrows():
            a1.annotate(f"n={int(r['n'])}", (r["P_MEDIA"], r["OBSERVADO"]),
                        textcoords="offset points", xytext=(8, -3), fontsize=8)
        a1.set_xlabel("P(victoria) proyectada del favorito (%)")
        a1.set_ylabel("Frecuencia real de acierto (%)")
        a1.set_title(f"Confiabilidad — {self.ds_base.ciclo} → {self.ds_real.ciclo} ({self.modo})")
        a1.legend(); a1.grid(alpha=.3)
        ok = self.tabla[self.tabla["ACIERTO"]]["P_GANADOR"]
        mal = self.tabla[~self.tabla["ACIERTO"]]["P_GANADOR"]
        a2.hist([ok, mal], bins=np.arange(50, 105, 5), stacked=True,
                color=["#2E8B57", "#C0392B"], label=["acierto", "fallo"], edgecolor="white")
        a2.set_xlabel("P(victoria) del favorito (%)"); a2.set_ylabel("Distritos")
        a2.set_title("Confianza vs. resultado"); a2.legend(); a2.grid(alpha=.3)
        fig.tight_layout()
        ruta = Path(ruta)
        fig.savefig(ruta, dpi=150)
        plt.close(fig)
        return ruta

    def exportar(self, ruta) -> Path:
        m, f = self.metricas, self.tabla
        comp = f[["ID_ENTIDAD", "ENTIDAD", "ID_DISTRITO", "CLAVE_BASE", "CLAVE_REAL",
                  "GANADOR_BASE", "GANADOR_PROBABLE", "P_GANADOR", "CLASIFICACION",
                  "GANADOR_REAL", "ACIERTO", "CATEGORIA", "ACIERTO_NAIVE", "FUENTE_SWING"]
                 + [f"PCT_{b}" for b in self.bloques]
                 + [f"REAL_PCT_{b}" for b in self.bloques]]
        met = [("modo", m["modo"]), ("distritos_evaluados", m["n"]),
               ("excluidos_sin_par", m["excluidos"]), ("acierto_%", m["acierto"]),
               ("acierto_ingenuo_%", m["acierto_naive"]), ("brier", m["brier"]),
               ("brier_ingenuo", m["brier_naive"]),
               ("skill_vs_ingenuo", m["skill_vs_ingenuo"]), ("ece_pp", m["ece_pp"]),
               ("mae_margen_pp", m["mae_margen_pp"]),
               ("flips_reales", m["flips_reales"]), ("flip_acertado", m["flip_acertado"]),
               ("flip_omitido", m["flip_omitido"]), ("falsa_alarma", m["falsa_alarma"])]
        met += [(f"MAE_pct_{b}", v) for b, v in m["mae_pct_bloque"].items()]
        notas = pd.DataFrame({"Nota": [
            f"Backtest {self.ds_base.ciclo} → {self.ds_real.ciclo} (modo {m['modo']}). "
            "La proyección NO usa el resultado real, salvo en modo ORACLE (intencional).",
            "Denominador de %: solo votos de bloques (sin nulos), igual que la proyección.",
            "ACIERTO_NAIVE = acierta el ganador del ciclo base. En STATUS_QUO el acierto "
            "puntual coincide con el ingenuo por construcción; el valor está en la "
            "calibración (Brier/ECE) y en la cobertura p5–p95.",
            "Anti-fuga: encuestas post-elección excluidas; σ_tiempo=0; el dataset real "
            "solo entra a la evaluación.", *self._nota,
            f"Fuentes: {self.ds_base.ruta.name}, {self.ds_real.ruta.name}."]})
        vc = f["CATEGORIA"].value_counts()
        flips = pd.DataFrame({"CATEGORIA": vc.index, "DISTRITOS": vc.values})
        with pd.ExcelWriter(ruta, engine="openpyxl") as xl:
            comp.to_excel(xl, sheet_name="Comparativo", index=False)
            pd.DataFrame(met, columns=["Metrica", "Valor"]).to_excel(
                xl, sheet_name="Metricas", index=False)
            self.tabla_confiabilidad().to_excel(xl, sheet_name="Confiabilidad", index=False)
            self._tabla_escanos().to_excel(xl, sheet_name="Escanos", index=False)
            flips.to_excel(xl, sheet_name="Flips", index=False)
            notas.to_excel(xl, sheet_name="Notas", index=False)
        return Path(ruta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backtest de la proyección electoral")
    ap.add_argument("--base", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--ciclo-base", default="2021")
    ap.add_argument("--ciclo-real", default="2024")
    ap.add_argument("--encuestas")
    ap.add_argument("--fecha-ref")
    ap.add_argument("--eleccion", default="2024-06-02")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--mapeo")
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--penal-oficialismo", type=float, default=0.0)
    ap.add_argument("--sigma-nacional", type=float, default=0.0)
    ap.add_argument("--sigma-estatal", type=float, default=0.0)
    ap.add_argument("--rho-shock", type=float, default=0.0)
    ap.add_argument("--salida", default="backtest.xlsx")
    ap.add_argument("--png")
    a = ap.parse_args()
    eleccion = date.fromisoformat(a.eleccion)
    enc = None
    if a.encuestas and not a.oracle:
        fref = date.fromisoformat(a.fecha_ref) if a.fecha_ref else eleccion - timedelta(days=21)
        enc = Encuestas(a.encuestas, fecha_referencia=fref)
    bt = Backtest(DatasetINE(a.base, ciclo=a.ciclo_base),
                  DatasetINE(a.real, ciclo=a.ciclo_real), encuestas=enc,
                  oracle=a.oracle, fecha_eleccion=eleccion,
                  penal_oficialismo=a.penal_oficialismo, sims=a.sims, seed=a.seed,
                  mapeo=pd.read_csv(a.mapeo, dtype={"ID_ENTIDAD": str}) if a.mapeo else None,
                  shock={"sigma_nacional": a.sigma_nacional,
                         "sigma_estatal": a.sigma_estatal, "rho_shock": a.rho_shock})
    print(bt.resumen())
    print(f"\n✅ Exportado → {bt.exportar(a.salida).resolve()}")
    if a.png:
        print(f"🖼️  {bt.grafica(a.png).resolve()}")
