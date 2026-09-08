"""auto_calibracion.py — Grid search de σ con el backtest → config_proyeccion.json."""
from __future__ import annotations

import argparse, copy, itertools, json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analisis_comparativo import BLOQUE_DE_PARTIDO
from backtest import Backtest
from dataset_ine import DatasetINE
from proyeccion_2027 import Encuestas

BINS = [0, 55, 65, 80, 95, 100.01]


def _brier_ece(df: pd.DataFrame, bloques: list[str]) -> tuple[float, float]:
    Y = (df["GANADOR_REAL"].to_numpy()[:, None] == np.array(bloques)[None, :]).astype(float)
    P = df[[f"P_{b}" for b in bloques]].to_numpy() / 100
    brier = float(((P - Y) ** 2).sum(axis=1).mean())
    rango = pd.cut(df["P_GANADOR"], bins=BINS, right=False, include_lowest=True)
    g = (df.assign(R=rango).groupby("R", observed=False)
           .agg(n=("ACIERTO", "size"), pm=("P_GANADOR", "mean"), obs=("ACIERTO", "mean")))
    ece = float((g["n"] / len(df) * (g["pm"] - g["obs"] * 100).abs()).fillna(0).sum())
    return brier, ece


class Calibrador:
    def __init__(self, ds_base, ds_real, encuestas=None, fecha_eleccion=None,
                 lambda_ece: float = 0.5, sims_fast: int = 2000, sims_final: int = 10000,
                 seed: int = 42, callback=None):
        self.ds_base, self.ds_real, self.encuestas = ds_base, ds_real, encuestas
        self.fecha_eleccion = fecha_eleccion or date(2024, 6, 2)
        self.lam, self.sims_fast, self.sims_final = lambda_ece, sims_fast, sims_final
        self.seed, self.callback = seed, callback
        self.bloques = sorted({b for b in BLOQUE_DE_PARTIDO.values() if b != "OTROS"})
        self.grid: pd.DataFrame | None = None
        self.tablas: dict[tuple, pd.DataFrame] = {}
        self.mejor: tuple | None = None
        self.bt_final: Backtest | None = None

    def _ejecutar(self, het, mn, pen, sims, seed) -> Backtest:
        return Backtest(self.ds_base, self.ds_real,
                        encuestas=copy.deepcopy(self.encuestas) if self.encuestas else None,
                        fecha_eleccion=self.fecha_eleccion, penal_oficialismo=pen,
                        sims=sims, seed=seed, sigma_het_default=het, sigma_min=mn)

    def evaluar_grid(self, grid_het, grid_min, grid_penal) -> pd.DataFrame:
        combos = list(itertools.product(grid_het, grid_min, grid_penal))
        filas = []
        for i, (h, mn, pe) in enumerate(combos, 1):
            bt = self._ejecutar(h, mn, pe, self.sims_fast, self.seed)
            m = bt.metricas
            obj = m["brier"] + self.lam * m["ece_pp"] / 100
            filas.append({"sigma_het": h, "sigma_min": mn, "penal": pe,
                          "brier": m["brier"], "ece_pp": m["ece_pp"],
                          "acierto": m["acierto"], "objetivo": round(obj, 5)})
            self.tablas[(h, mn, pe)] = bt.tabla[
                ["ID_ENTIDAD", "P_GANADOR", "ACIERTO", "GANADOR_REAL"]
                + [f"P_{b}" for b in self.bloques]].copy()
            if self.callback:
                self.callback(i, len(combos), obj)
        self.grid = pd.DataFrame(filas).sort_values("objetivo").reset_index(drop=True)
        return self.grid

    def refinar(self, k: int = 3) -> tuple:
        fin = []
        for _, r in self.grid.head(k).iterrows():
            combo = (r["sigma_het"], r["sigma_min"], r["penal"])
            bt = self._ejecutar(*combo, self.sims_final, self.seed + 1)
            m = bt.metricas
            fin.append((m["brier"] + self.lam * m["ece_pp"] / 100, combo, bt))
        fin.sort(key=lambda x: x[0])
        self.mejor, self.bt_final = fin[0][1], fin[0][2]
        return self.mejor

    def bootstrap(self, r: int = 200, seed: int = 7, top: int = 5) -> pd.DataFrame:
        combos = [tuple(self.grid.iloc[i][["sigma_het", "sigma_min", "penal"]])
                  for i in range(min(top, len(self.grid)))]
        datos = {c: {e: g for e, g in self.tablas[c].groupby("ID_ENTIDAD")} for c in combos}
        ents = sorted(datos[combos[0]])
        rng = np.random.default_rng(seed)
        objs = {c: [] for c in combos}
        votos: dict[tuple, int] = {}
        for _ in range(r):
            muestra_ents = rng.choice(ents, len(ents))
            for c in combos:
                b, e = _brier_ece(pd.concat([datos[c][x] for x in muestra_ents]), self.bloques)
                objs[c].append(b + self.lam * e / 100)
            gana = min(combos, key=lambda c: objs[c][-1])
            votos[gana] = votos.get(gana, 0) + 1
        return pd.DataFrame([{"sigma_het": c[0], "sigma_min": c[1], "penal": c[2],
                              "objetivo_mediana": round(float(np.median(objs[c])), 5),
                              "elegido_%": round(100 * votos.get(c, 0) / r, 1)}
                             for c in combos]).sort_values("objetivo_mediana")

    def sugerir_shock(self, candidatos: list[dict] | None = None) -> pd.DataFrame:
        candidatos = candidatos or [
            {"sigma_nacional": 0.0, "sigma_estatal": 0.0, "rho_shock": 0.0},
            {"sigma_nacional": 1.0, "sigma_estatal": 0.5, "rho_shock": 0.7},
            {"sigma_nacional": 1.5, "sigma_estatal": 1.0, "rho_shock": 0.7},
            {"sigma_nacional": 2.0, "sigma_estatal": 1.0, "rho_shock": 0.8}]
        h, mn, pe = self.mejor
        filas = []
        for sh in candidatos:
            bt = Backtest(self.ds_base, self.ds_real,
                          encuestas=copy.deepcopy(self.encuestas) if self.encuestas else None,
                          fecha_eleccion=self.fecha_eleccion, penal_oficialismo=pe,
                          sims=self.sims_fast, seed=self.seed,
                          sigma_het_default=h, sigma_min=mn, shock=sh)
            t = bt._tabla_escanos()
            filas.append({**sh, "cobertura_p5_p95": round(t["EN_P5_P95"].mean(), 2),
                          "desvio_pctil": round(float(np.abs(t["PCTIL_REAL"] - 50).mean()), 1)})
        return pd.DataFrame(filas)

    def escribir_config(self, ruta, shock: dict | None = None) -> Path:
        h, mn, pe = self.mejor
        cfg = {"sigma_het_default": float(h), "sigma_min": float(mn),
               "penalizacion_oficialismo": float(pe), "shock": shock or {},
               "meta": {"fuente": f"auto_calibracion {self.ds_base.ciclo}→{self.ds_real.ciclo}",
                        "objetivo": f"Brier + {self.lam}·ECE | sims_final={self.sims_final}",
                        "bt_brier": self.bt_final.metricas["brier"],
                        "bt_ece_pp": self.bt_final.metricas["ece_pp"],
                        "bt_acierto": self.bt_final.metricas["acierto"],
                        "advertencia": "calibrado con UNA elección: verificar bootstrap"}}
        ruta = Path(ruta)
        ruta.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        return ruta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-calibración de la proyección")
    ap.add_argument("--base", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--ciclo-base", default="2021")
    ap.add_argument("--ciclo-real", default="2024")
    ap.add_argument("--encuestas")
    ap.add_argument("--fecha-ref")
    ap.add_argument("--eleccion", default="2024-06-02")
    ap.add_argument("--lambda-ece", type=float, default=0.5)
    ap.add_argument("--grid-het", type=float, nargs="+", default=[2, 3, 4, 5, 6, 7])
    ap.add_argument("--grid-min", type=float, nargs="+", default=[2.5, 3, 3.5, 4])
    ap.add_argument("--grid-penal", type=float, nargs="+", default=[0.0, 2.0, 4.0])
    ap.add_argument("--sims-fast", type=int, default=2000)
    ap.add_argument("--sims-final", type=int, default=10000)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--salida-config", default="config_proyeccion.json")
    ap.add_argument("--env")
    a = ap.parse_args()
    elec = date.fromisoformat(a.eleccion)
    enc = None
    if a.encuestas:
        fref = (date.fromisoformat(a.fecha_ref) if a.fecha_ref
                else elec - pd.Timedelta(days=21).to_pytimedelta())
        enc = Encuestas(a.encuestas, fecha_referencia=fref)
    cal = Calibrador(DatasetINE(a.base, ciclo=a.ciclo_base),
                     DatasetINE(a.real, ciclo=a.ciclo_real), encuestas=enc,
                     fecha_eleccion=elec, lambda_ece=a.lambda_ece,
                     sims_fast=a.sims_fast, sims_final=a.sims_final)
    n = len(a.grid_het) * len(a.grid_min) * len(a.grid_penal)
    print(f"🔧 Grid: {n} combos × {a.sims_fast:,} sims (CRN seed={cal.seed})")
    cal.evaluar_grid(a.grid_het, a.grid_min, a.grid_penal)
    print("\n📊 Top 10:\n" + cal.grid.head(10).to_string(index=False))
    mejor = cal.refinar()
    print(f"\n🏆 Óptimo refinado: sigma_het={mejor[0]} · sigma_min={mejor[1]} · penal={mejor[2]}")
    stab = cal.bootstrap(a.bootstrap)
    print(f"\n🧮 Estabilidad (bootstrap ×{a.bootstrap}):\n" + stab.to_string(index=False))
    shocks = cal.sugerir_shock()
    print("\n🌊 Shock (cobertura objetivo ≈ 0.9):\n" + shocks.to_string(index=False))
    mejor_shock = shocks.sort_values("desvio_pctil").iloc[0].to_dict()
    ruta = cal.escribir_config(a.salida_config,
                               shock={k: float(v) for k, v in mejor_shock.items()
                                      if k.startswith(("sigma", "rho"))})
    print(f"\n✅ Config → {ruta.resolve()}")
    if a.env:
        p = Path(a.env)
        lineas = [l for l in (p.read_text().splitlines() if p.exists() else [])
                  if not l.startswith("RUTA_CONFIG_PROYECCION")]
        lineas.append(f"RUTA_CONFIG_PROYECCION={a.salida_config}")
        p.write_text("\n".join(lineas) + "\n")
        print(f"🔑 {p.resolve()} actualizado")
