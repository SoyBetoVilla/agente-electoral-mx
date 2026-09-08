"""demo_ine.py — Genera datasets sintéticos con estructura INE para el modo demo.
⚠️ FICTICIOS (prefijo DEMO_). Cómputos reales: computos.ine.mx"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PARTIDOS = ["PAN", "PRI", "PRD", "MORENA", "PT", "PVEM", "MC"]
ESPECIALES = ["CIUDADANOS", "NO_REGISTRADAS", "NULOS"]
BLOQUES = ["MORENA-PT-PVEM", "PAN-PRI-PRD", "MC"]
DENTRO = {"PAN-PRI-PRD": {"PAN": .5, "PRI": .3, "PRD": .2},
          "MORENA-PT-PVEM": {"MORENA": .7, "PT": .2, "PVEM": .1},
          "MC": {"MC": 1.0}}
ENTIDADES = {"01": "Alpha", "02": "Beta", "03": "Gamma",
             "04": "Delta", "05": "Epsilon", "06": "Zeta"}
D_POR_ENT, CABECERAS = 4, ["Puerto Alto", "Villa Norte", "San Lucas", "Monte Real"]


def _bloque(ent: str, dis: int, ciclo: str) -> str:
    b = BLOQUES[((int(ent) - 1) * 10 + dis) % 3]
    if ciclo == "2024" and (ent, dis) == ("02", 2):          # un flip diseñado
        b = BLOQUES[(BLOQUES.index(b) + 1) % 3]
    return b


def _fila(ent: str, dis: int, ciclo: str, rng) -> dict:
    gan = _bloque(ent, dis, ciclo)
    otros = [b for b in BLOQUES if b != gan]
    prop = np.clip(np.array([.52, .32, .16]) + rng.normal(0, .015, 3), .06, None)
    prop /= prop.sum()
    n = int((45_000 + 900 * ((int(ent) - 1) * 10 + dis)) * (1.05 if ciclo == "2024" else 1.0))
    votos = {p: int(n * pb * wp) for b, pb in zip([gan, *otros], prop)
             for p, wp in DENTRO[b].items()}
    esp = {"NULOS": int(n * .02), "CIUDADANOS": int(n * .004), "NO_REGISTRADAS": int(n * .002)}
    return {"ID_ENTIDAD": ent, "ENTIDAD": ENTIDADES[ent], "ID_DISTRITO": dis,
            "CABECERA_DISTRITAL": f"{CABECERAS[(int(ent) - 1) % 4]} {dis}",
            **{p: votos.get(p, 0) for p in PARTIDOS}, **esp,
            "TOTAL_VOTOS_CALCULADO": sum(votos.values()) + sum(esp.values())}


def escribir(ruta: Path, ciclo: str, semilla: int) -> Path:
    rng = np.random.default_rng(semilla)
    filas = [_fila(e, d, ciclo, rng) for e in ENTIDADES for d in range(1, D_POR_ENT + 1)]
    pd.DataFrame(filas).to_csv(ruta, index=False, encoding="utf-8-sig")
    return ruta


def generar(directorio="datos/demo") -> tuple[Path, Path]:
    d = Path(directorio)
    d.mkdir(parents=True, exist_ok=True)
    return (escribir(d / "DEMO_computos_2021.csv", "2021", 11),
            escribir(d / "DEMO_computos_2024.csv", "2024", 22))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="datos/demo")
    a = ap.parse_args()
    r21, r24 = generar(a.dir)
    print(f"🎮 Demo generada (24 distritos FICTICIOS):\n   {r21}\n   {r24}")
