"""conftest.py — Datasets sintéticos con estructura INE (3 entidades × 3 distritos)."""
from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PARTIDOS = ["PAN", "PRI", "PRD", "MORENA", "PT", "PVEM", "MC"]
ESPECIALES = ["CIUDADANOS", "NO_REGISTRADAS", "NULOS"]
META = ["ID_ENTIDAD", "ENTIDAD", "ID_DISTRITO", "CABECERA_DISTRITAL"]
COLS = META + PARTIDOS + ESPECIALES + ["TOTAL_VOTOS_CALCULADO"]
COLS_CAS = ["ID_CASILLA", "SECCION"] + COLS

ENT_NOMBRES = {"01": "Alpha", "02": "Beta", "03": "Gamma"}
CABECERAS = {("01", "1"): "San Lucas", ("01", "2"): "Villa Norte", ("01", "3"): "Puerto Alto",
             ("02", "1"): "El Rosario", ("02", "2"): "Las Palmas", ("02", "3"): "Monte Real",
             ("03", "1"): "Costa Azul", ("03", "2"): "Sierra Blanca", ("03", "3"): "Valle Verde"}
_DENTRO = {"PAN-PRI-PRD": {"PAN": .5, "PRI": .3, "PRD": .2},
           "MORENA-PT-PVEM": {"MORENA": .7, "PT": .2, "PVEM": .1}, "MC": {"MC": 1.0}}
BASE = ["MORENA-PT-PVEM", "PAN-PRI-PRD", "MC"]


def _fila(ent, dis, cab, ciclo, rng) -> dict:
    idx = (int(ent) - 1) * 10 + int(dis)
    gan = BASE[idx % 3]
    if ciclo == "2024" and (ent, str(dis)) == ("02", "2"):     # un flip diseñado
        gan = "PAN-PRI-PRD" if gan != "PAN-PRI-PRD" else "MORENA-PT-PVEM"
    prop = np.clip(np.array([.55, .30, .15]) + rng.normal(0, .01, 3), .08, None)
    prop /= prop.sum()
    n = int((40_000 + 1_500 * idx) * (1.05 if ciclo == "2024" else 1.0))
    votos = {}
    for b, pb in zip([gan] + [x for x in BASE if x != gan], prop):
        for p, wp in _DENTRO[b].items():
            votos[p] = int(n * pb * wp)
    esp = {"NULOS": int(n * .02), "CIUDADANOS": int(n * .004), "NO_REGISTRADAS": int(n * .002)}
    return {"ID_ENTIDAD": ent, "ENTIDAD": ENT_NOMBRES[ent], "ID_DISTRITO": int(dis),
            "CABECERA_DISTRITAL": cab,
            **{p: votos.get(p, 0) for p in PARTIDOS}, **esp,
            "TOTAL_VOTOS_CALCULADO": sum(votos.values()) + sum(esp.values())}


def _filas_casilla(ent, dis, cab, ciclo, rng) -> list[dict]:
    f0 = _fila(ent, dis, cab, ciclo, rng)
    numeric = [c for c in COLS if c not in META + ["TOTAL_VOTOS_CALCULADO"]]
    filas = []
    for i, frac in enumerate((.4, .3, .2, .1), 1):
        r = {"ID_CASILLA": f"{ent}{int(dis):03d}{i:02d}",
             "SECCION": (int(dis) - 1) * 100 + i, **{k: f0[k] for k in META}}
        for c in numeric:
            r[c] = int(round(f0[c] * frac))
        r["TOTAL_VOTOS_CALCULADO"] = sum(r[c] for c in numeric)
        filas.append(r)
    return filas


def escribir_ine(ruta: Path, ciclo: str, *, variantes: dict | None = None,
                 delim: str = ",", enc: str = "utf-8-sig", nivel: str = "distrito",
                 semilla: int | None = None, ent_nombres: dict | None = None) -> Path:
    v = variantes or {}
    ent_nombres = ent_nombres or ENT_NOMBRES
    rng = np.random.default_rng(semilla if semilla is not None
                                else {"2021": 11, "2024": 22}.get(ciclo, 7))
    claves = [k for k in CABECERAS if k not in v.get("quitar", ())]
    claves += list(v.get("agregar", ()))
    ren = v.get("renombrar", {})
    cols = COLS_CAS if nivel == "casillas" else COLS
    filas = []
    for ent, dis in sorted(claves):
        cab = ren.get((ent, dis), CABECERAS[(ent, dis)])
        out = (_filas_casilla if nivel == "casillas" else _fila)(ent, dis, cab, ciclo, rng)
        filas += out if isinstance(out, list) else [out]
    df = pd.DataFrame(filas, columns=cols)
    if ent_nombres != ENT_NOMBRES:
        df["ENTIDAD"] = df["ID_ENTIDAD"].map(ent_nombres)
    df.to_csv(ruta, sep=delim, index=False, encoding=enc)
    return ruta


ENCUESTAS_CSV = """fecha,casa,ambito,id_entidad,id_distrito,muestra,morena,pan,pri,prd,pt,pvem,mc,otros
2024-05-01,Financiero,NACIONAL,,,"1,800",42,26,2,1,5,5,14,5
2024-05-10,Mitofsky,NACIONAL,,,1500,44,27,2,1,4,5,13,4
2024-05-15,LocalA,ESTATAL,01,,900,47,30,2,1,4,5,9,2
2024-05-18,LocalB,ESTATAL,02,,800,40,33,2,1,4,6,11,3
2024-05-20,LocalC,DISTRITAL,01,1,600,48,32,2,1,3,5,7,2
"""


@pytest.fixture(scope="session")
def datos(tmp_path_factory) -> SimpleNamespace:
    d = tmp_path_factory.mktemp("datos_ine")
    ns = {"c2021": escribir_ine(d / "c2021.csv", "2021"),
          "c2024": escribir_ine(d / "c2024.csv", "2024"),
          "cB": escribir_ine(d / "c2027B.csv", "2024", semilla=33,
                             variantes={"quitar": (("01", "3"),),
                                        "agregar": (("03", "4"),),
                                        "renombrar": {("01", "1"): "San Lucas Norte"}}),
          "enc": d / "encuestas.csv"}
    ns["enc"].write_text(ENCUESTAS_CSV, encoding="utf-8")
    with zipfile.ZipFile(zp := d / "c2024.zip", "w") as z:
        z.write(ns["c2024"], "Computos_2024.csv")
    ns["czip"] = zp
    ns["csem"] = escribir_ine(d / "c2024_semicolon.csv", "2024", delim=";", enc="cp1252",
                              ent_nombres={"01": "Nuevo León", "02": "Yucatán",
                                           "03": "Michoacán"})
    ns["ccas"] = escribir_ine(d / "c2024_casillas.csv", "2024", nivel="casillas")
    return SimpleNamespace(**ns)
