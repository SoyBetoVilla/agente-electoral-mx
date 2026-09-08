"""Suite condensada: invariantes críticos de todo el pipeline con datasets sintéticos."""
from __future__ import annotations

import importlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from analisis_comparativo import ComparadorINE
from auto_calibracion import Calibrador, _brier_ece
from backtest import Backtest, EncuestasOracle
from dataset_ine import DatasetINE, formatear_distrito
from proyeccion_2027 import Encuestas, Proyeccion2027
from remapeo_2027 import ComparadorMapeado, construir_mapeo, previos_sin_uso

# Ganadores de partido esperados en 2024 (diseñados por el fixture)
ESP = {"11": "PAN", "12": "MC", "13": "MORENA", "21": "MC", "22": "PAN",
       "23": "PAN", "31": "MORENA", "32": "PAN", "33": "MC"}
BLOQUES = ["MC", "MORENA-PT-PVEM", "PAN-PRI-PRD"]


# ─────────────── dataset_ine ───────────────
def test_carga_y_ganadores(datos):
    ds = DatasetINE(datos.c2024, ciclo="2024")
    assert len(ds.df) == 9 and ds.col_total == "TOTAL_VOTOS_CALCULADO"
    assert set(ds.partidos) == {"PAN", "PRI", "PRD", "MORENA", "PT", "PVEM", "MC"}
    todo = ds.consultar_todos()
    for _, r in todo.iterrows():
        assert r["GANADOR"] == ESP[f"{int(r['ID_ENTIDAD'])}{int(r['ID_DISTRITO'])}"]
    assert todo["ID_ENTIDAD"].nunique() == 3


def test_zip_semicolon_y_encoding(datos):
    assert len(DatasetINE(datos.czip, ciclo="2024").df) == 9
    dss = DatasetINE(datos.csem, ciclo="2024")
    assert "Nuevo León" in set(dss.df["ENTIDAD"])
    assert dss.consultar("Nuevo León", 1)["entidad"] == "Nuevo León"


def test_consultas_y_errores(datos):
    ds = DatasetINE(datos.c2024, ciclo="2024")
    r = ds.consultar("01", 2)
    assert r["ganador"] == "MC" and r["distrito"] == 2
    assert "aviso" in ds.consultar("01")
    with pytest.raises(ValueError):
        ds.consultar("Zeta", 1)
    with pytest.raises(ValueError):
        ds.consultar("01", 99)


def test_agregacion_casillas(datos):
    dsc = DatasetINE(datos.ccas, ciclo="2024")
    assert len(dsc.df) == 9
    assert not ({"ID_CASILLA", "SECCION"} & set(dsc.partidos))
    assert dsc.consultar("01", 1)["total_votos"] > 0


def test_formateo(datos):
    md = formatear_distrito(DatasetINE(datos.c2024, ciclo="2024").consultar("01", 1))
    assert all(s in md for s in ("| Partido / Coalición |", "🏆 Ganador", "📚 Fuente"))


# ─────────────── comparativo y remapeo ───────────────
def test_comparador_flip_exacto(datos):
    comp = ComparadorINE(DatasetINE(datos.c2021, ciclo="2021"),
                         DatasetINE(datos.c2024, ciclo="2024"))
    t = comp.tabla_distritos()
    assert len(t) == 9
    flips = t[t["CAMBIO_GANADOR"]]
    assert len(flips) == 1
    assert (flips.iloc[0]["ID_ENTIDAD"], int(flips.iloc[0]["ID_DISTRITO"])) == ("02", 2)
    assert {"DPP_MORENA", "VOLATILIDAD_PP", "SWING_PP"} <= set(t.columns)
    assert t["VOLATILIDAD_PP"].min() >= 0 and t["VOLATILIDAD_PP"].max() > 0


def test_resumen_nacional_y_export(datos, tmp_path):
    comp = ComparadorINE(DatasetINE(datos.c2021, ciclo="2021"),
                         DatasetINE(datos.c2024, ciclo="2024"))
    res = comp.resumen_nacional()
    assert res["bloques"]["ESCANOS_2024"].sum() == 9
    assert res["partidos"]["ESCANOS_2024"].sum() == 9
    assert len(comp.top_volatiles(3)) == 3
    assert comp.exportar(tmp_path / "comp.xlsx").exists()


def test_mapeo_directo_cabecera_y_sin_par(datos):
    dsA, dsB = DatasetINE(datos.c2021, ciclo="2021"), DatasetINE(datos.cB, ciclo="2027")
    m = construir_mapeo(dsA, dsB)
    assert m["METODO"].value_counts().to_dict() == {"DIRECTO": 7, "CABECERA": 1, "SIN_PAR": 1}
    fila = m[m["METODO"] == "CABECERA"].iloc[0]
    assert (fila["ID_ENTIDAD"], int(fila["ID_DISTRITO"])) == ("01", 1)
    sp = m[m["METODO"] == "SIN_PAR"].iloc[0]
    assert (sp["ID_ENTIDAD"], int(sp["ID_DISTRITO"])) == ("03", 4)
    assert previos_sin_uso(m, dsA) == ["01 D003 (Puerto Alto)"]


def test_mapeo_manual(datos, tmp_path):
    dsA, dsB = DatasetINE(datos.c2021, ciclo="2021"), DatasetINE(datos.cB, ciclo="2027")
    manual = tmp_path / "manual.csv"
    manual.write_text("ID_ENTIDAD,ID_DISTRITO_NUEVO,ID_DISTRITO_PREV\n03,4,3\n")
    m = construir_mapeo(dsA, dsB, manual)
    met = m["METODO"].value_counts().to_dict()
    assert met.get("MANUAL") == 1
    assert (m[m["ID_DISTRITO"] == 3]["METODO"] == "SIN_PAR").all()   # (03,3) liberado
    assert previos_sin_uso(m, dsA) == ["01 D003 (Puerto Alto)"]


def test_comparador_mapeado(datos):
    dsA, dsB = DatasetINE(datos.c2021, ciclo="2021"), DatasetINE(datos.cB, ciclo="2027")
    comp = ComparadorMapeado(dsA, dsB, construir_mapeo(dsA, dsB))
    assert len(comp.tabla_distritos()) == 8
    assert comp.fuera_nuevo == {("03", 4)}
    assert "01-003" in comp.fuera_prev


# ─────────────── encuestas y proyección ───────────────
def test_encuestas_agrega(datos):
    enc = Encuestas(datos.enc, fecha_referencia=date(2024, 5, 25))
    med, sig, meta = enc.agregado("NACIONAL")
    assert meta["n_encuestas"] == 2 and set(med) == set(BLOQUES)
    assert 90 <= sum(med.values()) <= 110
    assert all(sig[b] >= 1.5 for b in BLOQUES)
    assert "01" in enc.estatales() and ("01", 1) in enc.distritales()


def test_proyeccion_statusquo(datos, tmp_path):
    p = Proyeccion2027(DatasetINE(datos.c2024, ciclo="2024"),
                       fecha_eleccion=date(2027, 6, 6))
    p.simular(600, seed=3)
    t = p.tabla_proyecciones()
    assert len(t) == 9 and t["GANADOR_PROBABLE"].isin(BLOQUES).all()
    assert np.allclose(t[[f"PCT_{b}" for b in p.bloques]].sum(axis=1), 100, atol=.5)
    assert p.sim["escanos"]["media"].sum() == pytest.approx(9)
    assert "Ganador probable" in p.formatear_proyeccion("01", 1)
    assert p.exportar(tmp_path / "p.xlsx").exists()


def test_shock_conserva_varianza_y_ensancha(datos):
    ds = DatasetINE(datos.c2024, ciclo="2024")
    a = Proyeccion2027(ds, fecha_eleccion=date(2027, 6, 6)); a.simular(2000, seed=5)
    c = Proyeccion2027(ds, fecha_eleccion=date(2027, 6, 6))
    c.simular(2000, seed=5, sigma_nacional=2.0, sigma_estatal=1.0, rho_shock=0.8)
    d_p = (a.tabla_proyecciones()["P_GANADOR"]
           - c.tabla_proyecciones()["P_GANADOR"]).abs()
    assert d_p.median() < 4 and d_p.max() < 15
    ancho = lambda s: (s["p95"] - s["p5"]).max()
    assert ancho(c.sim["escanos"]) >= ancho(a.sim["escanos"])


# ─────────────── backtest ───────────────
def test_backtest_statusquo_numeros_exactos(datos, tmp_path):
    bt = Backtest(DatasetINE(datos.c2021, ciclo="2021"),
                  DatasetINE(datos.c2024, ciclo="2024"),
                  fecha_eleccion=date(2024, 6, 2), sims=600, seed=11)
    m = bt.metricas
    assert m["n"] == 9 and m["excluidos"] == 0
    assert m["acierto"] == 88.9 and m["acierto_naive"] == 88.9
    vc = bt.tabla["CATEGORIA"].value_counts().to_dict()
    assert vc.get("Mantenido ✔") == 8 and vc.get("Flip omitido ✘") == 1
    assert m["flips_reales"] == 1 and m["falsa_alarma"] == 0
    assert 0 < m["brier"] < 2
    assert bt.tabla_confiabilidad()["n"].sum() == 9
    esc = bt._tabla_escanos()
    assert esc["REAL"].sum() == 9 and set(esc["BLOQUE"]) == set(BLOQUES)
    assert bt.exportar(tmp_path / "bt.xlsx").exists()


def test_antifuga_excluye_encuestas_posteriores(datos, tmp_path):
    f = tmp_path / "enc_fuga.csv"
    f.write_text("fecha,casa,ambito,muestra,morena,pan,pri,prd,pt,pvem,mc,otros\n"
                 "2024-05-01,X,NACIONAL,1000,42,26,2,1,5,5,14,5\n"
                 "2024-07-01,Y,NACIONAL,1000,42,26,2,1,5,5,14,5\n")
    enc = Encuestas(f, fecha_referencia=date(2024, 6, 1))
    Backtest(DatasetINE(datos.c2021, ciclo="2021"), DatasetINE(datos.c2024, ciclo="2024"),
             encuestas=enc, fecha_eleccion=date(2024, 6, 2), sims=300, seed=1)
    assert len(enc.df) == 1


def test_oracle(datos):
    ds24 = DatasetINE(datos.c2024, ciclo="2024")
    o = EncuestasOracle(ds24, BLOQUES)
    assert abs(sum(o.medias.values()) - 100) < 0.5
    bt = Backtest(DatasetINE(datos.c2021, ciclo="2021"), ds24, oracle=True,
                  fecha_eleccion=date(2024, 6, 2), sims=300, seed=1)
    assert bt.metricas["modo"] == "oracle" and 0 <= bt.metricas["acierto"] <= 100


# ─────────────── calibración y gráficas ───────────────
def test_brier_ece_a_mano():
    df = pd.DataFrame({"GANADOR_REAL": ["A", "A", "B"],
                       "P_A": [1.0, .8, .2], "P_B": [0.0, .2, .8],
                       "P_GANADOR": [100.0, 80.0, 80.0],
                       "ACIERTO": [True, True, False]})
    brier, ece = _brier_ece(df, ["A", "B"])
    assert brier == pytest.approx((0 + .08 + .08) / 3, abs=1e-6)
    assert 0 <= ece <= 100


def test_calibrador(datos, tmp_path):
    cal = Calibrador(DatasetINE(datos.c2021, ciclo="2021"),
                     DatasetINE(datos.c2024, ciclo="2024"),
                     fecha_eleccion=date(2024, 6, 2), sims_fast=250, sims_final=400)
    cal.evaluar_grid([4.0, 6.0], [3.0], [0.0])
    assert len(cal.grid) == 2 and {"brier", "objetivo"} <= set(cal.grid.columns)
    assert len(cal.refinar(k=1)) == 3
    stab = cal.bootstrap(r=6)
    assert len(stab) == 2 and "elegido_%" in stab.columns
    sh = cal.sugerir_shock(candidatos=[{"sigma_nacional": 0.0, "sigma_estatal": 0.0,
                                        "rho_shock": 0.0}])
    assert len(sh) == 1 and "cobertura_p5_p95" in sh.columns
    cfg = cal.escribir_config(tmp_path / "cfg.json")
    assert {"sigma_het_default", "sigma_min", "penalizacion_oficialismo",
            "shock", "meta"} <= set(json.loads(cfg.read_text(encoding="utf-8")))


def test_wall_maps(datos, tmp_path):
    pytest.importorskip("plotly")
    from graficas_ine import wall_map, wall_map_html
    df = DatasetINE(datos.c2024, ciclo="2024").consultar_todos()
    assert wall_map(df, "GANADOR", "Prueba", tmp_path / "m.png").stat().st_size > 10_000
    fig = wall_map_html(df, "GANADOR", "Prueba")
    assert len(fig.data) >= 1


# ─────────────── agente (solo herramientas offline) ───────────────
def test_agente_consulta_local(monkeypatch, datos):
    monkeypatch.setenv("RUTA_DATASET_INE", str(datos.c2024))
    import agente_electoral_mx as ag
    importlib.reload(ag)
    out = ag.consultar_dataset_local("01", 1)
    assert out["disponible"] and "Distrito 1" in out["markdown"]
    assert ag.despachar("herramienta_inexistente", {})["error"]
