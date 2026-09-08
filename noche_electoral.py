"""noche_electoral.py — Monitoreo en vivo de la noche electoral (PREP → informe/alertas)."""
from __future__ import annotations

import argparse, hashlib, json, time
from dataclasses import dataclass, fields
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from analisis_comparativo import BLOQUE_DE_PARTIDO, PARTIDOS_TOKEN, matriz_ciclo
from dataset_ine import DatasetINE
from proyeccion_2027 import Encuestas, Proyeccion2027

DISTRICTOS_TOTAL = 300
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NocheElectoralMX/1.0)"}


@dataclass
class ConfigNoche:
    fuente_url: str | None = None
    fuente_archivo: str | None = None
    intervalo_seg: int = 120
    salida_dir: Path = Path("noche")
    base: str = ""
    ciclo_base: str = "2024"
    encuestas: str | None = None
    config_proyeccion: str | None = None
    seed: int = 42
    sims: int = 10000
    margen_alerta_pp: float = 3.0
    webhook: str | None = None
    aviso_sorpresas: bool = True

    @classmethod
    def desde_json(cls, ruta) -> "ConfigNoche":
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        cfg = cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})
        cfg.salida_dir = Path(cfg.salida_dir)
        return cfg


def cargar_proyeccion(cfg: ConfigNoche):
    """Proyección congelada UNA vez antes del primer tick (con config si existe)."""
    bloques = sorted({b for b in BLOQUE_DE_PARTIDO.values() if b != "OTROS"})
    if not cfg.base or not Path(cfg.base).exists():
        print("   ⚠️ sin dataset base: comparación contra proyección desactivada.")
        return None, bloques
    cfgp = (json.loads(Path(cfg.config_proyeccion).read_text(encoding="utf-8"))
            if cfg.config_proyeccion and Path(cfg.config_proyeccion).exists() else {})
    enc = (Encuestas(cfg.encuestas)
           if cfg.encuestas and Path(cfg.encuestas).exists() else None)
    p = Proyeccion2027(DatasetINE(cfg.base, ciclo=cfg.ciclo_base), encuestas=enc,
                       fecha_eleccion=date(2027, 6, 6),
                       penalizacion_oficialismo=cfgp.get("penalizacion_oficialismo", 0.0),
                       sigma_het_default=cfgp.get("sigma_het_default", 4.0),
                       sigma_min=cfgp.get("sigma_min", 2.5))
    p.simular(cfg.sims, cfg.seed, **cfgp.get("shock", {}))
    print(f"🔒 Proyección congelada ({cfg.sims:,} sims; encuestas: {'sí' if enc else 'no'}).")
    t = p.tabla_proyecciones()
    t["CLAVE"] = (t["ID_ENTIDAD"].astype(str).str.zfill(2) + "-"
                  + t["ID_DISTRITO"].astype(int).astype(str).str.zfill(3))
    return t, bloques


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def obtener_snapshot(cfg: ConfigNoche, ultimo_hash):
    if cfg.fuente_archivo:
        p = Path(cfg.fuente_archivo)
        if not p.exists():
            return None, ultimo_hash
        data = p.read_bytes()
    else:
        r = requests.get(cfg.fuente_url, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.content
    h = _sha256(data)
    if h == ultimo_hash:
        return None, h
    cfg.salida_dir.mkdir(parents=True, exist_ok=True)
    sufijo = ".zip" if data[:2] == b"PK" else ".csv"
    ruta = cfg.salida_dir / f"snapshot_{h}{sufijo}"
    ruta.write_bytes(data)
    return ruta, h


def _cols_actas(df: pd.DataFrame):
    up = {c.upper(): c for c in df.columns}
    cap = next((o for k, o in up.items()
                if "ACTAS" in k and any(x in k for x in ("CAPT", "CONTAB", "PROCES"))), None)
    tot = next((o for k, o in up.items()
                if "ACTAS" in k and any(x in k for x in ("TOTAL", "ESPER"))), None)
    if cap and tot:
        return cap, tot
    pct = next((o for k, o in up.items()
                if ("PORCENT" in k or "PCT" in k) and "ACTAS" in k), None)
    return (pct, "__PCT__") if pct else None


def analizar_snapshot(ruta_csv: Path, bloques: list[str]):
    ds = DatasetINE(ruta_csv, ciclo="2027")
    m = matriz_ciclo(ds)
    V = pd.DataFrame(index=m.index)
    for b in bloques:
        toks = [t for t in PARTIDOS_TOKEN if BLOQUE_DE_PARTIDO.get(t) == b and t in m.columns]
        V[b] = m[toks].sum(axis=1) if toks else 0.0
    pct = 100 * V.div(V.sum(axis=1).replace(0, np.nan), axis=0)
    gan = V.idxmax(axis=1)
    s = np.sort(pct.values, axis=1)
    with np.errstate(invalid="ignore"):
        margen = pd.Series(np.round(s[:, -1] - s[:, -2], 2), index=m.index)

    avance = {"distritos_contados": int(V.sum(axis=1).gt(0).sum()),
              "pct_distritos": round(100 * V.sum(axis=1).gt(0).sum() / DISTRICTOS_TOTAL, 1)}
    par = _cols_actas(ds.df)
    if par:
        cap, tot = par
        if tot == "__PCT__":
            avance["pct_actas"] = round(float(pd.to_numeric(ds.df[cap],
                                                            errors="coerce").mean()), 1)
        else:
            t = float(pd.to_numeric(ds.df[tot], errors="coerce").sum())
            if t > 0:
                avance["pct_actas"] = round(100 * float(pd.to_numeric(
                    ds.df[cap], errors="coerce").sum()) / t, 1)

    actual = pd.DataFrame({"CLAVE": m["CLAVE"], "ID_ENTIDAD": m["ID_ENTIDAD"],
                           "ID_DISTRITO": m["ID_DISTRITO"], "ENTIDAD": m["ENTIDAD"],
                           "GANADOR_ACTUAL": gan, "MARGEN_ACTUAL": margen})
    for b in bloques:
        actual[f"ACTUAL_PCT_{b}"] = pct[b].round(1).values
    return actual, avance


def comparar(actual: pd.DataFrame, tabla_proy, bloques: list[str], margen_alerta: float) -> dict:
    if tabla_proy is None:
        return {"sorpresas": pd.DataFrame(), "cerrados": actual.nsmallest(15, "MARGEN_ACTUAL"),
                "escanos": None}
    f = actual.merge(tabla_proy[["CLAVE", "GANADOR_PROBABLE", "P_GANADOR", "CLASIFICACION"]],
                     on="CLAVE", how="inner")
    sorp = f[f["GANADOR_ACTUAL"] != f["GANADOR_PROBABLE"]].sort_values(
        "P_GANADOR", ascending=False)
    cer = f[f["MARGEN_ACTUAL"] < margen_alerta].sort_values("MARGEN_ACTUAL")
    esc_act = f["GANADOR_ACTUAL"].value_counts().reindex(bloques).fillna(0).astype(int)
    esc_proy = f["GANADOR_PROBABLE"].value_counts().reindex(bloques).fillna(0).astype(int)
    return {"sorpresas": sorp, "cerrados": cer,
            "escanos": pd.DataFrame({"ACTUAL": esc_act, "PROYECCION": esc_proy})}


def escribir_informes(cfg: ConfigNoche, ts, avance, cmp_res, bloques):
    d = cfg.salida_dir
    d.mkdir(parents=True, exist_ok=True)
    sorp, cer, esc = cmp_res["sorpresas"], cmp_res["cerrados"], cmp_res["escanos"]
    md = [f"# 🌙 Noche electoral — informe {ts:%d/%m/%Y %H:%M:%S}",
          f"**Avance:** {avance['distritos_contados']}/{DISTRICTOS_TOTAL} distritos "
          f"({avance['pct_distritos']}%)"
          + (f" · actas ≈ {avance['pct_actas']}%" if "pct_actas" in avance else ""),
          "", "## 🏛️ Escaños por bloque (distritos ya contados)"]
    if esc is not None:
        md.append("| Bloque | Actual | Proyección decía |")
        md += [f"| {b} | {r['ACTUAL']} | {r['PROYECCION']} |" for b, r in esc.iterrows()]
    else:
        md.append("_sin proyección cargada_")
    if cfg.aviso_sorpresas and len(sorp):
        md += ["", f"## ⚠️ Sorpresas ({len(sorp)}) — ganador ≠ proyección",
               sorp.head(12)[["ENTIDAD", "ID_DISTRITO", "GANADOR_PROBABLE", "P_GANADOR",
                              "GANADOR_ACTUAL", "MARGEN_ACTUAL"]].to_markdown(index=False)]
    if len(cer):
        md += ["", f"## 🤏 Demasiado cerrado para cantar (< {cfg.margen_alerta} pp)",
               cer.head(12)[["ENTIDAD", "ID_DISTRITO", "GANADOR_ACTUAL",
                             "MARGEN_ACTUAL"]].to_markdown(index=False)]
    (d / "informe_actual.md").write_text("\n".join(md), encoding="utf-8")
    (d / f"informe_{ts:%Y%m%d_%H%M%S}.md").write_text("\n".join(md), encoding="utf-8")
    estado = {"ts": ts.isoformat(), "avance": avance,
              "escanos": esc.to_dict("index") if esc is not None else None,
              "n_sorpresas": int(len(sorp)), "n_cerrados": int(len(cer)),
              "sorpresas_top": sorp.head(10)[["ENTIDAD", "ID_DISTRITO", "GANADOR_PROBABLE",
                                              "P_GANADOR", "GANADOR_ACTUAL"]]
              .to_dict("records") if len(sorp) else []}
    (d / "estado.json").write_text(json.dumps(estado, ensure_ascii=False, indent=2),
                                   encoding="utf-8")


def notificar(cfg: ConfigNoche, tipo: str, payload: dict):
    if not cfg.webhook:
        return
    try:
        requests.post(cfg.webhook, json={"tipo": tipo, **payload}, timeout=10)
    except Exception as e:
        print(f"   ⚠️ webhook falló: {e}")


def tick(cfg: ConfigNoche, estado: dict, tabla_proy, bloques: list[str]) -> dict:
    ts = datetime.now()
    ruta, h = obtener_snapshot(cfg, estado.get("hash"))
    if ruta is None:
        print(f"[{ts:%H:%M:%S}] sin cambios ({h}).")
        return estado
    print(f"[{ts:%H:%M:%S}] 📥 nuevo snapshot {h} → analizando…")
    try:
        actual, avance = analizar_snapshot(ruta, bloques)
    except Exception as e:
        print(f"   ⚠️ snapshot aún ilegible: {e}")
        return {**estado, "hash": h}
    cmp_res = comparar(actual, tabla_proy, bloques, cfg.margen_alerta_pp)
    escribir_informes(cfg, ts, avance, cmp_res, bloques)

    nsor = int(len(cmp_res["sorpresas"]))
    print(f"   🗳️ {avance['distritos_contados']}/{DISTRICTOS_TOTAL} distritos · "
          f"⚠️ {nsor} sorpresas · 🤏 {len(cmp_res['cerrados'])} cerrados")

    fila = {"ts": ts.isoformat(), "hash": h, **avance, "sorpresas": nsor}
    if cmp_res["escanos"] is not None:
        fila.update({f"ACT_{b}": v for b, v in cmp_res["escanos"]["ACTUAL"].items()})
    tl = cfg.salida_dir / "timeline.csv"
    pd.DataFrame([fila]).to_csv(tl, mode="a", header=not tl.exists(), index=False)

    previas = {s["CLAVE"] for s in estado.get("sorpresas_vistas", [])}
    nuevas = [r for r in cmp_res["sorpresas"].to_dict("records") if r["CLAVE"] not in previas]
    if cfg.aviso_sorpresas and nuevas:
        notificar(cfg, "sorpresa", {"avance": avance, "nuevas": nuevas[:8]})
    for hi in (25, 50, 75, 90, 100):
        if avance["pct_distritos"] >= hi and estado.get("hito", 0) < hi:
            notificar(cfg, "hito", {"pct": hi, "avance": avance,
                                    "escanos": cmp_res["escanos"].to_dict("index")
                                    if cmp_res["escanos"] is not None else None})
            estado["hito"] = hi
    return {**estado, "hash": h,
            "sorpresas_vistas": cmp_res["sorpresas"][["CLAVE"]].to_dict("records")}


def main():
    ap = argparse.ArgumentParser(description="Noche electoral en vivo")
    ap.add_argument("--config")
    ap.add_argument("--fuente-url", dest="fuente_url")
    ap.add_argument("--fuente-archivo", dest="fuente_archivo")
    ap.add_argument("--base"); ap.add_argument("--encuestas")
    ap.add_argument("--intervalo", type=int, dest="intervalo_seg")
    ap.add_argument("--una-vez", action="store_true")
    ap.add_argument("--margen-alerta", type=float, dest="margen_alerta_pp")
    ap.add_argument("--webhook"); ap.add_argument("--sin-webhook", action="store_true")
    a = ap.parse_args()
    cfg = ConfigNoche.desde_json(a.config) if a.config else ConfigNoche()
    for k in ("fuente_url", "fuente_archivo", "base", "encuestas", "intervalo_seg",
              "margen_alerta_pp", "webhook"):
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)
    if a.sin_webhook:
        cfg.webhook = None
    if not (cfg.fuente_url or cfg.fuente_archivo):
        ap.error("configura --fuente-url/--fuente-archivo o usa --config noche.json")
    tabla_proy, bloques = cargar_proyeccion(cfg)
    estado = {"hash": None, "hito": 0, "sorpresas_vistas": []}
    print(f"🌙 Noche electoral activa — fuente: {cfg.fuente_url or cfg.fuente_archivo} · "
          f"tick cada {cfg.intervalo_seg}s · salida: {cfg.salida_dir.resolve()}")
    while True:
        try:
            estado = tick(cfg, estado, tabla_proy, bloques)
        except requests.RequestException as e:
            print(f"   ⚠️ red: {e} (reintento en el próximo tick)")
        except Exception as e:
            print(f"   ⚠️ error inesperado: {e}")
        if a.una_vez:
            break
        try:
            time.sleep(cfg.intervalo_seg)
        except KeyboardInterrupt:
            print("\n👋 Noche terminada."); break


if __name__ == "__main__":
    main()
