"""agente_electoral_mx.py — Agente conversacional: dataset local primero, web después."""
from __future__ import annotations

import json, os, re
from datetime import date
from pathlib import Path

import requests

MODELO = "claude-sonnet-4-5"
MAX_PASOS = 12
TIMEOUT_HTTP = 30
MAX_CARACTERES_PAGINA = 15_000
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AgenteElectoralMX/1.0)"}
RUTA_DATASET = os.environ.get("RUTA_DATASET_INE", "")
RUTA_PREV = os.environ.get("RUTA_DATASET_INE_PREV", "")
CICLO_PREV = os.environ.get("CICLO_PREV", "2021")
RUTA_MAPEO = os.environ.get("RUTA_MAPEO_2027", "")
RUTA_ENCUESTAS = os.environ.get("RUTA_ENCUESTAS_2027", "")
RUTA_CONFIG = os.environ.get("RUTA_CONFIG_PROYECCION", "config_proyeccion.json")
FECHA_ELECCION = os.environ.get("FECHA_ELECCION_2027", "2027-06-06")

ENTIDADES = {"01": "Aguascalientes", "02": "Baja California", "03": "Baja California Sur",
    "04": "Campeche", "05": "Chiapas", "06": "Chihuahua", "07": "Ciudad de México",
    "08": "Coahuila", "09": "Colima", "10": "Durango", "11": "Guanajuato",
    "12": "Guerrero", "13": "Hidalgo", "14": "Jalisco", "15": "Estado de México",
    "16": "Michoacán", "17": "Morelos", "18": "Nayarit", "19": "Nuevo León",
    "20": "Oaxaca", "21": "Puebla", "22": "Querétaro", "23": "Quintana Roo",
    "24": "San Luis Potosí", "25": "Sinaloa", "26": "Sonora", "27": "Tabasco",
    "28": "Tamaulipas", "29": "Tlaxcala", "30": "Veracruz", "31": "Yucatán",
    "32": "Zacatecas"}


def buscar_web(consulta: str, incluir_dominios=None):
    clave = os.environ.get("TAVILY_API_KEY")
    if not clave:
        return {"error": "TAVILY_API_KEY no configurada; usa consultar_dataset_local."}
    payload = {"api_key": clave, "query": consulta, "max_results": 6,
               "search_depth": "advanced"}
    if incluir_dominios:
        payload["include_domains"] = incluir_dominios
    r = requests.post("https://api.tavily.com/search", json=payload, timeout=TIMEOUT_HTTP)
    r.raise_for_status()
    return [{"titulo": x["title"], "url": x["url"], "resumen": x["content"][:500]}
            for x in r.json().get("results", [])]


def obtener_pagina(url: str, extraer_tablas: bool = True):
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {"error": "beautifulsoup4 no instalada."}
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT_HTTP)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
    texto = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n")).strip()[:MAX_CARACTERES_PAGINA]
    tablas = None
    if extraer_tablas:
        partes = []
        for i, tb in enumerate(soup.find_all("table"), 1):
            filas = [" | ".join(c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"]))
                     for tr in tb.find_all("tr")]
            filas = [f for f in filas if f]
            if filas:
                partes.append(f"[TABLA {i}]\n" + "\n".join(filas))
        tablas = "\n\n".join(partes) or None
    return {"url": url, "texto": texto, "tablas": tablas}


_dataset_cache = None


def _dataset():
    global _dataset_cache
    if _dataset_cache is None and RUTA_DATASET and Path(RUTA_DATASET).exists():
        from dataset_ine import DatasetINE
        _dataset_cache = DatasetINE(RUTA_DATASET)
    return _dataset_cache


def consultar_dataset_local(entidad: str = "", distrito: int | None = None,
                            todos: bool = False):
    ds = _dataset()
    if ds is None:
        return {"disponible": False,
                "mensaje": "No hay dataset local (RUTA_DATASET_INE); usa buscar_web."}
    from dataset_ine import formatear_distrito
    if todos:
        df = ds.consultar_todos()
        return {"disponible": True, "total_distritos": len(df),
                "filas": df.head(30).to_dict("records")}
    res = ds.consultar(entidad, distrito)
    return {"disponible": True, "markdown": formatear_distrito(res), "datos": res}


def comparar_ciclos(entidad: str = "", distrito: int | None = None):
    if not (RUTA_PREV and Path(RUTA_PREV).exists()
            and RUTA_DATASET and Path(RUTA_DATASET).exists()):
        return {"disponible": False,
                "mensaje": "Configura RUTA_DATASET_INE y RUTA_DATASET_INE_PREV."}
    from analisis_comparativo import ComparadorINE
    from dataset_ine import DatasetINE
    prev = DatasetINE(RUTA_PREV, ciclo=CICLO_PREV)
    act = DatasetINE(RUTA_DATASET)
    if RUTA_MAPEO and Path(RUTA_MAPEO).exists():
        import pandas as pd
        from remapeo_2027 import ComparadorMapeado
        comp = ComparadorMapeado(prev, act,
                                 pd.read_csv(RUTA_MAPEO, dtype={"ID_ENTIDAD": str}))
    else:
        comp = ComparadorINE(prev, act)
    t = comp.tabla_distritos()
    if not entidad:
        res = comp.resumen_nacional()
        return {"disponible": True, "cambios_de_bloque": int(t["CAMBIO_GANADOR"].sum()),
                "total_distritos": len(t),
                "nacional_bloques": res["bloques"].to_dict("index"),
                "top10_volatiles": comp.top_volatiles(10).to_dict("records")}
    f = t[t["ID_ENTIDAD"] == str(entidad).zfill(2)]
    if distrito:
        f = f[f["ID_DISTRITO"] == distrito]
    if f.empty:
        return {"disponible": True, "filas": 0, "aviso": "Entidad/distrito sin datos"}
    return {"disponible": True, "filas": len(f), "datos": f.to_dict("records")}


def proyectar_2027(entidad: str = "", distrito: int | None = None,
                   solo_competitivos: bool = False):
    if not (RUTA_DATASET and Path(RUTA_DATASET).exists()):
        return {"disponible": False, "mensaje": "Falta RUTA_DATASET_INE."}
    from dataset_ine import DatasetINE
    from proyeccion_2027 import Encuestas, Proyeccion2027
    cfg = (json.loads(Path(RUTA_CONFIG).read_text(encoding="utf-8"))
           if Path(RUTA_CONFIG).exists() else {})
    enc = (Encuestas(RUTA_ENCUESTAS)
           if RUTA_ENCUESTAS and Path(RUTA_ENCUESTAS).exists() else None)
    p = Proyeccion2027(DatasetINE(RUTA_DATASET), encuestas=enc,
                       fecha_eleccion=date.fromisoformat(FECHA_ELECCION),
                       penalizacion_oficialismo=cfg.get("penalizacion_oficialismo", 0.0),
                       sigma_het_default=cfg.get("sigma_het_default", 4.0),
                       sigma_min=cfg.get("sigma_min", 2.5))
    p.simular(**cfg.get("shock", {}))
    if solo_competitivos:
        return {"disponible": True, "competitivos": p.competitivos(20).to_dict("records")}
    if entidad and distrito:
        return {"disponible": True, "ficha": p.formatear_proyeccion(entidad, distrito)}
    return {"disponible": True, "resumen": p.resumen(),
            "escanos": p.sim["escanos"].round(1).to_dict("index")}


def despachar(nombre: str, args: dict):
    try:
        if nombre == "buscar_web":
            return buscar_web(args.get("consulta", ""), args.get("incluir_dominios"))
        if nombre == "obtener_pagina":
            return obtener_pagina(args["url"], args.get("extraer_tablas", True))
        if nombre == "consultar_dataset_local":
            return consultar_dataset_local(args.get("entidad", ""), args.get("distrito"),
                                           args.get("todos", False))
        if nombre == "comparar_ciclos":
            return comparar_ciclos(args.get("entidad", ""), args.get("distrito"))
        if nombre == "proyectar_2027":
            return proyectar_2027(args.get("entidad", ""), args.get("distrito"),
                                  args.get("solo_competitivos", False))
        return {"error": f"Herramienta desconocida: {nombre}"}
    except Exception as e:
        return {"error": str(e)}


SYSTEM_PROMPT = f"""Eres un analista electoral experto en México. Respondes preguntas sobre
resultados por DISTRITO ELECTORAL FEDERAL (300 distritos, 32 entidades) y proyecciones 2027.

CONTEXTO
- Entidades (código INE → nombre): {json.dumps(ENTIDADES, ensure_ascii=False)}
- Elección objetivo: 6 de junio de 2027 (intermedia, Cámara de Diputados). Si preguntan por
  resultados 2027 y aún no existen, dilo y ofrece el ciclo disponible del dataset local.
- En 2024 compitieron MORENA, PAN, PRI, PRD, PT, PVEM y MC, en coaliciones
  ("Sigamos Haciendo Historia": MORENA-PT-PVEM; "Fuerza y Corazón por México": PAN-PRI-PRD;
  MC solo). El PRD perdió su registro tras 2024; verifica qué partidos aparecen en cada ciclo.

PROCEDIMIENTO
1. PRIMERO intenta consultar_dataset_local (cómputos oficiales del INE, cifras exactas).
2. Usa comparar_ciclos para preguntas entre ciclos (swings, flips, volatilidad) y
   proyectar_2027 para escenarios probabilísticos — preséntalos SIEMPRE como "ganador
   probable" con su P(victoria), nunca como resultado.
3. Solo si no hay dataset local, usa buscar_web priorizando dominios oficiales
   (["ine.mx", "computos.ine.mx", "resultadoselectorales.ine.mx"]) y abre con obtener_pagina.
4. Nunca inventes cifras; si un dato no aparece escribe "no disponible" y cita la URL.

FORMATO: tabla Markdown por distrito (Partido | Votos | %), ganador, total, fuentes con URL
y fecha de consulta, y notas (preliminar/definitivo, fuente secundaria)."""

TOOLS = [
    {"name": "consultar_dataset_local",
     "description": "Cómputos distritales oficiales del INE ya descargados (cifras exactas). USAR PRIMERO.",
     "input_schema": {"type": "object", "properties": {
         "entidad": {"type": "string", "description": "Nombre o número (01-32)"},
         "distrito": {"type": "integer"},
         "todos": {"type": "boolean", "description": "true = resumen de todos los distritos"}}}},
    {"name": "comparar_ciclos",
     "description": "Compara dos ciclos del dataset local: Δpp, volatilidad, flips y swings.",
     "input_schema": {"type": "object", "properties": {
         "entidad": {"type": "string"}, "distrito": {"type": "integer"}}}},
    {"name": "proyectar_2027",
     "description": "Escenario probabilístico 2027: ganador probable, % esperado y "
                    "P(victoria) por distrito, o distribución de escaños.",
     "input_schema": {"type": "object", "properties": {
         "entidad": {"type": "string"}, "distrito": {"type": "integer"},
         "solo_competitivos": {"type": "boolean"}}}},
    {"name": "buscar_web",
     "description": "Búsqueda web (títulos, URLs, resúmenes). Solo si no hay dataset local.",
     "input_schema": {"type": "object", "properties": {
         "consulta": {"type": "string"},
         "incluir_dominios": {"type": "array", "items": {"type": "string"}}},
         "required": ["consulta"]}},
    {"name": "obtener_pagina",
     "description": "Descarga una URL y devuelve su texto y tablas.",
     "input_schema": {"type": "object", "properties": {
         "url": {"type": "string"}, "extraer_tablas": {"type": "boolean"}},
         "required": ["url"]}},
]


def ejecutar_agente(pregunta: str) -> str:
    try:
        import anthropic
    except ImportError:
        return "⚠️ Falta `anthropic`. Instala con: pip install anthropic"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "⚠️ Configura ANTHROPIC_API_KEY (en Secrets de la app o variable de entorno)."
    client = anthropic.Anthropic()
    mensajes = [{"role": "user", "content": pregunta}]
    for _ in range(MAX_PASOS):
        resp = client.messages.create(model=MODELO, max_tokens=4096, system=SYSTEM_PROMPT,
                                      tools=TOOLS, messages=mensajes)
        mensajes.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "tool_use":
            resultados = []
            for bloque in resp.content:
                if bloque.type == "tool_use":
                    salida = despachar(bloque.name, bloque.input)
                    resultados.append({"type": "tool_result", "tool_use_id": bloque.id,
                                       "content": json.dumps(salida, ensure_ascii=False)[:20000]})
            mensajes.append({"role": "user", "content": resultados})
        else:
            return "".join(b.text for b in resp.content if b.type == "text")
    return "⚠️ Se alcanzó el límite de pasos sin respuesta final."


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Agente Electoral MX")
    ap.add_argument("pregunta", nargs="*")
    a = ap.parse_args()
    pregunta = " ".join(a.pregunta) or input("Pregunta: ")
    print(f"\n🔎 {pregunta}\n")
    print(ejecutar_agente(pregunta))
