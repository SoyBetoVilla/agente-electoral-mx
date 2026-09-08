"""cli.py — Punto de entrada unificado: electoral-mx <comando> [args]."""
import sys

COMANDOS = {
    "agente":    ("agente_electoral_mx", "Agente conversacional (LLM + web + dataset local)"),
    "batch":     ("dataset_ine",         "Carga/consulta/exporta cómputos distritales INE"),
    "comparar":  ("analisis_comparativo", "Comparativo entre ciclos + gráficas"),
    "remapear":  ("remapeo_2027",        "Mapeo de redistritación entre ciclos"),
    "proyectar": ("proyeccion_2027",     "Proyección probabilística 2027"),
    "backtest":  ("backtest",            "Validación retrospectiva de la metodología"),
    "calibrar":  ("auto_calibracion",    "Auto-calibración de σ → config JSON"),
    "noche":     ("noche_electoral",     "Monitoreo en vivo de la noche electoral"),
}


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("electoral-mx — Pipeline electoral México\n\nUso: electoral-mx <comando> [args]\n"
              + "\n".join(f"  {c:<11}{d}" for c, (_, d) in COMANDOS.items()))
        return 0
    cmd, resto = argv[0], argv[1:]
    if cmd not in COMANDOS:
        print(f"❌ Comando desconocido: {cmd}. Usa 'electoral-mx --help'.", file=sys.stderr)
        return 2
    sys.argv = [f"electoral-mx {cmd}", *resto]
    import runpy
    runpy.run_module(COMANDOS[cmd][0], run_name="__main__")
    return 0


def dashboard() -> int:
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("Instala streamlit: pip install streamlit", file=sys.stderr)
        return 1
    extra = [a for a in sys.argv[1:] if a != "dashboard"]
    sys.argv = ["streamlit", "run", "dashboard.py", *extra]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
