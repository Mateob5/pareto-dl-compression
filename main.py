"""
main.py
=======
Punto de entrada principal del pipeline de optimización de Deep Learning.

Modos de uso:
    python main.py verificar         # Verifica entorno (GPU, CUDA, deps)
    python main.py demo              # Ejecuta 4 configs clave (~20 min)
    python main.py completo          # Ejecuta las 22 configs (~4-6 horas)
    python main.py solo-pareto       # Solo genera visualización de Pareto
    python main.py listar            # Lista las 22 configuraciones

Ejemplo:
    python main.py demo
    python main.py solo-pareto --csv resultados/resultados_referencia.csv

Autor: Jerónimo Bazán (2243590)
Proyecto: Optimización Computacional de Modelos de Deep Learning
Universidad Industrial de Santander
"""

import argparse
import os
import sys

# Añadir raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.experimentos.configuraciones import CONFIGURACIONES, listar_configuraciones


def cmd_verificar(args):
    """Ejecuta la verificación del entorno."""
    print("\n>>> Verificando entorno...\n")
    import subprocess
    script = os.path.join(config.RAIZ_PROYECTO, "verificar_entorno.py")
    subprocess.call([sys.executable, script])


def cmd_demo(args):
    """Ejecuta 4 configuraciones clave para demostración rápida."""
    from src.experimentos.pipeline import ejecutar_todas_configuraciones
    configs_demo = ["B0", "Q1", "D1", "D1P3Q2"]
    print(f"\n>>> MODO DEMO: {len(configs_demo)} configuraciones")
    print(f">>> Configuraciones: {configs_demo}")
    print(f">>> Tiempo estimado: 15-30 minutos en RTX 4060\n")
    ejecutar_todas_configuraciones(ids_configuraciones=configs_demo)


def cmd_completo(args):
    """Ejecuta las 22 configuraciones experimentales completas."""
    from src.experimentos.pipeline import ejecutar_todas_configuraciones
    print(f"\n>>> MODO COMPLETO: 22 configuraciones")
    print(f">>> Tiempo estimado: 4-6 horas en RTX 4060 Laptop\n")
    ejecutar_todas_configuraciones()


def cmd_solo_pareto(args):
    """Solo genera la visualización de Pareto desde un CSV existente."""
    from src.analisis.pareto import ejecutar_analisis_completo
    csv_path = args.csv if args.csv else None
    ejecutar_analisis_completo(ruta_csv=csv_path)


def cmd_listar(args):
    """Lista las 22 configuraciones experimentales."""
    listar_configuraciones()


def cmd_limpiar_cache(args):
    """Elimina el cache del modelo destilado para forzar reentrenamiento."""
    from src.experimentos.pipeline import limpiar_cache_destilado
    limpiar_cache_destilado()


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de optimización de Deep Learning: "
                    "Cuantización + Poda + Destilación + Frontera de Pareto",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    python main.py verificar
    python main.py demo
    python main.py completo
    python main.py solo-pareto --csv resultados/resultados_referencia.csv
    python main.py listar
    python main.py limpiar-cache    # Borra cache del destilado (forzar reentrenar D1)
        """,
    )

    subparsers = parser.add_subparsers(dest="modo", help="Modo de ejecución")

    # Subcomando: verificar
    sp_verificar = subparsers.add_parser("verificar", help="Verifica entorno (GPU, CUDA, deps)")

    # Subcomando: demo
    sp_demo = subparsers.add_parser("demo", help="Demo rápida con 4 configs clave")

    # Subcomando: completo
    sp_completo = subparsers.add_parser("completo", help="Ejecuta las 22 configuraciones")

    # Subcomando: solo-pareto
    sp_pareto = subparsers.add_parser("solo-pareto", help="Solo visualización de Pareto")
    sp_pareto.add_argument("--csv", type=str, default=None,
                          help="Ruta al CSV de resultados (default: resultados_usuario.csv "
                               "o resultados_referencia.csv)")

    # Subcomando: listar
    sp_listar = subparsers.add_parser("listar", help="Lista las 22 configuraciones")

    # Subcomando: limpiar-cache
    sp_cache = subparsers.add_parser("limpiar-cache",
        help="Borra el cache del destilado (fuerza reentrenar D1)")

    args = parser.parse_args()

    if args.modo is None:
        parser.print_help()
        return

    # Mostrar info del entorno
    info = config.info_entorno()
    print("\n" + "=" * 70)
    print("PIPELINE DE OPTIMIZACIÓN DE DEEP LEARNING")
    print("Cuantización + Poda + Destilación + Frontera de Pareto")
    print("=" * 70)
    for k, v in info.items():
        print(f"  {k:<25}: {v}")
    print("=" * 70)

    # Ejecutar comando
    if args.modo == "verificar":
        cmd_verificar(args)
    elif args.modo == "demo":
        cmd_demo(args)
    elif args.modo == "completo":
        cmd_completo(args)
    elif args.modo == "solo-pareto":
        cmd_solo_pareto(args)
    elif args.modo == "listar":
        cmd_listar(args)
    elif args.modo == "limpiar-cache":
        cmd_limpiar_cache(args)


if __name__ == "__main__":
    main()
