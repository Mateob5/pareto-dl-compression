"""
src/experimentos/configuraciones.py
===================================
Definición de las 22 configuraciones experimentales del proyecto.

Estructura:
- B0:  Línea base (BERT-base FP32)
- Q1:  BERT-base INT8 (PTQ con bitsandbytes)
- Q2:  BERT-base INT4 (PTQ con bitsandbytes)
- P1:  BERT-base + Poda estructurada 20%
- P2:  BERT-base + Poda estructurada 40%
- P3:  BERT-base + Poda estructurada 60%
- P1Q1: BERT-base + Poda 20% + INT8
- P2Q1: BERT-base + Poda 40% + INT8
- P3Q1: BERT-base + Poda 60% + INT8
- D1:  DistilBERT destilado FP32
- D1Q1: DistilBERT + INT8
- D1Q2: DistilBERT + INT4
- D1P1: DistilBERT + Poda 20%
- D1P2: DistilBERT + Poda 40%
- D1P3: DistilBERT + Poda 60%
- D1P1Q1: DistilBERT + Poda 20% + INT8
- D1P2Q1: DistilBERT + Poda 40% + INT8
- D1P3Q1: DistilBERT + Poda 60% + INT8
- D1P1Q2: DistilBERT + Poda 20% + INT4
- D1P2Q2: DistilBERT + Poda 40% + INT4
- D1P3Q2: DistilBERT + Poda 60% + INT4
- D1P3Q2FT: DistilBERT + Poda 60% + INT4 + fine-tuning adicional

Autor: Jerónimo Bazán (2243590)
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfiguracionExperimental:
    """Estructura de una configuración experimental del pipeline."""
    id: str                                   # Identificador (e.g. "B0", "D1P1Q1")
    nombre: str                               # Nombre descriptivo
    modelo_base: str                          # "bert-base-uncased" o "distilbert-base-uncased"
    cuantizacion: Optional[str] = None        # None, "INT8", "INT4"
    poda_pct: Optional[float] = None          # None, 0.20, 0.40, 0.60
    destilado: bool = False                   # True si usa estudiante destilado
    fine_tuning_extra: bool = False           # True si requiere fine-tuning adicional
    descripcion: str = ""                     # Descripción humana


# =============================================================================
# LAS 22 CONFIGURACIONES EXPERIMENTALES
# =============================================================================
CONFIGURACIONES = [
    # --- LÍNEA BASE ---
    ConfiguracionExperimental(
        id="B0",
        nombre="Línea Base BERT-base FP32",
        modelo_base="bert-base-uncased",
        descripcion="Modelo original sin modificar. Punto de comparación B0.",
    ),

    # --- CUANTIZACIÓN ---
    ConfiguracionExperimental(
        id="Q1",
        nombre="BERT-base INT8 (PTQ)",
        modelo_base="bert-base-uncased",
        cuantizacion="INT8",
        descripcion="Post-Training Quantization a INT8 con bitsandbytes.",
    ),
    ConfiguracionExperimental(
        id="Q2",
        nombre="BERT-base INT4 (PTQ NF4)",
        modelo_base="bert-base-uncased",
        cuantizacion="INT4",
        descripcion="PTQ a INT4 con NormalFloat4 y double quantization.",
    ),

    # --- PODA ---
    ConfiguracionExperimental(
        id="P1",
        nombre="BERT-base + Poda 20%",
        modelo_base="bert-base-uncased",
        poda_pct=0.20,
        descripcion="Poda estructurada del 20% de cabezas de atención + fine-tuning.",
    ),
    ConfiguracionExperimental(
        id="P2",
        nombre="BERT-base + Poda 40%",
        modelo_base="bert-base-uncased",
        poda_pct=0.40,
        descripcion="Poda estructurada del 40% de cabezas de atención + fine-tuning.",
    ),
    ConfiguracionExperimental(
        id="P3",
        nombre="BERT-base + Poda 60%",
        modelo_base="bert-base-uncased",
        poda_pct=0.60,
        descripcion="Poda estructurada del 60% de cabezas de atención + fine-tuning.",
    ),

    # --- COMBINACIONES BERT (Poda + Cuantización) ---
    ConfiguracionExperimental(
        id="P1Q1",
        nombre="BERT-base + Poda 20% + INT8",
        modelo_base="bert-base-uncased",
        cuantizacion="INT8",
        poda_pct=0.20,
        descripcion="Poda 20% + cuantización INT8. Combinación moderada.",
    ),
    ConfiguracionExperimental(
        id="P2Q1",
        nombre="BERT-base + Poda 40% + INT8",
        modelo_base="bert-base-uncased",
        cuantizacion="INT8",
        poda_pct=0.40,
        descripcion="Poda 40% + INT8. Balance ahorro/precisión.",
    ),
    ConfiguracionExperimental(
        id="P3Q1",
        nombre="BERT-base + Poda 60% + INT8",
        modelo_base="bert-base-uncased",
        cuantizacion="INT8",
        poda_pct=0.60,
        descripcion="Poda 60% + INT8. Compresión agresiva.",
    ),

    # --- DESTILACIÓN (DistilBERT) ---
    ConfiguracionExperimental(
        id="D1",
        nombre="DistilBERT destilado FP32",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        descripcion="Estudiante destilado de BERT-base. ~40% más pequeño.",
    ),

    # --- DESTILACIÓN + CUANTIZACIÓN ---
    ConfiguracionExperimental(
        id="D1Q1",
        nombre="DistilBERT + INT8",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        cuantizacion="INT8",
        descripcion="Estudiante destilado + INT8. Alta compresión sin perder precisión.",
    ),
    ConfiguracionExperimental(
        id="D1Q2",
        nombre="DistilBERT + INT4",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        cuantizacion="INT4",
        descripcion="Estudiante destilado + INT4. Máxima compresión numérica.",
    ),

    # --- DESTILACIÓN + PODA ---
    ConfiguracionExperimental(
        id="D1P1",
        nombre="DistilBERT + Poda 20%",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        poda_pct=0.20,
        descripcion="Estudiante destilado + poda estructurada 20%.",
    ),
    ConfiguracionExperimental(
        id="D1P2",
        nombre="DistilBERT + Poda 40%",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        poda_pct=0.40,
        descripcion="Estudiante destilado + poda estructurada 40%.",
    ),
    ConfiguracionExperimental(
        id="D1P3",
        nombre="DistilBERT + Poda 60%",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        poda_pct=0.60,
        descripcion="Estudiante destilado + poda estructurada 60%.",
    ),

    # --- DESTILACIÓN + PODA + CUANTIZACIÓN INT8 ---
    ConfiguracionExperimental(
        id="D1P1Q1",
        nombre="DistilBERT + Poda 20% + INT8",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        poda_pct=0.20,
        cuantizacion="INT8",
        descripcion="Combinación: destilación + poda 20% + INT8.",
    ),
    ConfiguracionExperimental(
        id="D1P2Q1",
        nombre="DistilBERT + Poda 40% + INT8",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        poda_pct=0.40,
        cuantizacion="INT8",
        descripcion="Combinación: destilación + poda 40% + INT8.",
    ),
    ConfiguracionExperimental(
        id="D1P3Q1",
        nombre="DistilBERT + Poda 60% + INT8",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        poda_pct=0.60,
        cuantizacion="INT8",
        descripcion="Combinación: destilación + poda 60% + INT8.",
    ),

    # --- DESTILACIÓN + PODA + CUANTIZACIÓN INT4 ---
    ConfiguracionExperimental(
        id="D1P1Q2",
        nombre="DistilBERT + Poda 20% + INT4",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        poda_pct=0.20,
        cuantizacion="INT4",
        descripcion="Combinación: destilación + poda 20% + INT4.",
    ),
    ConfiguracionExperimental(
        id="D1P2Q2",
        nombre="DistilBERT + Poda 40% + INT4",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        poda_pct=0.40,
        cuantizacion="INT4",
        descripcion="Combinación: destilación + poda 40% + INT4.",
    ),
    ConfiguracionExperimental(
        id="D1P3Q2",
        nombre="DistilBERT + Poda 60% + INT4",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        poda_pct=0.60,
        cuantizacion="INT4",
        descripcion="Combinación: destilación + poda 60% + INT4. Compresión extrema.",
    ),

    # --- CONFIGURACIÓN ADICIONAL CON FINE-TUNING EXTRA ---
    ConfiguracionExperimental(
        id="D1P3Q2FT",
        nombre="DistilBERT + Poda 60% + INT4 + FT",
        modelo_base="distilbert-base-uncased",
        destilado=True,
        poda_pct=0.60,
        cuantizacion="INT4",
        fine_tuning_extra=True,
        descripcion="Configuración extrema + fine-tuning adicional para recuperar precisión.",
    ),
]


def listar_configuraciones() -> None:
    """Imprime tabla con las 22 configuraciones."""
    print(f"\n{'ID':<10} {'Modelo':<25} {'Cuant.':<8} {'Poda':<6} {'Dest.':<7} {'FT':<4} {'Descripción'}")
    print("-" * 100)
    for cfg in CONFIGURACIONES:
        cuant = cfg.cuantizacion or "-"
        poda = f"{cfg.poda_pct*100:.0f}%" if cfg.poda_pct else "-"
        dest = "Sí" if cfg.destilado else "No"
        ft = "Sí" if cfg.fine_tuning_extra else "No"
        modelo = "DistilBERT" if "distilbert" in cfg.modelo_base else "BERT-base"
        print(f"{cfg.id:<10} {modelo:<25} {cuant:<8} {poda:<6} {dest:<7} {ft:<4} {cfg.descripcion}")
    print(f"\nTotal: {len(CONFIGURACIONES)} configuraciones")


def obtener_configuracion(id_config: str) -> ConfiguracionExperimental:
    """
    Busca una configuración por su ID.

    Args:
        id_config: Identificador (e.g. "B0", "D1P1Q1").

    Returns:
        ConfiguracionExperimental.

    Raises:
        ValueError: Si el ID no existe.
    """
    for cfg in CONFIGURACIONES:
        if cfg.id == id_config:
            return cfg
    raise ValueError(f"Configuración no encontrada: {id_config}")


if __name__ == "__main__":
    listar_configuraciones()
