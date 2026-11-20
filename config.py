"""
config.py
=========
Configuración global del pipeline de optimización de modelos de Deep Learning.

Centraliza todos los hiperparámetros y constantes del proyecto para garantizar
reproducibilidad y consistencia entre las 22 configuraciones experimentales.

Autor: Jerónimo Bazán (2243590)
Proyecto: Optimización Computacional de Modelos de Deep Learning
Universidad Industrial de Santander
"""

import os
import random
import numpy as np
import torch

# =============================================================================
# SEMILLAS PARA REPRODUCIBILIDAD
# =============================================================================
# Semillas indicadas en el documento del proyecto (sección 5.5)
SEMILLAS = [42, 123, 2026]
SEMILLA_DEFAULT = 42

def fijar_semilla(semilla: int = SEMILLA_DEFAULT) -> None:
    """
    Fija todas las semillas aleatorias para garantizar reproducibilidad.

    Args:
        semilla: Valor numérico de la semilla (42, 123 o 2026).
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
        # Determinismo en CUDA (puede reducir rendimiento)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============================================================================
# CONFIGURACIÓN DE HARDWARE
# =============================================================================
# RTX 4060 Laptop = 8 GB VRAM. Reservamos ~1 GB para overhead de CUDA
VRAM_TOTAL_GB = 8.0
VRAM_RESERVA_GB = 1.0
VRAM_DISPONIBLE_GB = VRAM_TOTAL_GB - VRAM_RESERVA_GB  # 7.0 GB útiles

# Dispositivo de cómputo
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tipo de dato para mixed precision (acelera entrenamiento y reduce VRAM)
DTYPE_MIXED_PRECISION = torch.bfloat16  # bfloat16 también válido en RTX 4060

# =============================================================================
# MODELOS
# =============================================================================
MODELO_PROFESOR = "bert-base-uncased"        # 110 M parámetros
MODELO_ESTUDIANTE = "distilbert-base-uncased"  # 66 M parámetros

# Longitud máxima de secuencia (IMDb tiene reseñas largas; 128 es buen compromiso)
MAX_LEN = 128

# =============================================================================
# DATASET IMDb
# =============================================================================
# OPTIMIZACIÓN v2: Subset reducido para acelerar ejecución total.
# 1,500 ejemplos estratificados son suficientes para estimación confiable
# de accuracy/F1 (error muestral ±1.5%) y reducen el tiempo ~40%.
DATASET_NOMBRE = "imdb"
NUM_EJEMPLOS_TRAIN = 1500   # Reducido de 2,500 → 1,500 (training)
NUM_EJEMPLOS_TEST = 1500    # Reducido de 2,500 → 1,500 (test)

# =============================================================================
# ENTRENAMIENTO Y DESTILACIÓN
# =============================================================================
# OPTIMIZACIÓN v2: Hiperparámetros ajustados para ejecución total ~1.5-2 h.
# - Batch size 24: optimiza uso de VRAM en RTX 4060 (3.5 GB con DistilBERT)
# - Destilación 1 epoch + cache del modelo destilado: D1 se entrena UNA sola vez
#   y se reutiliza en las 13 configs D1* (antes se reentrenaba cada vez)
BATCH_SIZE_ENTRENAMIENTO = 24   # Aumentado de 16 → 24 (mejor uso VRAM)
BATCH_SIZE_INFERENCIA = 1       # Para medición de latencia (sección 5.4)
EPOCHS_DESTILACION = 1          # Reducido de 2 → 1 (suficiente con cache)
EPOCHS_FINE_TUNING = 1          # Fine-tuning post-poda (sección 4.2)
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50               # Reducido de 100 (menos batches por epoch)

# Hiperparámetros de destilación (sección 4.4 del documento)
TEMPERATURA_DESTILACION = 4.0   # T: suaviza las probabilidades del profesor
ALPHA_DESTILACION = 0.5         # Peso entre CE (etiquetas) y KL (profesor)

# =============================================================================
# MEDICIÓN DE HARDWARE (sección 5.5)
# =============================================================================
WARMUP_ITERACIONES = 50         # Descartadas antes de medir latencia
REPETICIONES_LATENCIA = 200     # Muestras para promediar latencia
BATCH_SIZE_MEDICION = 1         # Inferencia batch=1 (sección 5.4)

# =============================================================================
# PODA ESTRUCTURADA (sección 4.2)
# =============================================================================
PORCENTAJES_PODA = [0.20, 0.40, 0.60]  # P1, P2, P3

# =============================================================================
# RUTAS DEL PROYECTO
# =============================================================================
RAIZ_PROYECTO = os.path.dirname(os.path.abspath(__file__))
DIR_RESULTADOS = os.path.join(RAIZ_PROYECTO, "resultados")
DIR_DOCS = os.path.join(RAIZ_PROYECTO, "docs")

# OPTIMIZACIÓN v2: Cache de modelo destilado (D1).
# Tras entrenar D1 la primera vez, se guarda aquí y se reutiliza en D1Q1, D1Q2,
# D1P1, D1P2, D1P3, D1P*Q* y D1P3Q2FT sin reentrenar.
DIR_CACHE_MODELOS = os.path.join(RAIZ_PROYECTO, "cache_modelos")
RUTA_ESTUDIANTE_DESTILADO = os.path.join(DIR_CACHE_MODELOS, "estudiante_destilado")

# Flag para activar/desactivar el cache (False = reentrenar siempre, útil para debug)
USAR_CACHE_DESTILADO = True

# Archivos de salida
ARCHIVO_RESULTADOS_USUARIO = os.path.join(DIR_RESULTADOS, "resultados_usuario.csv")
ARCHIVO_RESULTADOS_REFERENCIA = os.path.join(DIR_RESULTADOS, "resultados_referencia.csv")
ARCHIVO_PARETO_PNG = os.path.join(DIR_RESULTADOS, "frontera_pareto.png")
ARCHIVO_PARETO_HTML = os.path.join(DIR_RESULTADOS, "frontera_pareto_interactiva.html")

# Asegurar que los directorios existen
os.makedirs(DIR_RESULTADOS, exist_ok=True)
os.makedirs(DIR_DOCS, exist_ok=True)
os.makedirs(DIR_CACHE_MODELOS, exist_ok=True)

# =============================================================================
# VERIFICACIÓN INICIAL
# =============================================================================
def info_entorno() -> dict:
    """
    Retorna un diccionario con información del entorno de cómputo actual.
    Útil para logging y depuración.
    """
    info = {
        "dispositivo": str(DISPOSITIVO),
        "cuda_disponible": torch.cuda.is_available(),
        "vram_total_gb": VRAM_TOTAL_GB,
        "vram_disponible_gb": VRAM_DISPONIBLE_GB,
        "semilla_default": SEMILLA_DEFAULT,
    }
    if torch.cuda.is_available():
        info["gpu_nombre"] = torch.cuda.get_device_name(0)
        info["gpu_vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["cuda_version"] = torch.version.cuda
    return info

if __name__ == "__main__":
    # Al ejecutar este archivo directamente, mostrar la configuración
    print("=" * 70)
    print("CONFIGURACIÓN DEL PIPELINE DE OPTIMIZACIÓN")
    print("=" * 70)
    for clave, valor in info_entorno().items():
        print(f"  {clave:<25}: {valor}")
    print("=" * 70)
