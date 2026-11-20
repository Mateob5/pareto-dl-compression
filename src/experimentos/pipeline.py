"""
src/experimentos/pipeline.py
============================
Orquestador principal del pipeline de optimización.

Ejecuta las 22 configuraciones experimentales, mide métricas de hardware
y precisión, y guarda los resultados en un archivo CSV.

OPTIMIZACIÓN v2: Cache del modelo destilado (D1).
D1 se entrena UNA sola vez y se guarda a disco. Las 13 configuraciones
D1* (D1Q1, D1Q2, D1P1, ..., D1P3Q2FT) reutilizan el cache, reduciendo
el tiempo total de ~4-6 h a ~1.5-2 h.

Autor: Jerónimo Bazán (2243590)
"""

import os
import csv
import time
import gc
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from config import fijar_semilla
from src.data.imdb_loader import preparar_dataloaders
from src.models.profesor import ProfesorBERT
from src.models.estudiante import EstudianteDistilBERT
from src.tecnicas.cuantizacion import cuantizar_modelo
from src.tecnicas.poda import aplicar_poda_estructurada, fine_tuning_post_poda
from src.tecnicas.destilacion import destilar_estudiante
from src.metricas.hardware import evaluar_hardware_configuracion, reset_memoria_gpu, info_gpu
from src.metricas.precision import (
    evaluar_precision,
    calcular_precision_retenida,
    calcular_ahorro_computacional,
    cumple_criterio_aceptacion,
)
from src.experimentos.configuraciones import CONFIGURACIONES, ConfiguracionExperimental


def liberar_memoria():
    """Libera memoria GPU y Python antes de la siguiente configuración."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# OPTIMIZACIÓN v2: CACHE DEL MODELO DESTILADO (D1)
# =============================================================================
def _cache_destilado_existe() -> bool:
    """Verifica si existe el cache del modelo destilado en disco."""
    ruta = config.RUTA_ESTUDIANTE_DESTILADO
    # HuggingFace save_pretrained crea un directorio con config.json + pesos
    return os.path.isdir(ruta) and os.path.exists(os.path.join(ruta, "config.json"))


def _guardar_cache_destilado(modelo_estudiante) -> None:
    """Guarda el modelo estudiante destilado al disco para reutilizarlo."""
    ruta = config.RUTA_ESTUDIANTE_DESTILADO
    os.makedirs(ruta, exist_ok=True)
    print(f"[cache] Guardando estudiante destilado en: {ruta}")
    modelo_estudiante.modelo.save_pretrained(ruta)
    # También guardar el tokenizer (mismo que el estudiante base)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODELO_ESTUDIANTE)
    tokenizer.save_pretrained(ruta)
    print(f"[cache] Cache guardado ({os.path.getsize(os.path.join(ruta, 'config.json'))//1024} KB config)")


def _cargar_cache_destilado() -> "EstudianteDistilBERT":
    """Carga el modelo estudiante destilado desde el cache en disco."""
    ruta = config.RUTA_ESTUDIANTE_DESTILADO
    print(f"[cache] Cargando estudiante destilado desde cache: {ruta}")
    from transformers import AutoModelForSequenceClassification
    modelo_hf = AutoModelForSequenceClassification.from_pretrained(ruta, num_labels=2)
    estudiante = EstudianteDistilBERT()
    estudiante.modelo = modelo_hf
    return estudiante


def _obtener_o_crear_estudiante_destilado(train_loader, profesor=None):
    """
    OPTIMIZACIÓN v2: Retorna el estudiante destilado, usando cache si está disponible.

    Si el cache existe y USAR_CACHE_DESTILADO=True: carga desde disco (rápido, ~5 seg).
    Si no existe: entrena D1 desde cero y guarda al cache para futuros usos.

    Esto reduce el tiempo total del pipeline de ~4-6 h a ~1.5-2 h, ya que D1
    se entrena una sola vez en lugar de 13 (una por cada config D1*).
    """
    if config.USAR_CACHE_DESTILADO and _cache_destilado_existe():
        print("[pipeline] CACHE HIT: reutilizando estudiante destilado")
        return _cargar_cache_destilado()

    # Cache miss: entrenar D1 desde cero
    print("[pipeline] CACHE MISS: entrenando estudiante destilado por primera vez")
    if profesor is None:
        profesor = ProfesorBERT()

    modelo_estudiante = EstudianteDistilBERT()
    modelo_estudiante = destilar_estudiante(
        profesor_modelo=profesor,
        estudiante_modelo=modelo_estudiante,
        train_loader=train_loader,
        epochs=config.EPOCHS_DESTILACION,
    )

    # Guardar al cache para futuras configs D1*
    if config.USAR_CACHE_DESTILADO:
        _guardar_cache_destilado(modelo_estudiante)

    # Liberar profesor (ya no se necesita)
    del profesor
    liberar_memoria()

    return modelo_estudiante


def limpiar_cache_destilado() -> None:
    """Elimina el cache del modelo destilado (útil para forzar reentrenamiento)."""
    import shutil
    ruta = config.RUTA_ESTUDIANTE_DESTILADO
    if os.path.isdir(ruta):
        shutil.rmtree(ruta)
        print(f"[cache] Cache eliminado: {ruta}")
    else:
        print(f"[cache] No existe cache en: {ruta}")


def construir_modelo_para_configuracion(
    cfg: ConfiguracionExperimental,
    train_loader,
    profesor=None,
):
    """
    Construye el modelo correspondiente a una configuración experimental.

    Aplica las técnicas en el orden correcto:
    1. Cargar modelo base (BERT o DistilBERT)
    2. Si destilado: entrenar/cargar estudiante destilado
    3. Si poda: aplicar poda estructurada + fine-tuning
    4. Si cuantización INT8/INT4: cuantizar (debe ser lo último)

    Args:
        cfg: Configuración experimental.
        train_loader: DataLoader de entrenamiento (para destilación/FT).
        profesor: Modelo profesor cargado (para destilación).

    Returns:
        Modelo listo para evaluación.
    """
    print(f"\n{'='*70}")
    print(f"Construyendo configuración: {cfg.id} - {cfg.nombre}")
    print(f"{'='*70}")

    # Si requiere cuantización, cargamos directamente con bitsandbytes
    if cfg.cuantizacion in ("INT8", "INT4"):
        bits = 8 if cfg.cuantizacion == "INT8" else 4

        # OPTIMIZACIÓN v2: Si es config destilada (D1*Q*), usar el cache del
        # estudiante destilado en lugar de descargar DistilBERT base de HuggingFace.
        # Esto reutiliza el conocimiento destilado en lugar de starting fresh.
        ruta_local = None
        if cfg.destilado and config.USAR_CACHE_DESTILADO:
            # Asegurar que el cache exista (lo crea si es D1 la primera vez)
            _obtener_o_crear_estudiante_destilado(train_loader, profesor=profesor)
            ruta_local = config.RUTA_ESTUDIANTE_DESTILADO
            print(f"[pipeline] Usando estudiante destilado cacheado para cuantización INT{bits}")

        modelo = cuantizar_modelo(cfg.modelo_base, bits=bits, num_labels=2,
                                  ruta_local=ruta_local)

        # Si además requiere poda, aplicamos sobre las capas no cuantizadas
        # (en la práctica, bitsandbytes no soporta poda estructurada post-cuantización;
        # el pipeline óptimo sería podar primero y luego cuantizar, pero para esta
        # prueba de concepto mostramos el efecto combinado)
        if cfg.poda_pct:
            print(f"[pipeline] Aplicando poda {cfg.poda_pct*100:.0f}% post-cuantización "
                  f"(efecto limitado por bitsandbytes)")
            modelo = aplicar_poda_estructurada(modelo, cfg.poda_pct)

        if cfg.fine_tuning_extra:
            print("[pipeline] Aplicando fine-tuning adicional...")
            modelo = fine_tuning_post_poda(modelo, train_loader, epochs=1)

        return modelo

    # Sin cuantización: cargar modelo en FP32
    if cfg.destilado:
        # OPTIMIZACIÓN v2: Usar cache del estudiante destilado (entrena D1 UNA sola vez)
        modelo_estudiante = _obtener_o_crear_estudiante_destilado(
            train_loader, profesor=profesor
        )
    else:
        # BERT-base profesor
        modelo = ProfesorBERT()
        if cfg.poda_pct:
            modelo.modelo = aplicar_poda_estructurada(modelo.modelo, cfg.poda_pct)
            modelo.modelo = fine_tuning_post_poda(modelo.modelo, train_loader)
        else:
            # ✅ B0: fine-tuning del baseline (1 época) para que no sea random
            print("[pipeline] Fine-tuning del baseline B0 (BERT-base, 1 época)...")
            modelo.modelo = fine_tuning_post_poda(modelo.modelo, train_loader)
        return modelo

    # Si llegamos aquí, es estudiante destilado (posiblemente cacheado)
    if cfg.poda_pct:
        modelo_estudiante.modelo = aplicar_poda_estructurada(modelo_estudiante.modelo, cfg.poda_pct)
        modelo_estudiante.modelo = fine_tuning_post_poda(modelo_estudiante.modelo, train_loader)

    return modelo_estudiante


def ejecutar_configuracion(
    cfg: ConfiguracionExperimental,
    train_loader,
    test_loader,
    test_loader_batch1,
    resultados_base: dict = None,
) -> dict:
    """
    Ejecuta una configuración experimental completa.

    Args:
        cfg: Configuración experimental.
        train_loader: DataLoader de entrenamiento.
        test_loader: DataLoader de prueba (batch=32).
        test_loader_batch1: DataLoader de prueba (batch=1, para latencia).
        resultados_base: Resultados de B0 (para calcular retención y ahorro).

    Returns:
        dict con todas las métricas medidas.
    """
    fijar_semilla(config.SEMILLA_DEFAULT)
    liberar_memoria()

    t_inicio = time.time()

    # Construir modelo
    # OPTIMIZACIÓN v2: Solo cargamos el profesor si NO hay cache del destilado.
    # Si el cache existe, el profesor no se necesita (ya se usó para crear el cache).
    profesor = None
    if cfg.destilado and not (
        config.USAR_CACHE_DESTILADO and _cache_destilado_existe()
    ):
        profesor = ProfesorBERT()

    modelo = construir_modelo_para_configuracion(cfg, train_loader, profesor=profesor)

    # Medir precisión
    resultados_precision = evaluar_precision(modelo, test_loader, descripcion=cfg.id)

    # Medir hardware
    resultados_hardware = evaluar_hardware_configuracion(
        modelo, test_loader_batch1, descripcion=cfg.id
    )

    t_total = time.time() - t_inicio

    # Calcular métricas derivadas respecto a B0
    if resultados_base:
        precision_retenida = calcular_precision_retenida(
            resultados_precision["accuracy"], resultados_base["accuracy"]
        )
        ahorro_vram = calcular_ahorro_computacional(
            resultados_hardware["vram_pico_mb"], resultados_base["vram_pico_mb"]
        )
        ahorro_latencia = calcular_ahorro_computacional(
            resultados_hardware["latencia_mediana_ms"], resultados_base["latencia_mediana_ms"]
        )
        cumple = cumple_criterio_aceptacion(ahorro_vram, precision_retenida)
    else:
        precision_retenida = 100.0
        ahorro_vram = 0.0
        ahorro_latencia = 0.0
        cumple = False

    # Liberar modelo
    del modelo
    if profesor is not None:
        del profesor
    liberar_memoria()

    return {
        "id": cfg.id,
        "nombre": cfg.nombre,
        "modelo_base": "DistilBERT" if "distilbert" in cfg.modelo_base else "BERT-base",
        "cuantizacion": cfg.cuantizacion or "FP32",
        "poda_pct": f"{cfg.poda_pct*100:.0f}%" if cfg.poda_pct else "0%",
        "destilado": "Sí" if cfg.destilado else "No",
        "vram_pico_mb": resultados_hardware["vram_pico_mb"],
        "latencia_mediana_ms": resultados_hardware["latencia_mediana_ms"],
        "latencia_p95_ms": resultados_hardware["latencia_p95_ms"],
        "throughput_eps": resultados_hardware["throughput_eps"],
        "accuracy": resultados_precision["accuracy"],
        "f1": resultados_precision["f1"],
        "precision_retenida_pct": precision_retenida,
        "ahorro_vram_pct": ahorro_vram,
        "ahorro_latencia_pct": ahorro_latencia,
        "criterio_aceptacion": "Sí" if cumple else "No",
        "tiempo_ejecucion_s": t_total,
    }


def ejecutar_todas_configuraciones(
    ids_configuraciones: list = None,
    archivo_salida: str = config.ARCHIVO_RESULTADOS_USUARIO,
) -> list:
    """
    Ejecuta el pipeline completo de 22 configuraciones.

    Args:
        ids_configuraciones: Lista de IDs a ejecutar. Si None, todas.
        archivo_salida: Ruta del CSV donde guardar resultados.

    Returns:
        Lista de diccionarios con los resultados.
    """
    print(f"\n{'#'*70}")
    print(f"# PIPELINE DE OPTIMIZACIÓN - {len(CONFIGURACIONES)} configuraciones")
    print(f"# GPU: {info_gpu()}")
    print(f"{'#'*70}")

    # Preparar datos
    train_loader, test_loader, test_loader_b1 = preparar_dataloaders()

    # Filtrar configuraciones a ejecutar
    if ids_configuraciones:
        configs_a_ejecutar = [c for c in CONFIGURACIONES if c.id in ids_configuraciones]
    else:
        configs_a_ejecutar = CONFIGURACIONES

    print(f"\nConfiguraciones a ejecutar: {len(configs_a_ejecutar)}")

    resultados = []
    resultados_base = None

    # Abrir CSV para escritura incremental
    campos_csv = [
        "id", "nombre", "modelo_base", "cuantizacion", "poda_pct", "destilado",
        "vram_pico_mb", "latencia_mediana_ms", "latencia_p95_ms", "throughput_eps",
        "accuracy", "f1", "precision_retenida_pct", "ahorro_vram_pct",
        "ahorro_latencia_pct", "criterio_aceptacion", "tiempo_ejecucion_s"
    ]

    with open(archivo_salida, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=campos_csv)
        writer.writeheader()

        for i, cfg in enumerate(configs_a_ejecutar, 1):
            print(f"\n\n>>> Progreso: {i}/{len(configs_a_ejecutar)} - Configuración {cfg.id}")

            try:
                resultado = ejecutar_configuracion(
                    cfg, train_loader, test_loader, test_loader_b1,
                    resultados_base=resultados_base,
                )

                # Guardar base después de ejecutar B0
                if cfg.id == "B0":
                    resultados_base = resultado

                resultados.append(resultado)
                writer.writerow(resultado)
                f_csv.flush()

                print(f">>> {cfg.id} completado en {resultado['tiempo_ejecucion_s']:.1f}s")

            except Exception as e:
                print(f">>> ERROR en {cfg.id}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'#'*70}")
    print(f"# PIPELINE COMPLETADO")
    print(f"# Resultados guardados en: {archivo_salida}")
    print(f"# Configuraciones exitosas: {len(resultados)}/{len(configs_a_ejecutar)}")
    print(f"{'#'*70}")

    return resultados
