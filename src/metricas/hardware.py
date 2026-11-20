"""
src/metricas/hardware.py
========================
Medición de métricas de hardware: VRAM pico, latencia, throughput.

Implementa el protocolo de medición descrito en la sección 5.5 del
documento del proyecto:
- Warm-up de 50 iteraciones (descartadas)
- Semillas fijas (42, 123, 2026)
- Batch size fijo (=1) para inferencia
- Sincronización CUDA obligatoria para mediciones precisas

Autor: Jerónimo Bazán (2243590)
"""

import time
import torch
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def reset_memoria_gpu() -> None:
    """Resetea los contadores de memoria de CUDA."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def medir_vram_pico_mb() -> float:
    """
    Retorna la VRAM pico consumida desde el último reset, en MB.

    Returns:
        float: MB de VRAM pico.
    """
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e6


def medir_latencia_ms(
    modelo: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    repeticiones: int = config.REPETICIONES_LATENCIA,
    warmup: int = config.WARMUP_ITERACIONES,
) -> dict:
    """
    Mide la latencia de inferencia por petición (batch_size=1).

    Protocolo (sección 5.5):
    1. Warm-up de `warmup` iteraciones (descartadas).
    2. Sincronización CUDA.
    3. `repeticiones` inferencias con eventos CUDA para alta precisión.
    4. Reporta mediana + percentil 95 + desviación estándar.

    Args:
        modelo: Modelo cargado en GPU.
        input_ids: Tensor de input IDs (batch=1).
        attention_mask: Tensor de attention mask.
        repeticiones: Número de inferencias a medir.
        warmup: Iteraciones de calentamiento descartadas.

    Returns:
        dict con latencia_mediana_ms, latencia_p95_ms, latencia_std_ms.
    """
    dispositivo = config.DISPOSITIVO
    modelo = modelo.to(dispositivo)
    modelo.eval()

    input_ids = input_ids.to(dispositivo)
    attention_mask = attention_mask.to(dispositivo)

    # 1. Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = modelo(input_ids=input_ids, attention_mask=attention_mask)

    # 2. Sincronización antes de medir
    if dispositivo.type == "cuda":
        torch.cuda.synchronize()

    # 3. Mediciones con eventos CUDA (alta precisión)
    latencias = []
    with torch.no_grad():
        for _ in range(repeticiones):
            if dispositivo.type == "cuda":
                inicio = torch.cuda.Event(enable_timing=True)
                fin = torch.cuda.Event(enable_timing=True)
                inicio.record()
                _ = modelo(input_ids=input_ids, attention_mask=attention_mask)
                fin.record()
                torch.cuda.synchronize()
                latencia_ms = inicio.elapsed_time(fin)
            else:
                t0 = time.perf_counter()
                _ = modelo(input_ids=input_ids, attention_mask=attention_mask)
                t1 = time.perf_counter()
                latencia_ms = (t1 - t0) * 1000
            latencias.append(latencia_ms)

    latencias_np = np.array(latencias)
    return {
        "latencia_mediana_ms": float(np.median(latencias_np)),
        "latencia_p95_ms": float(np.percentile(latencias_np, 95)),
        "latencia_std_ms": float(np.std(latencias_np)),
        "latencias_crudas": latencias_np.tolist(),
    }


def medir_throughput_eps(latencia_mediana_ms: float) -> float:
    """
    Calcula throughput en ejemplos por segundo a partir de la latencia mediana.

    Throughput = 1000 / latencia_ms (cuando batch_size=1).

    Args:
        latencia_mediana_ms: Latencia mediana en ms.

    Returns:
        float: Ejemplos por segundo.
    """
    if latencia_mediana_ms <= 0:
        return 0.0
    return 1000.0 / latencia_mediana_ms


def evaluar_hardware_configuracion(
    modelo: torch.nn.Module,
    test_loader_batch1,
    descripcion: str = "",
) -> dict:
    """
    Ejecuta el protocolo completo de medición de hardware para una configuración.

    Mide:
    - VRAM pico durante inferencia (MB)
    - Latencia mediana, p95 y std (ms)
    - Throughput (ejemplos/seg)

    Args:
        modelo: Modelo a evaluar.
        test_loader_batch1: DataLoader con batch_size=1.
        descripcion: Nombre de la configuración (para logging).

    Returns:
        dict con todas las métricas de hardware.
    """
    dispositivo = config.DISPOSITIVO

    print(f"\n[hardware] Midiendo configuración: {descripcion}")

    # Tomar una muestra representativa del test_loader (batch=1)
    input_ids, attention_mask, _ = next(iter(test_loader_batch1))

    # Reset memoria antes de medir
    reset_memoria_gpu()

    # Cargar modelo en GPU si no está
    modelo = modelo.to(dispositivo)
    modelo.eval()

    # Medir latencia (esto también consume VRAM)
    resultados_latencia = medir_latencia_ms(modelo, input_ids, attention_mask)

    # Medir VRAM pico
    vram_pico_mb = medir_vram_pico_mb()

    # Throughput
    throughput = medir_throughput_eps(resultados_latencia["latencia_mediana_ms"])

    print(f"  VRAM pico: {vram_pico_mb:.1f} MB")
    print(f"  Latencia mediana: {resultados_latencia['latencia_mediana_ms']:.2f} ms")
    print(f"  Latencia p95: {resultados_latencia['latencia_p95_ms']:.2f} ms")
    print(f"  Throughput: {throughput:.1f} ej/s")

    return {
        "vram_pico_mb": vram_pico_mb,
        "latencia_mediana_ms": resultados_latencia["latencia_mediana_ms"],
        "latencia_p95_ms": resultados_latencia["latencia_p95_ms"],
        "latencia_std_ms": resultados_latencia["latencia_std_ms"],
        "throughput_eps": throughput,
    }


def info_gpu() -> dict:
    """
    Retorna información de la GPU disponible.

    Returns:
        dict con nombre, VRAM total, etc. None si no hay GPU.
    """
    if not torch.cuda.is_available():
        return {"disponible": False}

    props = torch.cuda.get_device_properties(0)
    return {
        "disponible": True,
        "nombre": torch.cuda.get_device_name(0),
        "vram_total_gb": props.total_memory / 1e9,
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda,
    }


if __name__ == "__main__":
    info = info_gpu()
    print("Información de GPU:")
    for k, v in info.items():
        print(f"  {k}: {v}")
