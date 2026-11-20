"""
src/tecnicas/cuantizacion.py
============================
Implementación de Post-Training Quantization (PTQ) con bitsandbytes.

Convierte modelos FP32 a INT8 (Q1) e INT4 (Q2) usando la librería
bitsandbytes integrada con transformers de HuggingFace.

Referencia bibliográfica del proyecto: Nahshan et al., "Loss aware
post-training quantization" (Machine Learning, 2021).

Autor: Jerónimo Bazán (2243590)
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def configuracion_cuantizacion(bits: int) -> BitsAndBytesConfig:
    """
    Construye la configuración de bitsandbytes para PTQ.

    Args:
        bits: 8 para INT8 (Q1), 4 para INT4 (Q2).

    Returns:
        BitsAndBytesConfig lista para pasar a from_pretrained.

    Raises:
        ValueError: Si bits no es 4 ni 8.
    """
    if bits == 8:
        # INT8: configuración estándar de bitsandbytes
        return BitsAndBytesConfig(load_in_8bit=True)

    elif bits == 4:
        # INT4: usa cuantización NF4 (NormalFloat 4-bit) con compute en bfloat16.
        # NF4 es óptimo para pesos con distribución normal (caso típico en LLMs y Transformers).
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # RTX 4060 soporta bfloat16
            bnb_4bit_use_double_quant=True,         # Doble cuantización: ahorra ~0.4 bits/param
        )

    else:
        raise ValueError(f"Solo se soportan 8 o 4 bits, recibido: {bits}")


def cuantizar_modelo(nombre_modelo: str, bits: int, num_labels: int = 2,
                     ruta_local: str = None) -> nn.Module:
    """
    Carga un modelo HuggingFace ya cuantizado con bitsandbytes.

    Args:
        nombre_modelo: Nombre del modelo en HuggingFace Hub
                      (e.g. "bert-base-uncased" o "distilbert-base-uncased").
        bits: 8 para INT8 (Q1), 4 para INT4 (Q2).
        num_labels: Número de clases (2 para clasificación binaria IMDb).
        ruta_local: (Opcional) Ruta local al modelo destilado cacheado.
                   Si se especifica, se carga desde ahí en lugar de HuggingFace Hub.
                   Esto permite cuantizar el estudiante destilado sin reentrenarlo.

    Returns:
        Modelo cuantizado cargado directamente en GPU.

    Notas:
        - bitsandbytes requiere GPU (no funciona en CPU).
        - El modelo se carga ya en CUDA; no es necesario llamar .to(device).
        - Para Inferencia se usa .eval() y torch.no_grad().
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "bitsandbytes requiere GPU CUDA. Ejecuta en un entorno con GPU."
        )

    bnb_config = configuracion_cuantizacion(bits)
    # Si hay ruta_local, cargar desde ahí (modelo destilado cacheado)
    origen = ruta_local if ruta_local else nombre_modelo
    etiqueta = f"{origen} (cache local)" if ruta_local else nombre_modelo
    print(f"[cuantizacion] Cargando {etiqueta} en INT{bits}...")

    modelo = AutoModelForSequenceClassification.from_pretrained(
        origen,
        quantization_config=bnb_config,
        num_labels=num_labels,
        device_map={"": 0},  # Distribuye automáticamente entre GPU/CPU si es necesario
    )
    modelo.eval()

    bits_str = f"INT{bits}"
    print(f"[cuantizacion] Modelo cargado: {modelo.config.model_type} ({bits_str})")
    return modelo


def estadisticas_memoria_modelo_cuantizado(modelo: nn.Module) -> dict:
    """
    Estima el tamaño en memoria del modelo cuantizado.

    Para INT8: ~1 byte por parámetro.
    Para INT4: ~0.5 bytes por parámetro (+ overhead de double quant).
    """
    total_params = sum(p.numel() for p in modelo.parameters())
    # Detectar tipo de cuantización
    primer_param = next(modelo.parameters())
    dtype = primer_param.dtype

    if dtype == torch.int8:
        bytes_por_param = 1.0
        bits = 8
    elif "int4" in str(dtype) or dtype == torch.uint8:
        # bitsandbytes empaqueta INT4 en uint8 (2 params por byte)
        bytes_por_param = 0.55  # 0.5 + overhead double quant
        bits = 4
    else:
        bytes_por_param = 4.0  # FP32 fallback
        bits = 32

    tamanio_mb = total_params * bytes_por_param / 1e6
    return {
        "parametros": total_params,
        "bits_por_param": bits,
        "tamanio_estimado_mb": tamanio_mb,
        "dtype": str(dtype),
    }


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Este módulo requiere GPU. Prueba desde un entorno con CUDA.")
    else:
        # Test rápido con DistilBERT INT8
        m = cuantizar_modelo(config.MODELO_ESTUDIANTE, bits=8)
        stats = estadisticas_memoria_modelo_cuantizado(m)
        print("\nEstadísticas INT8:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
