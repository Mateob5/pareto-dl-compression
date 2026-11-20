"""
src/metricas/precision.py
=========================
Medición de métricas de precisión: Accuracy, F1-Score, % Precisión Retenida.

Implementa la evaluación del modelo sobre el set de prueba de IMDb y el
cálculo de las métricas indicadas en la sección 5.4 del documento.

Autor: Jerónimo Bazán (2243590)
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def evaluar_precision(
    modelo: torch.nn.Module,
    test_loader,
    descripcion: str = "",
) -> dict:
    """
    Evalúa el modelo sobre el set de prueba y calcula métricas de clasificación.

    Args:
        modelo: Modelo a evaluar.
        test_loader: DataLoader con datos de prueba.
        descripcion: Nombre de la configuración (para logging).

    Returns:
        dict con: accuracy, f1, precision, recall, todas_predicciones, todas_etiquetas.
    """
    dispositivo = config.DISPOSITIVO
    try:
        modelo = modelo.to(dispositivo)
    except (ValueError, RuntimeError):
        # Modelo ya dispatchado por bitsandbytes (INT4/INT8)
        pass
    modelo.eval()

    todas_predicciones = []
    todas_etiquetas = []

    print(f"[precision] Evaluando: {descripcion}")

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(dispositivo)
            attention_mask = attention_mask.to(dispositivo)

            # Forward
            outputs = modelo(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Predicción = argmax
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            todas_predicciones.extend(preds)
            todas_etiquetas.extend(labels.numpy())

    todas_predicciones = np.array(todas_predicciones)
    todas_etiquetas = np.array(todas_etiquetas)

    # Calcular métricas
    accuracy = accuracy_score(todas_etiquetas, todas_predicciones)
    f1 = f1_score(todas_etiquetas, todas_predicciones, average="binary")
    precision = precision_score(todas_etiquetas, todas_predicciones, average="binary")
    recall = recall_score(todas_etiquetas, todas_predicciones, average="binary")

    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "num_ejemplos": len(todas_etiquetas),
    }


def calcular_precision_retenida(
    accuracy_config: float,
    accuracy_base: float,
) -> float:
    """
    Calcula el % de precisión retenida respecto a la línea base B0.

    % Retención = (accuracy_config / accuracy_base) * 100

    Sección 5.4 del documento.

    Args:
        accuracy_config: Accuracy de la configuración evaluada.
        accuracy_base: Accuracy de la línea base B0.

    Returns:
        float: % de precisión retenida.
    """
    if accuracy_base <= 0:
        return 0.0
    return (accuracy_config / accuracy_base) * 100.0


def calcular_ahorro_computacional(
    vram_config: float,
    vram_base: float,
) -> float:
    """
    Calcula el % de ahorro de VRAM respecto a la línea base B0.

    % Ahorro = (1 - vram_config / vram_base) * 100

    Sección 6.1 del documento.

    Args:
        vram_config: VRAM pico de la configuración evaluada (MB).
        vram_base: VRAM pico de la línea base B0 (MB).

    Returns:
        float: % de ahorro de VRAM.
    """
    if vram_base <= 0:
        return 0.0
    return (1.0 - vram_config / vram_base) * 100.0


def cumple_criterio_aceptacion(
    ahorro_vram_pct: float,
    precision_retenida_pct: float,
    ahorro_minimo: float = 50.0,
    precision_minima: float = 98.0,
) -> bool:
    """
    Verifica si una configuración cumple el criterio de aceptación (sección 5.5).

    "Se considerará un éxito rotundo si una configuración logra un ahorro de
    VRAM mayor al 50% manteniendo al menos el 98% de la precisión original."

    Args:
        ahorro_vram_pct: % de ahorro de VRAM.
        precision_retenida_pct: % de precisión retenida.
        ahorro_minimo: Umbral mínimo de ahorro (default 50%).
        precision_minima: Umbral mínimo de precisión retenida (default 98%).

    Returns:
        bool: True si cumple el criterio de éxito rotundo.
    """
    return ahorro_vram_pct >= ahorro_minimo and precision_retenida_pct >= precision_minima
