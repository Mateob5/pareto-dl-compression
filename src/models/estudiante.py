"""
src/models/estudiante.py
========================
Wrapper del modelo estudiante: DistilBERT-base-uncased (66 M parámetros).

DistilBERT es un modelo ya destilado por HuggingFace (referencia [8] del
documento), pero en este pipeline se utiliza como base sobre la cual
aplicar cuantización y poda adicional, y opcionalmente re-destilar
contra el profesor BERT-base.

Autor: Jerónimo Bazán (2243590)
"""

import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class EstudianteDistilBERT(nn.Module):
    """
    Modelo estudiante: DistilBERT con cabeza de clasificación binaria.

    Arquitectura ~40% más pequeña que BERT-base (66 M vs 110 M parámetros).
    Es el modelo base para todas las configuraciones D1-D14.
    """

    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.modelo = DistilBertForSequenceClassification.from_pretrained(
            config.MODELO_ESTUDIANTE,
            num_labels=num_labels,
        )
        self.num_labels = num_labels
        self.nombre = "Estudiante_DistilBERT"

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass estándar de DistilBERT para clasificación."""
        outputs = self.modelo(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def predecir(self, input_ids, attention_mask=None):
        """Retorna las logits (sin softmax) para inferencia."""
        with torch.no_grad():
            outputs = self.modelo(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.logits

    def parametros_totales(self) -> int:
        """Número total de parámetros del modelo."""
        return sum(p.numel() for p in self.parameters())

    def tamanio_mb_fp32(self) -> float:
        """Tamaño en MB asumiendo FP32 (4 bytes por parámetro)."""
        return self.parametros_totales() * 4 / 1e6

    def to(self, dispositivo):
        """Mueve el modelo al dispositivo indicado."""
        self.modelo = self.modelo.to(dispositivo)
        return self


def cargar_estudiante_destilado(ruta_checkpoint: str = None) -> EstudianteDistilBERT:
    """
    Carga el estudiante DestilBERT. Si existe un checkpoint de destilación
    previo (generado por src/tecnicas/destilacion.py), se carga desde allí.
    De lo contrario, se usa el modelo pre-entrenado base.

    Args:
        ruta_checkpoint: Ruta al modelo destilado local. Si None, usa base.

    Returns:
        Instancia de EstudianteDistilBERT.
    """
    modelo = EstudianteDistilBERT()
    if ruta_checkpoint and os.path.exists(ruta_checkpoint):
        print(f"[estudiante] Cargando checkpoint destilado: {ruta_checkpoint}")
        state_dict = torch.load(ruta_checkpoint, map_location="cpu")
        modelo.load_state_dict(state_dict)
    else:
        print("[estudiante] Usando DistilBERT pre-entrenado de HuggingFace")
    return modelo


if __name__ == "__main__":
    modelo = EstudianteDistilBERT()
    print(f"Modelo: {modelo.nombre}")
    print(f"Parámetros: {modelo.parametros_totales():,}")
    print(f"Tamaño FP32: {modelo.tamanio_mb_fp32():.1f} MB")
