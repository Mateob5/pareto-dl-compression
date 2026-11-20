"""
src/models/profesor.py
======================
Wrapper del modelo profesor: BERT-base-uncased (110 M parámetros).

Es el punto de comparación (línea base B0) en FP32 para todas las
configuraciones experimentales del proyecto.

Autor: Jerónimo Bazán (2243590)
"""

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class ProfesorBERT(nn.Module):
    """
    Modelo profesor: BERT-base con cabeza de clasificación binaria.

    Es el modelo de referencia (B0) en el pipeline. Se carga en FP32 y
    NO se modifica durante las pruebas de cuantización/poda/destilación
    sobre el estudiante.
    """

    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.modelo = BertForSequenceClassification.from_pretrained(
            config.MODELO_PROFESOR,
            num_labels=num_labels,
        )
        self.num_labels = num_labels
        self.nombre = "Profesor_BERT_base"

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass estándar de BERT para clasificación."""
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


if __name__ == "__main__":
    modelo = ProfesorBERT()
    print(f"Modelo: {modelo.nombre}")
    print(f"Parámetros: {modelo.parametros_totales():,}")
    print(f"Tamaño FP32: {modelo.tamanio_mb_fp32():.1f} MB")
