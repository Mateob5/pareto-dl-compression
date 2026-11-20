"""
src/data/imdb_loader.py
=======================
Cargador del dataset IMDb para la prueba de concepto local.

Descarga un subset de IMDb (5,000 ejemplos en total) mediante la librería
`datasets` de HuggingFace, lo tokeniza con el tokenizer de BERT y construye
DataLoaders de PyTorch listos para entrenamiento e inferencia.

El tamaño reducido permite que todas las 22 configuraciones experimentales
se ejecuten en una GPU RTX 4060 Laptop (8 GB VRAM) sin saturar la memoria.

Autor: Jerónimo Bazán (2243590)
"""

import os
import sys
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

# Añadir la raíz del proyecto al path para importar config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from transformers import AutoTokenizer

# Cache local para no descargar el dataset cada vez
CACHE_DATASET = os.path.join(config.RAIZ_PROYECTO, "cache_dataset")
os.makedirs(CACHE_DATASET, exist_ok=True)


def cargar_subset_imdb(
    num_train: int = config.NUM_EJEMPLOS_TRAIN,
    num_test: int = config.NUM_EJEMPLOS_TEST,
) -> Tuple[dict, dict]:
    """
    Descarga y prepara un subset del dataset IMDb.

    Args:
        num_train: Número de ejemplos de entrenamiento (default 2,500).
        num_test: Número de ejemplos de prueba (default 2,500).

    Returns:
        Tuple de (train_dict, test_dict) con keys 'textos' y 'etiquetas'.
    """
    from datasets import load_dataset

    print(f"[datos] Descargando IMDb (cache: {CACHE_DATASET})...")
    dataset = load_dataset(
        config.DATASET_NOMBRE,
        split=["train", "test"],
        cache_dir=CACHE_DATASET,
    )

    # Tomar subset estratificado (mitad positiva, mitad negativa)
    train_pos = [x for x in dataset[0] if x["label"] == 1][: num_train // 2]
    train_neg = [x for x in dataset[0] if x["label"] == 0][: num_train // 2]
    test_pos = [x for x in dataset[1] if x["label"] == 1][: num_test // 2]
    test_neg = [x for x in dataset[1] if x["label"] == 0][: num_test // 2]

    train_data = {"textos": [x["text"] for x in train_pos + train_neg],
                  "etiquetas": [x["label"] for x in train_pos + train_neg]}
    test_data = {"textos": [x["text"] for x in test_pos + test_neg],
                 "etiquetas": [x["label"] for x in test_pos + test_neg]}

    print(f"[datos] Train: {len(train_data['textos'])} ejemplos "
          f"(pos={sum(train_data['etiquetas'])}, "
          f"neg={len(train_data['etiquetas']) - sum(train_data['etiquetas'])})")
    print(f"[datos] Test:  {len(test_data['textos'])} ejemplos "
          f"(pos={sum(test_data['etiquetas'])}, "
          f"neg={len(test_data['etiquetas']) - sum(test_data['etiquetas'])})")

    return train_data, test_data


def tokenizar_y_crear_dataloader(
    textos: list,
    etiquetas: list,
    tokenizer,
    batch_size: int,
    max_len: int = config.MAX_LEN,
    shuffle: bool = False,
) -> DataLoader:
    """
    Tokeniza una lista de textos y crea un DataLoader de PyTorch.

    Args:
        textos: Lista de strings (reseñas IMDb).
        etiquetas: Lista de enteros (0 o 1).
        tokenizer: Tokenizer de HuggingFace.
        batch_size: Tamaño del lote.
        max_len: Longitud máxima de secuencia.
        shuffle: Si barajar los datos.

    Returns:
        DataLoader de PyTorch listo para iterar.
    """
    encodings = tokenizer(
        textos,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(etiquetas, dtype=torch.long),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 0 para evitar overhead en GPU
        pin_memory=torch.cuda.is_available(),
    )


def preparar_dataloaders(
    nombre_modelo: str = config.MODELO_PROFESOR,
    batch_size_train: int = config.BATCH_SIZE_ENTRENAMIENTO,
    batch_size_eval: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Pipeline completo: descarga IMDb, tokeniza y crea dataloaders.

    Returns:
        Tuple (train_loader, test_loader, test_loader_batch1).
        El último tiene batch_size=1 para medición de latencia.
    """
    train_data, test_data = cargar_subset_imdb()
    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)

    train_loader = tokenizar_y_crear_dataloader(
        train_data["textos"], train_data["etiquetas"],
        tokenizer, batch_size_train, shuffle=True,
    )
    test_loader = tokenizar_y_crear_dataloader(
        test_data["textos"], test_data["etiquetas"],
        tokenizer, batch_size_eval, shuffle=False,
    )
    test_loader_batch1 = tokenizar_y_crear_dataloader(
        test_data["textos"], test_data["etiquetas"],
        tokenizer, config.BATCH_SIZE_MEDICION, shuffle=False,
    )

    return train_loader, test_loader, test_loader_batch1


if __name__ == "__main__":
    # Test rápido del cargador
    train_loader, test_loader, test_b1 = preparar_dataloaders()
    print(f"\nDataloaders creados:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  Test batch=1:  {len(test_b1)}")
