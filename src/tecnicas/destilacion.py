"""
src/tecnicas/destilacion.py
===========================
Implementación de Knowledge Distillation (Destilación de Conocimiento).

Un modelo grande (Profesor = BERT-base) entrena a un modelo pequeño
(Estudiante = DistilBERT) para que imite sus probabilidades suaves,
logrando alta precisión con menos recursos.

Función de pérdida combinada (sección 4.4 del documento):

    L = alpha * CrossEntropy(y_true, p_student)
      + (1 - alpha) * T^2 * KL(p_teacher_suave || p_student_suave)

Donde:
    T = temperatura (suaviza las probabilidades; default T=4)
    alpha = peso entre pérdida dura (etiquetas) y suave (profesor)
    p_suave = softmax(logits / T)

Referencia bibliográfica: Sanh et al., "DistilBERT..." (arXiv:1910.01108, 2019)

Autor: Jerónimo Bazán (2243590)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class FuncionPerdidaDestilacion(nn.Module):
    """
    Implementa la función de pérdida combinada para destilación de conocimiento.

    Sección 4.4 del documento del proyecto.

    L = alpha * CE(y_true, p_student)
      + (1 - alpha) * T^2 * KL(softmax(p_teacher/T) || softmax(p_student/T))

    El factor T^2 escala la divergencia KL porque al dividir por T, los
    gradientes se atenúan por 1/T^2; multiplicarlos por T^2 los restaura.
    """

    def __init__(
        self,
        temperatura: float = config.TEMPERATURA_DESTILACION,
        alpha: float = config.ALPHA_DESTILACION,
    ):
        super().__init__()
        self.temperatura = temperatura
        self.alpha = alpha
        self.funcion_ce = nn.CrossEntropyLoss()
        # KL div con reduction="batchmean" para que sea comparable con CE
        self.funcion_kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        logits_estudiante: torch.Tensor,
        logits_profesor: torch.Tensor,
        etiquetas: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcula la pérdida total de destilación.

        Args:
            logits_estudiante: Logits del modelo estudiante (B, num_clases).
            logits_profesor: Logits del modelo profesor (B, num_clases).
            etiquetas: Etiquetas reales (B,).

        Returns:
            Escalar de pérdida combinada.
        """
        # 1. Pérdida dura: Cross-Entropy con etiquetas reales
        perdida_ce = self.funcion_ce(logits_estudiante, etiquetas)

        # 2. Pérdida suave: KL divergence con temperaturas
        # log_softmax del estudiante (estabilización numérica)
        log_p_estudiante = F.log_softmax(logits_estudiante / self.temperatura, dim=1)
        # softmax del profesor (sin log, es el target)
        p_profesor = F.softmax(logits_profesor / self.temperatura, dim=1)
        perdida_kl = self.funcion_kl(log_p_estudiante, p_profesor)

        # 3. Combinación ponderada (T^2 escala la KL)
        perdida_total = (
            self.alpha * perdida_ce
            + (1 - self.alpha) * (self.temperatura ** 2) * perdida_kl
        )

        return perdida_total, perdida_ce.item(), perdida_kl.item()


def destilar_estudiante(
    profesor_modelo: nn.Module,
    estudiante_modelo: nn.Module,
    train_loader: DataLoader,
    epochs: int = config.EPOCHS_DESTILACION,
    learning_rate: float = config.LEARNING_RATE,
    temperatura: float = config.TEMPERATURA_DESTILACION,
    alpha: float = config.ALPHA_DESTILACION,
    ruta_checkpoint: str = None,
) -> nn.Module:
    """
    Entrena el estudiante DistilBERT usando el profesor BERT-base como guía.

    El profesor se mantiene en modo eval() (sin gradientes). Solo el
    estudiante recibe gradientes.

    Args:
        profesor_modelo: Modelo profesor (BERT-base) cargado en GPU.
        estudiante_modelo: Modelo estudiante (DistilBERT) a entrenar.
        train_loader: DataLoader de entrenamiento tokenizado.
        epochs: Número de epochs de destilación.
        learning_rate: Learning rate del optimizador.
        temperatura: T para suavizar probabilidades.
        alpha: Peso entre CE y KL.
        ruta_checkpoint: Ruta donde guardar el modelo destilado.

    Returns:
        Modelo estudiante destilado.
    """
    dispositivo = config.DISPOSITIVO

    # Profesor en eval mode (sin gradientes)
    profesor_modelo = profesor_modelo.to(dispositivo)
    profesor_modelo.eval()
    for p in profesor_modelo.parameters():
        p.requires_grad = False

    # Estudiante en train mode
    estudiante_modelo = estudiante_modelo.to(dispositivo)
    estudiante_modelo.train()

    # Función de pérdida
    criterio = FuncionPerdidaDestilacion(temperatura=temperatura, alpha=alpha)

    # Optimizador y scheduler
    optimizador = torch.optim.AdamW(
        estudiante_modelo.parameters(),
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
    )
    total_pasos = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizador,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_pasos,
    )

    # Mixed precision para ahorrar VRAM y tiempo
    usa_scaler = dispositivo.type == "cuda" and config.DTYPE_MIXED_PRECISION == torch.float16
    escalador = torch.cuda.amp.GradScaler() if usa_scaler else None

    print(f"[destilacion] Iniciando destilación: T={temperatura}, alpha={alpha}, "
          f"epochs={epochs}, lr={learning_rate}")

    for epoch in range(epochs):
        perdida_total_epoch = 0.0
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids = input_ids.to(dispositivo)
            attention_mask = attention_mask.to(dispositivo)
            labels = labels.to(dispositivo)

            # Logits del profesor (sin gradientes)
            with torch.no_grad():
                if escalador:
                    with torch.cuda.amp.autocast(dtype=config.DTYPE_MIXED_PRECISION):
                        outputs_prof = profesor_modelo(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                else:
                    outputs_prof = profesor_modelo(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                logits_profesor = outputs_prof.logits.detach()

            # Logits del estudiante (con gradientes)
            optimizador.zero_grad()
            if escalador:
                with torch.cuda.amp.autocast(dtype=config.DTYPE_MIXED_PRECISION):
                    outputs_est = estudiante_modelo(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    logits_estudiante = outputs_est.logits
                    perdida, perdida_ce, perdida_kl = criterio(
                        logits_estudiante, logits_profesor, labels
                    )
                escalador.scale(perdida).backward()
                escalador.step(optimizador)
            else:
                outputs_est = estudiante_modelo(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                logits_estudiante = outputs_est.logits
                perdida, perdida_ce, perdida_kl = criterio(
                    logits_estudiante, logits_profesor, labels
                )
                perdida.backward()
                optimizador.step()

            scheduler.step()
            perdida_total_epoch += perdida.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(train_loader)} "
                      f"- Total: {perdida.item():.4f} | CE: {perdida_ce:.4f} | "
                      f"KL: {perdida_kl:.4f}")

        promedio = perdida_total_epoch / len(train_loader)
        print(f"  Epoch {epoch+1} completado. Loss promedio: {promedio:.4f}")

    # Guardar checkpoint si se especificó ruta
    if ruta_checkpoint:
        os.makedirs(os.path.dirname(ruta_checkpoint), exist_ok=True)
        torch.save(estudiante_modelo.state_dict(), ruta_checkpoint)
        print(f"[destilacion] Checkpoint guardado: {ruta_checkpoint}")

    estudiante_modelo.eval()
    return estudiante_modelo


if __name__ == "__main__":
    # Test de la función de pérdida con tensores sintéticos
    criterio = FuncionPerdidaDestilacion(temperatura=4.0, alpha=0.5)
    logits_est = torch.randn(8, 2)
    logits_prof = torch.randn(8, 2)
    etiquetas = torch.randint(0, 2, (8,))
    perdida, ce, kl = criterio(logits_est, logits_prof, etiquetas)
    print(f"Test función de pérdida: total={perdida.item():.4f}, CE={ce:.4f}, KL={kl:.4f}")
