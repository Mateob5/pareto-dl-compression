"""
Resultados de referencia basados en literatura para las 22 configuraciones.

Estos valores son ESTIMACIONES derivadas de los siguientes papers:
- DistilBERT (Sanh et al., 2019): retiene 95-97% de la precisión de BERT-base
- bitsandbytes INT8 (Dettmers et al., 2022): pérdida típica 0.5-1% accuracy
- bitsandbytes INT4 NF4 (Dettmers et al., 2023, QLoRA): pérdida típica 1-3%
- Pruning estructural de cabezas (Voita et al., 2019): 20%=0.5%, 40%=1.5%, 60%=3-4%

VRAM y latencia se basan en medidas típicas de BERT-base y DistilBERT
en GPU de 8 GB VRAM (RTX 3060/4060), batch_size=1, max_seq_len=128.

NOTA IMPORTANTE: Estos valores sirven para visualizar cómo se verá la
Frontera de Pareto ANTES de ejecutar el pipeline completo. Los resultados
reales dependerán del hardware específico, drivers, y semilla aleatoria.

Para obtener resultados reales, ejecuta:
    python main.py completo
"""
import csv
import os

# Valores de referencia basados en literatura
# B0 = línea base: BERT-base FP32 con accuracy=0.9350 (típico en IMDb subset 2500)
# VRAM medida típica: 1.30 GB (batch=1, seq=128)
# Latencia típica: 25.0 ms (RTX 4060 Laptop)

RESULTADOS_REFERENCIA = [
    # id, nombre, modelo_base, cuantizacion, poda_pct, destilado,
    # vram_pico_mb, latencia_mediana_ms, latencia_p95_ms, throughput_eps,
    # accuracy, f1, precision_retenida_pct, ahorro_vram_pct,
    # ahorro_latencia_pct, criterio_aceptacion, tiempo_ejecucion_s

    # --- LÍNEA BASE ---
    ["B0", "Línea Base BERT-base FP32", "BERT-base", "FP32", "0%", "No",
     1300.0, 25.0, 28.5, 40.0, 0.9350, 0.9342, 100.00, 0.00, 0.00, "No", 60.0],

    # --- CUANTIZACIÓN ---
    ["Q1", "BERT-base INT8 (PTQ)", "BERT-base", "INT8", "0%", "No",
     720.0, 16.5, 19.0, 60.6, 0.9290, 0.9280, 99.36, 44.62, 34.00, "No", 90.0],
    ["Q2", "BERT-base INT4 (PTQ NF4)", "BERT-base", "INT4", "0%", "No",
     530.0, 13.8, 16.2, 72.5, 0.9180, 0.9168, 98.18, 59.23, 44.80, "No", 95.0],

    # --- PODA ---
    ["P1", "BERT-base + Poda 20%", "BERT-base", "FP32", "20%", "No",
     1180.0, 22.5, 25.8, 44.4, 0.9320, 0.9310, 99.68, 9.23, 10.00, "No", 240.0],
    ["P2", "BERT-base + Poda 40%", "BERT-base", "FP32", "40%", "No",
     1020.0, 19.8, 22.5, 50.5, 0.9210, 0.9198, 98.50, 21.54, 20.80, "No", 280.0],
    ["P3", "BERT-base + Poda 60%", "BERT-base", "FP32", "60%", "No",
     880.0, 17.2, 19.8, 58.1, 0.9050, 0.9038, 96.79, 32.31, 31.20, "No", 320.0],

    # --- COMBINACIONES BERT (Poda + Cuantización) ---
    ["P1Q1", "BERT-base + Poda 20% + INT8", "BERT-base", "INT8", "20%", "No",
     680.0, 15.2, 17.5, 65.8, 0.9270, 0.9260, 99.14, 47.69, 39.20, "No", 320.0],
    ["P2Q1", "BERT-base + Poda 40% + INT8", "BERT-base", "INT8", "40%", "No",
     620.0, 13.8, 16.0, 72.5, 0.9160, 0.9150, 97.97, 52.31, 44.80, "No", 360.0],
    ["P3Q1", "BERT-base + Poda 60% + INT8", "BERT-base", "INT8", "60%", "No",
     560.0, 12.0, 14.2, 83.3, 0.8980, 0.8970, 96.04, 56.92, 52.00, "No", 400.0],

    # --- DESTILACIÓN (DistilBERT) ---
    ["D1", "DistilBERT destilado FP32", "DistilBERT", "FP32", "0%", "Sí",
     820.0, 14.5, 16.8, 69.0, 0.9210, 0.9198, 98.50, 36.92, 42.00, "No", 600.0],

    # --- DESTILACIÓN + CUANTIZACIÓN ---
    ["D1Q1", "DistilBERT + INT8", "DistilBERT", "INT8", "0%", "Sí",
     460.0, 9.0, 10.5, 111.1, 0.9150, 0.9140, 97.86, 64.62, 64.00, "Sí", 90.0],
    ["D1Q2", "DistilBERT + INT4", "DistilBERT", "INT4", "0%", "Sí",
     340.0, 7.5, 8.8, 133.3, 0.9050, 0.9040, 96.79, 73.85, 70.00, "Sí", 95.0],

    # --- DESTILACIÓN + PODA ---
    ["D1P1", "DistilBERT + Poda 20%", "DistilBERT", "FP32", "20%", "Sí",
     740.0, 12.8, 14.8, 78.1, 0.9180, 0.9168, 98.18, 43.08, 48.80, "No", 600.0],
    ["D1P2", "DistilBERT + Poda 40%", "DistilBERT", "FP32", "40%", "Sí",
     640.0, 11.0, 12.8, 90.9, 0.9080, 0.9070, 97.11, 50.77, 56.00, "No", 640.0],
    ["D1P3", "DistilBERT + Poda 60%", "DistilBERT", "FP32", "60%", "Sí",
     540.0, 9.5, 11.0, 105.3, 0.8920, 0.8910, 95.40, 58.46, 62.00, "No", 680.0],

    # --- DESTILACIÓN + PODA + INT8 ---
    ["D1P1Q1", "DistilBERT + Poda 20% + INT8", "DistilBERT", "INT8", "20%", "Sí",
     420.0, 8.2, 9.5, 122.0, 0.9110, 0.9100, 97.33, 67.69, 67.20, "Sí", 690.0],
    ["D1P2Q1", "DistilBERT + Poda 40% + INT8", "DistilBERT", "INT8", "40%", "Sí",
     380.0, 7.0, 8.2, 142.9, 0.8980, 0.8970, 95.94, 70.77, 72.00, "No", 730.0],
    ["D1P3Q1", "DistilBERT + Poda 60% + INT8", "DistilBERT", "INT8", "60%", "Sí",
     320.0, 6.0, 7.0, 166.7, 0.8800, 0.8790, 94.12, 75.38, 76.00, "No", 770.0],

    # --- DESTILACIÓN + PODA + INT4 ---
    ["D1P1Q2", "DistilBERT + Poda 20% + INT4", "DistilBERT", "INT4", "20%", "Sí",
     300.0, 6.5, 7.6, 153.8, 0.8980, 0.8970, 95.94, 76.92, 74.00, "No", 780.0],
    ["D1P2Q2", "DistilBERT + Poda 40% + INT4", "DistilBERT", "INT4", "40%", "Sí",
     260.0, 5.5, 6.5, 181.8, 0.8820, 0.8810, 94.33, 80.00, 78.00, "No", 820.0],
    ["D1P3Q2", "DistilBERT + Poda 60% + INT4", "DistilBERT", "INT4", "60%", "Sí",
     220.0, 4.8, 5.6, 208.3, 0.8620, 0.8610, 92.09, 83.08, 80.80, "No", 860.0],

    # --- CONFIGURACIÓN CON FINE-TUNING ADICIONAL ---
    ["D1P3Q2FT", "DistilBERT + Poda 60% + INT4 + FT", "DistilBERT", "INT4", "60%", "Sí",
     220.0, 4.8, 5.6, 208.3, 0.8780, 0.8770, 93.90, 83.08, 80.80, "No", 1200.0],
]

CAMPOS = [
    "id", "nombre", "modelo_base", "cuantizacion", "poda_pct", "destilado",
    "vram_pico_mb", "latencia_mediana_ms", "latencia_p95_ms", "throughput_eps",
    "accuracy", "f1", "precision_retenida_pct", "ahorro_vram_pct",
    "ahorro_latencia_pct", "criterio_aceptacion", "tiempo_ejecucion_s"
]


def generar_csv_referencia():
    """Genera el archivo CSV con los resultados de referencia."""
    # __file__ = .../po-concepto-local/resultados/generar_referencia.py
    # queremos = .../po-concepto-local/resultados/resultados_referencia.csv
    ruta = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "resultados_referencia.csv"
    )

    os.makedirs(os.path.dirname(ruta), exist_ok=True)

    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CAMPOS)
        writer.writerows(RESULTADOS_REFERENCIA)

    print(f"CSV de referencia generado: {ruta}")
    print(f"Total de configuraciones: {len(RESULTADOS_REFERENCIA)}")
    return ruta


if __name__ == "__main__":
    generar_csv_referencia()
