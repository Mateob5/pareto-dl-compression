"""
src/analisis/pareto.py
=====================
Construcción de la Frontera de Pareto y visualización.

Implementa:
1. Identificación de puntos no dominados (Frontera de Pareto).
2. Visualización estática con Matplotlib (PNG para el documento).
3. Visualización interactiva con Plotly (HTML para exploración).

Sección 6 del documento del proyecto.

Autor: Jerónimo Bazán (2243590)
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Configurar Matplotlib para soportar caracteres en español
matplotlib.use("Agg")  # Sin display
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


def cargar_resultados(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga resultados desde un CSV. Si no se especifica ruta, intenta primero
    con resultados_usuario.csv (resultados reales del usuario) y cae a
    resultados_referencia.csv si no existe.

    Args:
        ruta_csv: Ruta al CSV explícita. Si None, usa orden de preferencia.

    Returns:
        DataFrame de pandas con los resultados.
    """
    if ruta_csv is None:
        if os.path.exists(config.ARCHIVO_RESULTADOS_USUARIO):
            ruta_csv = config.ARCHIVO_RESULTADOS_USUARIO
            print(f"[pareto] Usando resultados reales: {ruta_csv}")
        else:
            ruta_csv = config.ARCHIVO_RESULTADOS_REFERENCIA
            print(f"[pareto] Usando resultados de referencia: {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    print(f"[pareto] {len(df)} configuraciones cargadas")
    return df


def identificar_frontera_pareto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica los puntos no dominados (Frontera de Pareto).

    Definición: una configuración A domina a B si:
    - A tiene mayor o igual ahorro que B
    - A tiene mayor o igual precisión retenida que B
    - Y al menos una es estrictamente mayor

    Un punto es "no dominado" si ninguna otra configuración lo domina.

    Args:
        df: DataFrame con columnas 'ahorro_vram_pct' y 'precision_retenida_pct'.

    Returns:
        DataFrame solo con los puntos de la frontera, ordenados por ahorro.
    """
    puntos = df[["ahorro_vram_pct", "precision_retenida_pct"]].values

    no_dominados = []
    for i, p_i in enumerate(puntos):
        dominado = False
        for j, p_j in enumerate(puntos):
            if i == j:
                continue
            # p_j domina a p_i si es >= en ambos y > en al menos uno
            if (p_j[0] >= p_i[0] and p_j[1] >= p_i[1] and
                (p_j[0] > p_i[0] or p_j[1] > p_i[1])):
                dominado = True
                break
        if not dominado:
            no_dominados.append(i)

    frontera = df.iloc[no_dominados].copy()
    frontera = frontera.sort_values("ahorro_vram_pct").reset_index(drop=True)

    print(f"[pareto] {len(frontera)} configuraciones en la Frontera de Pareto:")
    for _, row in frontera.iterrows():
        print(f"  {row['id']:<10} ahorro={row['ahorro_vram_pct']:.1f}%  "
              f"precisión={row['precision_retenida_pct']:.2f}%")

    return frontera


def identificar_codo_pareto(frontera: pd.DataFrame) -> str:
    """
    Identifica el "codo" de la curva de Pareto.

    El codo es el punto donde la pendiente cambia más abruptamente,
    indicando el equilibrio óptimo entre ahorro y precisión.

    Método: distancia máxima a la línea que conecta el primer y último punto.

    Args:
        frontera: DataFrame con la frontera de Pareto.

    Returns:
        ID de la configuración en el codo.
    """
    if len(frontera) < 3:
        return frontera.iloc[0]["id"] if len(frontera) > 0 else None

    x = frontera["ahorro_vram_pct"].values
    y = frontera["precision_retenida_pct"].values

    # Línea entre primer y último punto
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    # Distancia perpendicular de cada punto a la línea
    distancias = []
    for xi, yi in zip(x, y):
        d = abs((y2 - y1) * xi - (x2 - x1) * yi + x2 * y1 - y2 * x1) / \
            np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distancias.append(d)

    idx_codo = np.argmax(distancias)
    id_codo = frontera.iloc[idx_codo]["id"]

    print(f"[pareto] Codo de la curva: {id_codo} "
          f"(ahorro={x[idx_codo]:.1f}%, precisión={y[idx_codo]:.2f}%)")

    return id_codo


def visualizar_pareto_matplotlib(
    df: pd.DataFrame,
    frontera: pd.DataFrame,
    id_codo: str = None,
    archivo_salida: str = config.ARCHIVO_PARETO_PNG,
) -> str:
    """
    Genera la visualización estática de la Frontera de Pareto con Matplotlib.

    Args:
        df: DataFrame completo.
        frontera: DataFrame solo con la frontera.
        id_codo: ID de la configuración en el codo.
        archivo_salida: Ruta del PNG.

    Returns:
        Ruta del archivo generado.
    """
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    # Colorear por tipo de modelo
    colores = {
        "BERT-base": "#0F62FE",      # Azul IBM
        "DistilBERT": "#EE5C42",     # Naranja
    }

    # Plotear todos los puntos
    for modelo in df["modelo_base"].unique():
        subset = df[df["modelo_base"] == modelo]
        ax.scatter(
            subset["ahorro_vram_pct"],
            subset["precision_retenida_pct"],
            c=colores.get(modelo, "gray"),
            label=f"{modelo} (todos)",
            s=120,
            alpha=0.5,
            edgecolors="white",
            linewidths=1.5,
        )

    # Plotear frontera de Pareto
    ax.plot(
        frontera["ahorro_vram_pct"],
        frontera["precision_retenida_pct"],
        "k--",
        linewidth=2,
        alpha=0.6,
        label="Frontera de Pareto",
    )
    ax.scatter(
        frontera["ahorro_vram_pct"],
        frontera["precision_retenida_pct"],
        c="black",
        s=180,
        marker="D",
        label="Puntos no dominados",
        zorder=5,
    )

    # Marcar el codo si se identificó
    if id_codo:
        codo = df[df["id"] == id_codo].iloc[0]
        ax.scatter(
            codo["ahorro_vram_pct"],
            codo["precision_retenida_pct"],
            c="#FFD700",
            s=300,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label=f"Codo óptimo: {id_codo}",
            zorder=10,
        )

    # Etiquetar cada configuración
    for _, row in df.iterrows():
        ax.annotate(
            row["id"],
            xy=(row["ahorro_vram_pct"], row["precision_retenida_pct"]),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
        )

    # Líneas de referencia: criterio de aceptación
    ax.axhline(y=98, color="green", linestyle=":", alpha=0.5, label="Criterio: 98% precisión")
    ax.axvline(x=50, color="green", linestyle=":", alpha=0.5, label="Criterio: 50% ahorro")

    ax.set_xlabel("Ahorro Computacional (% VRAM)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Precisión Retenida (%)", fontsize=13, fontweight="bold")
    ax.set_title("Frontera de Pareto: Pipeline de Cuantización + Poda + Destilación\n"
                 "Modelos BERT-base y DistilBERT sobre IMDb (RTX 4060 8GB)",
                 fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)

    # Rango de ejes
    ax.set_xlim(-5, 100)
    ax.set_ylim(85, 101)

    # Guardar
    fig.savefig(archivo_salida, dpi=150)
    plt.close(fig)

    print(f"[pareto] PNG estático guardado: {archivo_salida}")
    return archivo_salida


def visualizar_pareto_plotly(
    df: pd.DataFrame,
    frontera: pd.DataFrame,
    id_codo: str = None,
    archivo_salida: str = config.ARCHIVO_PARETO_HTML,
) -> str:
    """
    Genera la visualización interactiva de la Frontera de Pareto con Plotly.

    El HTML resultante permite hover sobre cada punto para ver detalles.

    Args:
        df: DataFrame completo.
        frontera: DataFrame solo con la frontera.
        id_codo: ID de la configuración en el codo.
        archivo_salida: Ruta del HTML.

    Returns:
        Ruta del archivo generado.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[pareto] Plotly no instalado. Ejecuta: pip install plotly")
        return None

    fig = go.Figure()

    # Todos los puntos, coloreados por modelo
    for modelo in df["modelo_base"].unique():
        subset = df[df["modelo_base"] == modelo]
        hover_text = [
            f"<b>{row['id']}</b><br>"
            f"Modelo: {row['modelo_base']}<br>"
            f"Cuantización: {row['cuantizacion']}<br>"
            f"Poda: {row['poda_pct']}<br>"
            f"Destilado: {row['destilado']}<br>"
            f"VRAM: {row['vram_pico_mb']:.1f} MB<br>"
            f"Latencia: {row['latencia_mediana_ms']:.2f} ms<br>"
            f"Accuracy: {row['accuracy']*100:.2f}%<br>"
            f"F1: {row['f1']:.4f}<br>"
            f"Ahorro VRAM: {row['ahorro_vram_pct']:.1f}%<br>"
            f"Precisión retenida: {row['precision_retenida_pct']:.2f}%<br>"
            f"Criterio aceptación: {row['criterio_aceptacion']}"
            for _, row in subset.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=subset["ahorro_vram_pct"],
            y=subset["precision_retenida_pct"],
            mode="markers+text",
            text=subset["id"],
            textposition="top center",
            name=f"{modelo} (todos)",
            marker=dict(size=14, opacity=0.6,
                       line=dict(width=1, color="white")),
            hovertext=hover_text,
            hoverinfo="text",
        ))

    # Frontera de Pareto
    fig.add_trace(go.Scatter(
        x=frontera["ahorro_vram_pct"],
        y=frontera["precision_retenida_pct"],
        mode="lines+markers",
        name="Frontera de Pareto",
        line=dict(dash="dash", color="black", width=2),
        marker=dict(size=18, symbol="diamond", color="black"),
    ))

    # Codo
    if id_codo:
        codo = df[df["id"] == id_codo].iloc[0]
        fig.add_trace(go.Scatter(
            x=[codo["ahorro_vram_pct"]],
            y=[codo["precision_retenida_pct"]],
            mode="markers",
            name=f"Codo óptimo: {id_codo}",
            marker=dict(size=28, symbol="star", color="gold",
                       line=dict(width=2, color="black")),
        ))

    # Líneas de criterio
    fig.add_hline(y=98, line_dash="dot", line_color="green",
                  annotation_text="Criterio: 98% precisión")
    fig.add_vline(x=50, line_dash="dot", line_color="green",
                  annotation_text="Criterio: 50% ahorro")

    fig.update_layout(
        title=dict(
            text="Frontera de Pareto: Pipeline de Optimización de Deep Learning<br>"
                 "<sup>BERT-base y DistilBERT sobre IMDb - RTX 4060 8GB</sup>",
            x=0.5,
        ),
        xaxis_title="Ahorro Computacional (% VRAM)",
        yaxis_title="Precisión Retenida (%)",
        template="plotly_white",
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01),
        width=1000,
        height=700,
        xaxis=dict(range=[-5, 100]),
        yaxis=dict(range=[85, 101]),
    )

    fig.write_html(archivo_salida)
    print(f"[pareto] HTML interactivo guardado: {archivo_salida}")
    return archivo_salida


def ejecutar_analisis_completo(ruta_csv: str = None) -> dict:
    """
    Ejecuta el análisis completo de Pareto sobre los resultados.

    Args:
        ruta_csv: Ruta al CSV de resultados. Si None, usa default.

    Returns:
        dict con rutas de archivos generados y datos de la frontera.
    """
    print("\n" + "=" * 70)
    print("ANÁLISIS DE LA FRONTERA DE PARETO")
    print("=" * 70)

    df = cargar_resultados(ruta_csv)
    frontera = identificar_frontera_pareto(df)
    id_codo = identificar_codo_pareto(frontera)

    png_path = visualizar_pareto_matplotlib(df, frontera, id_codo)
    html_path = visualizar_pareto_plotly(df, frontera, id_codo)

    return {
        "frontera": frontera,
        "id_codo": id_codo,
        "png": png_path,
        "html": html_path,
    }


if __name__ == "__main__":
    resultados = ejecutar_analisis_completo()
    print(f"\nAnálisis completo:")
    print(f"  Codo: {resultados['id_codo']}")
    print(f"  PNG: {resultados['png']}")
    print(f"  HTML: {resultados['html']}")
