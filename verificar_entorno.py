"""
verificar_entorno.py
====================
Script de verificación del entorno de cómputo local.

Comprueba:
1. Python y versiones de librerías clave
2. GPU NVIDIA disponible y compatible
3. VRAM suficiente (>= 4 GB requerido, 8 GB recomendado)
4. PyTorch compilado con CUDA
5. bitsandbytes instalado correctamente
6. Conexión a HuggingFace Hub (para descarga de modelos)

Autor: Jerónimo Bazán (2243590)
"""

import sys
import platform
import importlib
import subprocess
import os

# Colores ANSI para output legible
class Colors:
    VERDE = "\033[92m"
    ROJO = "\033[91m"
    AMARILLO = "\033[93m"
    AZUL = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {Colors.VERDE}[OK]{Colors.RESET} {msg}")


def error(msg: str) -> None:
    print(f"  {Colors.ROJO}[ERROR]{Colors.RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {Colors.AMARILLO}[WARN]{Colors.RESET} {msg}")


def info(msg: str) -> None:
    print(f"  {Colors.AZUL}[INFO]{Colors.RESET} {msg}")


def verificar_python():
    """Verifica versión de Python."""
    print(f"\n{Colors.BOLD}1. Verificación de Python{Colors.RESET}")
    version = sys.version.split()[0]
    if sys.version_info >= (3, 9):
        ok(f"Python {version} ({platform.system()} {platform.machine()})")
    else:
        error(f"Python {version} - se requiere 3.9 o superior")
        return False
    return True


def verificar_librerias():
    """Verifica que las librerías necesarias estén instaladas."""
    print(f"\n{Colors.BOLD}2. Verificación de librerías{Colors.RESET}")

    librerias_requeridas = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "datasets": "HuggingFace Datasets",
        "bitsandbytes": "bitsandbytes (cuantización INT8/INT4)",
        "sklearn": "scikit-learn (métricas)",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "plotly": "Plotly (visualización interactiva)",
    }

    todas_ok = True
    for modulo, nombre in librerias_requeridas.items():
        try:
            mod = importlib.import_module(modulo)
            version = getattr(mod, "__version__", "desconocida")
            ok(f"{nombre} ({version})")
        except ImportError:
            error(f"{nombre} NO instalado. Ejecuta: pip install {modulo}")
            todas_ok = False

    return todas_ok


def verificar_gpu():
    """Verifica la disponibilidad y características de la GPU."""
    print(f"\n{Colors.BOLD}3. Verificación de GPU{Colors.RESET}")

    try:
        import torch
    except ImportError:
        error("PyTorch no instalado, no se puede verificar GPU")
        return False

    if not torch.cuda.is_available():
        error("CUDA NO disponible. PyTorch no detecta GPU.")
        info("Instala PyTorch con CUDA: https://pytorch.org/get-started/locally/")
        return False

    # Información de GPU
    nombre_gpu = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9

    ok(f"GPU: {nombre_gpu}")
    ok(f"VRAM total: {vram_gb:.1f} GB")
    ok(f"Compute capability: {props.major}.{props.minor}")
    ok(f"CUDA versión (PyTorch): {torch.version.cuda}")

    # Verificar VRAM suficiente
    if vram_gb >= 8.0:
        ok(f"VRAM suficiente para el pipeline completo (>= 8 GB)")
    elif vram_gb >= 4.0:
        warn(f"VRAM limitada ({vram_gb:.1f} GB). Modo demo recomendado.")
    else:
        error(f"VRAM insuficiente ({vram_gb:.1f} GB). Se requieren mínimo 4 GB.")
        return False

    # Verificar nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            ok(f"nvidia-smi disponible: {result.stdout.strip()}")
        else:
            warn("nvidia-smi no respondió correctamente")
    except FileNotFoundError:
        warn("nvidia-smi no encontrado en PATH")
    except Exception as e:
        warn(f"Error ejecutando nvidia-smi: {e}")

    return True


def verificar_bitsandbytes():
    """Verifica específicamente bitsandbytes (crítico para cuantización)."""
    print(f"\n{Colors.BOLD}4. Verificación de bitsandbytes{Colors.RESET}")

    try:
        import bitsandbytes as bnb
        ok(f"bitsandbytes versión: {bnb.__version__}")

        # Verificar que detecta CUDA
        if hasattr(bnb, "cuda_setup"):
            setup = bnb.cuda_setup
            if hasattr(setup, "evaluate_cuda_setup"):
                # No ejecutar evaluate_cuda_setup (puede causar warnings)
                pass
        ok("bitsandbytes importado correctamente")

    except ImportError:
        error("bitsandbytes NO instalado. Ejecuta: pip install bitsandbytes")
        return False
    except Exception as e:
        error(f"Error cargando bitsandbytes: {e}")
        info("Si el error es por libcuda, instala los drivers NVIDIA recientes")
        return False

    return True


def verificar_huggingface():
    """Verifica conexión a HuggingFace Hub."""
    print(f"\n{Colors.BOLD}5. Verificación de HuggingFace Hub{Colors.RESET}")

    try:
        from transformers import AutoTokenizer
        info("Intentando descargar tokenizer bert-base-uncased (50 MB)...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        ok(f"Tokenizer cargado: {tokenizer.__class__.__name__}")
        return True
    except Exception as e:
        error(f"No se pudo descargar modelo de HuggingFace: {e}")
        info("Verifica conexión a internet o configura HF_HOME en disco con espacio")
        return False


def verificar_espacio_disco():
    """Verifica espacio en disco disponible."""
    print(f"\n{Colors.BOLD}6. Verificación de espacio en disco{Colors.RESET}")

    import shutil
    total, usado, libre = shutil.disk_usage("/")
    libre_gb = libre / 1e9

    ok(f"Espacio libre en disco: {libre_gb:.1f} GB")

    # Modelos + dataset + caches ~ 3 GB
    if libre_gb < 3.0:
        error(f"Espacio insuficiente. Se requieren mínimo 3 GB para caches.")
        return False
    elif libre_gb < 5.0:
        warn(f"Espacio ajustado. Recomendado: 5+ GB.")
    else:
        ok(f"Espacio suficiente")

    return True


def main():
    print(f"{Colors.BOLD}")
    print("=" * 70)
    print("VERIFICACIÓN DEL ENTORNO LOCAL")
    print("Pipeline de Optimización de Deep Learning")
    print("RTX 4060 Laptop (8 GB VRAM) - Ryzen 7 serie 7000 - 16 GB RAM")
    print("=" * 70)
    print(f"{Colors.RESET}")

    checks = [
        verificar_python(),
        verificar_librerias(),
        verificar_gpu(),
        verificar_bitsandbytes(),
        verificar_espacio_disco(),
    ]

    # HuggingFace solo si todo lo demás está OK
    if all(checks):
        checks.append(verificar_huggingface())
    else:
        print(f"\n{Colors.AMARILLO}Saltando verificación de HuggingFace "
              f"(requiere librerías previas){Colors.RESET}")

    # Resumen final
    print(f"\n{Colors.BOLD}")
    print("=" * 70)
    print("RESUMEN DE VERIFICACIÓN")
    print("=" * 70)
    print(f"{Colors.RESET}")

    todas_ok = all(checks)
    if todas_ok:
        print(f"  {Colors.VERDE}{Colors.BOLD}ENTORNO LISTO{Colors.RESET}")
        print(f"  {Colors.VERDE}Puedes ejecutar: python main.py demo{Colors.RESET}")
    else:
        print(f"  {Colors.ROJO}{Colors.BOLD}ENTORNO INCOMPLETO{Colors.RESET}")
        print(f"  {Colors.ROJO}Revisa los errores arriba antes de continuar.{Colors.RESET}")
        info("Instala dependencias: pip install -r requirements.txt")

    print()
    return 0 if todas_ok else 1


if __name__ == "__main__":
    sys.exit(main())
