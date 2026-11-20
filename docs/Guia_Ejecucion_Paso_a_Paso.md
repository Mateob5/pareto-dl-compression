# Guía de Ejecución Paso a Paso
# Pipeline de Optimización de Deep Learning

**Proyecto:** Optimización Computacional de Modelos de Deep Learning
**Hardware objetivo:** RTX 4060 Laptop (8 GB VRAM), Ryzen 7 serie 7000, 16 GB RAM
**Tiempo total estimado (incluida instalación):** 30 min setup + 1.5-2 horas ejecución

---

## Índice

1. [Pre-requisitos del sistema](#1-pre-requisitos-del-sistema)
2. [Verificación de hardware y drivers](#2-verificación-de-hardware-y-drivers)
3. [Instalación de Python](#3-instalación-de-python)
4. [Descarga del proyecto](#4-descarga-del-proyecto)
5. [Creación del entorno virtual (venv)](#5-creación-del-entorno-virtual-venv)
6. [Activación del entorno virtual](#6-activación-del-entorno-virtual)
7. [Actualización de pip y herramientas base](#7-actualización-de-pip-y-herramientas-base)
8. [Instalación de PyTorch con CUDA 12.1](#8-instalación-de-pytorch-con-cuda-121)
9. [Instalación del resto de dependencias](#9-instalación-del-resto-de-dependencias)
10. [Verificación completa del entorno](#10-verificación-completa-del-entorno)
11. [Pre-checks antes de la primera ejecución](#11-pre-checks-antes-de-la-primera-ejecución)
12. [Ejecución 1: Verificación del entorno](#12-ejecución-1-verificación-del-entorno)
13. [Ejecución 2: Visualización rápida con datos de referencia](#13-ejecución-2-visualización-rápida-con-datos-de-referencia)
14. [Ejecución 3: Modo demo (4 configs clave)](#14-ejecución-3-modo-demo-4-configs-clave)
15. [Ejecución 4: Pipeline completo (22 configs)](#15-ejecución-4-pipeline-completo-22-configs)
16. [Ejecución 5: Visualización final con resultados reales](#15-ejecución-5-visualización-final-con-resultados-reales)
17. [Checklist final antes de entregar al profesor](#16-checklist-final-antes-de-entregar-al-profesor)
18. [Solución de problemas comunes](#17-solución-de-problemas-comunes)

---

## 1. Pre-requisitos del sistema

Antes de empezar, verifica que tu sistema cumple estos requisitos mínimos:

### Hardware mínimo

| Componente | Mínimo | Recomendado | Tu equipo |
|---|---|---|---|
| GPU NVIDIA | 4 GB VRAM (modo demo) | 8 GB VRAM (completo) | ✓ RTX 4060 Laptop |
| RAM | 8 GB | 16 GB | ✓ 16 GB |
| CPU | 4 núcleos | 8 núcleos | ✓ Ryzen 7 serie 7000 |
| Disco libre | 5 GB | 10 GB | Verificar ↓ |
| Sistema operativo | Windows 10/11, Ubuntu 20.04+, macOS | Windows 11 / Ubuntu 22.04 | Cualquiera |

### Software mínimo

- **Python 3.9, 3.10 o 3.11** (NO 3.12+, bitsandbytes puede tener problemas)
- **pip** versión 23+
- **Git** (opcional, para clonar el proyecto)
- **Drivers NVIDIA** versión 525+ (para CUDA 12.1)
- **Visual C++ Redistributable** (solo Windows)

### Verificación rápida del sistema (antes de empezar)

**Linux (Ubuntu/Debian):**
```bash
# Verificar GPU NVIDIA
lspci | grep -i nvidia

# Verificar RAM
free -h

# Verificar espacio en disco
df -h .

# Verificar Python instalado
python3 --version

# Verificar pip
python3 -m pip --version
```

**Windows (PowerShell):**
```powershell
# Verificar GPU NVIDIA
nvidia-smi

# Verificar RAM
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory

# Verificar espacio en disco
Get-PSDrive C

# Verificar Python instalado
python --version

# Verificar pip
python -m pip --version
```

**Check ✅:** Antes de continuar, asegúrate de:
- [ ] Tener una GPU NVIDIA visible en el sistema
- [ ] Tener al menos 5 GB libres en disco
- [ ] Tener Python 3.9-3.11 instalado (si no, ir al paso 3)

---

## 2. Verificación de hardware y drivers

Antes de instalar nada, verifica que los drivers NVIDIA están correctamente instalados y que la GPU es visible.

### 2.1. Verificar drivers NVIDIA

**Linux:**
```bash
# Verificar versión del driver
nvidia-smi

# Debes ver algo como:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.171.04   Driver Version: 535.171.04   CUDA Version: 12.2     |
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA GeForce RTX 4060 ...  00000000:01:00.0  On |                  N/A |
# +-----------------------------------------------------------------------------+
```

**Windows:**
```powershell
nvidia-smi
# Misma salida esperada
```

### 2.2. Si no tienes drivers o son muy antiguos

**Linux (Ubuntu):**
```bash
# Añadir repositorio de drivers NVIDIA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Instalar driver recomendado (535 o superior para CUDA 12.1)
sudo apt install nvidia-driver-535

# Reiniciar
sudo reboot
```

**Windows:**
1. Visita https://www.nvidia.com/Download/index.aspx
2. Selecciona "GeForce RTX 4060 Laptop GPU"
3. Descarga e instala el "Game Ready Driver" o "Studio Driver" (versión 535+)
4. Reinicia el equipo

### 2.3. Verificar CUDA disponible (no toolkit, solo runtime)

```bash
# Después de instalar drivers, nvidia-smi debe mostrar "CUDA Version: 12.x"
nvidia-smi | grep "CUDA Version"
```

**Check ✅:**
- [ ] `nvidia-smi` funciona y muestra tu RTX 4060
- [ ] La versión de CUDA mostrada es 12.1 o superior
- [ ] Tienes al menos 8 GB de VRAM total visible

> **Nota importante:** NO necesitas instalar el CUDA Toolkit por separado. PyTorch ya viene con las librerías CUDA necesarias. Solo necesitas los drivers.

---

## 3. Instalación de Python

Si ya tienes Python 3.9-3.11, salta al paso 4.

### 3.1. Verificar versión instalada

```bash
python --version
# o en Linux:
python3 --version
```

**Versiones compatibles:**
- ✅ Python 3.9.x
- ✅ Python 3.10.x (recomendado)
- ✅ Python 3.11.x
- ❌ Python 3.12+ (bitsandbytes puede tener problemas)
- ❌ Python 3.8 o inferior (no soportado por PyTorch reciente)

### 3.2. Instalar Python si no lo tienes

**Linux (Ubuntu):**
```bash
# Instalar Python 3.10 y herramientas
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev python3-pip

# Verificar instalación
python3.10 --version
```

**Windows:**
1. Visita https://www.python.org/downloads/
2. Descarga "Python 3.10.x" (último estable de 3.10)
3. Ejecuta el instalador
4. **IMPORTANTE:** Marca la casilla "Add Python 3.10 to PATH" al inicio
5. Selecciona "Customize installation" → marca "pip" y "py launcher"
6. Completa la instalación
7. Abre un nuevo PowerShell y verifica:
```powershell
python --version
pip --version
```

**macOS (con Homebrew):**
```bash
brew install python@3.10
python3.10 --version
```

**Check ✅:**
- [ ] `python --version` muestra 3.9, 3.10 o 3.11
- [ ] `pip --version` funciona
- [ ] En Linux: `python3 -m venv` funciona (si no: `sudo apt install python3-venv`)

---

## 4. Descarga del proyecto

El proyecto está en la carpeta `po-concepto-local/`. Si lo recibiste como archivo ZIP:

### 4.1. Si tienes un archivo ZIP

**Linux:**
```bash
# Descomprimir en tu directorio de trabajo
cd ~/proyectos  # o donde quieras ubicarlo
unzip po-concepto-local.zip
cd po-concepto-local
```

**Windows:**
1. Clic derecho sobre `po-concepto-local.zip` → "Extraer todo..."
2. Selecciona una ruta corta, por ejemplo `C:\proyectos\po-concepto-local`
3. Evita rutas con espacios o acentos (no uses `C:\Mi Carpeta\proyectos`)
4. Abre PowerShell en esa carpeta:
```powershell
cd C:\proyectos\po-concepto-local
```

### 4.2. Verificar estructura del proyecto

```bash
# Debes ver estos archivos:
ls
# En Windows: dir
```

Debes ver:
```
README.md
requirements.txt
verificar_entorno.py
main.py
config.py
src/
resultados/
docs/
```

### 4.3. Verificar archivos críticos

```bash
# Verificar que los archivos principales existen
ls config.py main.py verificar_entorno.py requirements.txt
ls src/experimentos/pipeline.py
ls src/experimentos/configuraciones.py
ls src/tecnicas/cuantizacion.py
ls src/tecnicas/poda.py
ls src/tecnicas/destilacion.py
```

**Check ✅:**
- [ ] La carpeta `po-concepto-local/` contiene `main.py`, `config.py`, `requirements.txt`
- [ ] La subcarpeta `src/` contiene `experimentos/`, `tecnicas/`, `metricas/`, `data/`, `models/`, `analisis/`
- [ ] La ruta no contiene espacios ni acentos

> **⚠️ Importante:** Si la ruta contiene espacios (ej. `C:\Mi Proyecto\po-concepto-local`), muévela a una ruta sin espacios. PyTorch y bitsandbytes pueden tener problemas con rutas con espacios.

---

## 5. Creación del entorno virtual (venv)

**¿Por qué usar un venv?** Sí, es **muy recomendable**. Razones:
- Aísla las dependencias del proyecto (no afecta otros proyectos Python)
- Evita conflictos de versiones con paquetes del sistema
- Permite reproducibilidad exacta
- Facilita la depuración (puedes recrear el entorno limpio si algo falla)
- Es estándar en proyectos Python profesionales

### 5.1. Crear el entorno virtual

**Linux (desde la carpeta del proyecto):**
```bash
cd ~/proyectos/po-concepto-local  # ajusta tu ruta

# Crear venv llamado "venv" (puedes llamarlo como quieras)
python3 -m venv venv

# Verificar que se creó
ls venv
# Debes ver: bin/  include/  lib/  pyvenv.cfg
```

**Windows (PowerShell desde la carpeta del proyecto):**
```powershell
cd C:\proyectos\po-concepto-local  # ajusta tu ruta

# Crear venv
python -m venv venv

# Verificar que se creó
dir venv
# Debes ver: Scripts/  Include/  Lib/  pyvenv.cfg
```

**macOS:**
```bash
cd ~/proyectos/po-concepto-local
python3 -m venv venv
```

### 5.2. Tamaño esperado del venv

El venv recién creado ocupa ~30 MB. Después de instalar todo, ocupará ~5 GB (principalmente PyTorch + CUDA).

**Check ✅:**
- [ ] La carpeta `venv/` existe dentro de `po-concepto-local/`
- [ ] Contiene `Scripts/` (Windows) o `bin/` (Linux/Mac)
- [ ] Contiene `pyvenv.cfg`

---

## 6. Activación del entorno virtual

**IMPORTANTE:** Debes activar el venv **cada vez** que abras una nueva terminal para trabajar en el proyecto.

### 6.1. Activar en Linux / macOS

```bash
# Desde la carpeta del proyecto
source venv/bin/activate

# Verificar que está activo: el prompt cambia a (venv)
# (venv) usuario@equipo:~/proyectos/po-concepto-local$
```

### 6.2. Activar en Windows (PowerShell)

```powershell
# Desde la carpeta del proyecto
.\venv\Scripts\Activate.ps1

# Verificar que está activo: el prompt cambia a (venv)
# (venv) PS C:\proyectos\po-concepto-local>
```

### 6.3. Activar en Windows (CMD - Command Prompt)

```cmd
cd C:\proyectos\po-concepto-local
venv\Scripts\activate.bat
```

### 6.4. Si Windows bloquea la ejecución del script (error Execution Policy)

Si al activar en PowerShell ves: *"Activate.ps1 cannot be loaded because running scripts is disabled"*

```powershell
# Solución: cambiar la política de ejecución (solo para tu usuario)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Vuelve a intentar la activación
.\venv\Scripts\Activate.ps1
```

### 6.5. Verificar que el venv está activo

```bash
# En Linux/Mac
which python
# Debe mostrar: /ruta/al/proyecto/venv/bin/python

# En Windows
where python
# Debe mostrar: C:\ruta\al\proyecto\venv\Scripts\python.exe
```

### 6.6. Cómo desactivar el venv (cuando termines)

```bash
deactivate
# En Windows también: .\venv\Scripts\deactivate.bat
```

**Check ✅:**
- [ ] El prompt de la terminal muestra `(venv)` al inicio
- [ ] `which python` (Linux/Mac) o `where python` (Windows) apunta al venv
- [ ] No estás usando el Python del sistema

---

## 7. Actualización de pip y herramientas base

Antes de instalar nada, actualiza pip a la última versión.

### 7.1. Actualizar pip

```bash
# Con el venv activado
python -m pip install --upgrade pip setuptools wheel

# Verificar versión
python -m pip --version
# Debe mostrar pip 23.x o superior
```

### 7.2. Configurar pip para usar cache (opcional pero recomendado)

Esto acelera reinstalaciones si necesitas recrear el venv:

```bash
# Linux/Mac
export PIP_CACHE_DIR=~/.cache/pip
mkdir -p $PIP_CACHE_DIR

# Windows PowerShell
$env:PIP_CACHE_DIR = "$env:USERPROFILE\.cache\pip"
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR
```

**Check ✅:**
- [ ] `python -m pip --version` muestra versión 23+
- [ ] No hay errores en la actualización

---

## 8. Instalación de PyTorch con CUDA 12.1

Este es el paso **más crítico**. PyTorch debe instalarse con el index-url correcto para que incluya soporte CUDA.

### 8.1. Instalar PyTorch con CUDA 12.1

**Linux / Windows / macOS (con el venv activado):**
```bash
# IMPORTANTE: usar exactamente este comando
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **Tamaño:** ~2.5 GB de descarga. Toma 5-15 minutos según tu conexión.
> **No uses `pip install torch` sin el index-url**, eso instalará la versión CPU-only.

### 8.2. Verificar instalación de PyTorch

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Versión CUDA: {torch.version.cuda}')"
```

**Salida esperada:**
```
PyTorch: 2.1.x+cu121
CUDA disponible: True
Versión CUDA: 12.1
```

### 8.3. Verificar que PyTorch detecta tu GPU

```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
```

**Salida esperada:**
```
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
VRAM total: 8.0 GB
```

**Check ✅:**
- [ ] `torch.__version__` contiene `+cu121`
- [ ] `torch.cuda.is_available()` retorna `True`
- [ ] `torch.cuda.get_device_name(0)` muestra "RTX 4060"
- [ ] VRAM total muestra ~8 GB

**❌ Si algo falla aquí:**
- Si `CUDA disponible: False` → reinstala con el index-url correcto
- Si versión CUDA es None → drivers NVIDIA no están instalados o son muy antiguos
- Ver sección [17. Solución de problemas](#17-solución-de-problemas-comunes)

---

## 9. Instalación del resto de dependencias

### 9.1. Instalar desde requirements.txt

```bash
# Asegúrate de estar en la carpeta del proyecto con venv activado
pip install -r requirements.txt
```

> **Tamaño:** ~1 GB de descarga (transformers, bitsandbytes, datasets, plotly, etc.)
> **Tiempo:** 5-10 minutos

### 9.2. Si algún paquete falla durante la instalación

Instala los problemáticos individualmente:

```bash
# bitsandbytes (crítico para cuantización)
pip install bitsandbytes

# transformers
pip install transformers>=4.36.0

# datasets (para IMDb)
pip install datasets>=2.16.0

# plotly (para visualización interactiva)
pip install plotly

# Otros
pip install scikit-learn pandas matplotlib seaborn tqdm psutil
```

### 9.3. Verificar todas las dependencias

```bash
python -c "
import torch; print(f'torch: {torch.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
import datasets; print(f'datasets: {datasets.__version__}')
import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')
import sklearn; print(f'scikit-learn: {sklearn.__version__}')
import pandas; print(f'pandas: {pandas.__version__}')
import matplotlib; print(f'matplotlib: {matplotlib.__version__}')
import plotly; print(f'plotly: {plotly.__version__}')
import numpy; print(f'numpy: {numpy.__version__}')
print()
print('=== TODAS LAS DEPENDENCIAS OK ===')
"
```

**Salida esperada:**
```
torch: 2.1.x+cu121
transformers: 4.36.x
datasets: 2.16.x
bitsandbytes: 0.41.x
scikit-learn: 1.3.x
pandas: 2.1.x
matplotlib: 3.7.x
plotly: 5.18.x
numpy: 1.24.x

=== TODAS LAS DEPENDENCIAS OK ===
```

**Check ✅:**
- [ ] Todos los imports funcionan sin error
- [ ] Las versiones son las indicadas o superiores
- [ ] `bitsandbytes` carga sin errores de CUDA

---

## 10. Verificación completa del entorno

El proyecto incluye un script automatizado que verifica todo en un solo comando.

### 10.1. Ejecutar el verificador

```bash
# Con venv activado, desde la carpeta del proyecto
python verificar_entorno.py
```

### 10.2. Salida esperada

```
======================================================================
VERIFICACIÓN DEL ENTORNO LOCAL
Pipeline de Optimización de Deep Learning
RTX 4060 Laptop (8 GB VRAM) - Ryzen 7 serie 7000 - 16 GB RAM
======================================================================

1. Verificación de Python
  [OK] Python 3.10.x (Linux x86_64)

2. Verificación de librerías
  [OK] PyTorch (2.1.x)
  [OK] HuggingFace Transformers (4.36.x)
  [OK] HuggingFace Datasets (2.16.x)
  [OK] bitsandbytes (0.41.x)
  [OK] scikit-learn (1.3.x)
  [OK] NumPy (1.24.x)
  [OK] Pandas (2.1.x)
  [OK] Matplotlib (3.7.x)
  [OK] Plotly (5.18.x)

3. Verificación de GPU
  [OK] GPU: NVIDIA GeForce RTX 4060 Laptop GPU
  [OK] VRAM total: 8.0 GB
  [OK] Compute capability: 8.9
  [OK] CUDA versión (PyTorch): 12.1
  [OK] VRAM suficiente para el pipeline completo (>= 8 GB)

4. Verificación de bitsandbytes
  [OK] bitsandbytes versión: 0.41.x
  [OK] bitsandbytes importado correctamente

5. Verificación de espacio en disco
  [OK] Espacio libre en disco: XX.X GB

6. Verificación de HuggingFace Hub
  [INFO] Intentando descargar tokenizer bert-base-uncased (50 MB)...
  [OK] Tokenizer cargado: BertTokenizerFast

======================================================================
RESUMEN DE VERIFICACIÓN
======================================================================

  ENTORNO LISTO
  Puedes ejecutar: python main.py demo
```

### 10.3. Si hay algún [ERROR]

Lee el mensaje de error específico y ve a la sección [17. Solución de problemas](#17-solución-de-problemas-comunes).

Los errores más comunes:
- `[ERROR] bitsandbytes NO instalado` → `pip install bitsandbytes`
- `[ERROR] CUDA NO disponible` → reinstala PyTorch con el index-url correcto
- `[ERROR] No se pudo descargar modelo de HuggingFace` → verifica conexión a internet

**Check ✅:**
- [ ] `python verificar_entorno.py` muestra "ENTORNO LISTO"
- [ ] No hay errores en ninguna sección
- [ ] Las advertencias (warnings) son aceptables

---

## 11. Pre-checks antes de la primera ejecución

Antes de ejecutar el pipeline, haz estos últimos chequeos:

### 11.1. Verificar que no hay otros procesos usando la GPU

```bash
# En Linux/Mac/Windows
nvidia-smi
```

**Importante:** La sección "Processes" debe estar vacía o solo mostrar procesos ligeros. Si hay procesos pesados consumiendo VRAM (juegos, navegadores con GPU acceleration, otras instancias de Python):
- Cierra Steam, juegos, navegadores pesados
- Espera a que `nvidia-smi` muestre < 500 MB usados

### 11.2. Verificar temperatura de la GPU (opcional pero recomendado)

```bash
# En Linux
nvidia-smi -q -d TEMPERATURE
```

La temperatura idle debe ser < 50°C. Si está más alta, espera unos minutos o mejora la ventilación de tu laptop.

### 11.3. Verificar espacio en disco

```bash
# Linux
df -h .

# Windows
Get-PSDrive C
```

Necesitas mínimo 3 GB libres adicionales para los caches de HuggingFace y del modelo destilado.

### 11.4. Conexión a internet estable

La primera ejecución descargará:
- **BERT-base-uncased:** ~440 MB
- **DistilBERT-base-uncased:** ~270 MB
- **Dataset IMDb:** ~80 MB
- **Total: ~790 MB**

Verifica con:
```bash
# Test rápido de conexión a HuggingFace
curl -I https://huggingface.co  # Linux/Mac
# o
Invoke-WebRequest -Method Head -Uri https://huggingface.co  # Windows PowerShell
```

### 11.5. Cerrar aplicaciones que puedan interferir

Cierra:
- Navegadores con muchas pestañas (consumen VRAM si tienen GPU acceleration)
- Steam, Epic Games, Discord overlay
- Otros IDEs Python (PyCharm, VS Code pueden consumir VRAM)
- Procesos de backup o antivirus intensivos

**Check ✅:**
- [ ] `nvidia-smi` muestra < 500 MB usados antes de empezar
- [ ] Temperatura GPU idle < 50°C
- [ ] Al menos 3 GB libres en disco
- [ ] Conexión a internet estable
- [ ] Aplicaciones pesadas cerradas

---

## 12. Ejecución 1: Verificación del entorno

Este paso ya lo hiciste en el paso 10, pero lo repetimos usando el CLI del proyecto para confirmar que todo funciona desde `main.py`:

```bash
python main.py verificar
```

**Salida esperada:** igual que `python verificar_entorno.py`, debe terminar con "ENTORNO LISTO".

**Tiempo:** 30 segundos.

---

## 13. Ejecución 2: Visualización rápida con datos de referencia

Antes de ejecutar el pipeline completo (que toma 1.5-2 horas), genera la visualización de Pareto con los datos de referencia (valores estimados de literatura) para ver cómo se verá el resultado final.

```bash
python main.py solo-pareto --csv resultados/resultados_referencia.csv
```

**Salida esperada:**
```
======================================================================
ANÁLISIS DE LA FRONTERA DE PARETO
======================================================================
[pareto] Usando resultados de referencia: resultados/resultados_referencia.csv
[pareto] 22 configuraciones cargadas
[pareto] 11 configuraciones en la Frontera de Pareto:
  B0         ahorro=0.0%  precisión=100.00%
  P1         ahorro=9.2%  precisión=99.68%
  Q1         ahorro=44.6%  precisión=99.36%
  ...
[pareto] Codo de la curva: P1Q1 (ahorro=47.7%, precisión=99.14%)
[pareto] PNG estático guardado: resultados/frontera_pareto.png
[pareto] HTML interactivo guardado: resultados/frontera_pareto_interactiva.html
```

**Tiempo:** 5 segundos.

### 13.1. Verificar los archivos generados

```bash
# Linux/Mac
ls -la resultados/frontera_pareto.*
# Debes ver:
# frontera_pareto.png (~155 KB)
# frontera_pareto_interactiva.html (~4.7 MB)

# Windows
dir resultados\frontera_pareto.*
```

### 13.2. Abrir el HTML interactivo en tu navegador

```bash
# Linux
xdg-open resultados/frontera_pareto_interactiva.html

# macOS
open resultados/frontera_pareto_interactiva.html

# Windows
start resultados\frontera_pareto_interactiva.html
```

**Debes ver:** Un gráfico interactivo con los 22 puntos, la Frontera de Pareto como línea discontinua, y el codo marcado con una estrella dorada. Pasa el mouse sobre cada punto para ver sus detalles.

**Check ✅:**
- [ ] `frontera_pareto.png` generado y pesa > 100 KB
- [ ] `frontera_pareto_interactiva.html` generado y abre en el navegador
- [ ] El gráfico muestra 22 puntos y la frontera

---

## 14. Ejecución 3: Modo demo (4 configs clave)

Antes de lanzar las 22 configuraciones, ejecuta el modo demo con solo 4 configs (B0, Q1, D1, D1P3Q2) para validar que todo el pipeline funciona correctamente.

### 14.1. Ejecutar demo

```bash
python main.py demo
```

**Tiempo estimado:** 8-12 minutos.

### 14.2. Qué hace el demo paso a paso

1. Descarga el dataset IMDb (5-10 segundos, ~80 MB)
2. Tokeniza los 3,000 ejemplos (1,500 train + 1,500 test)
3. **B0:** Carga BERT-base, mide VRAM/latencia/accuracy (~30 seg)
4. **Q1:** Carga BERT-base INT8, mide (~1-2 min)
5. **D1:** Entrena DistilBERT destilado, cachea, mide (~5-7 min)
6. **D1P3Q2:** Carga destilado cacheado, cuantiza INT4, mide (~2-3 min)

### 14.3. Salida esperada (resumida)

```
======================================================================
PIPELINE DE OPTIMIZACIÓN - 4 configuraciones
======================================================================

>>> Progreso: 1/4 - Configuración B0
[hardware] Midiendo configuración: B0
  VRAM pico: 1300.5 MB
  Latencia mediana: 24.8 ms
  Throughput: 40.3 ej/s

>>> Progreso: 2/4 - Configuración Q1
...

>>> Progreso: 3/4 - Configuración D1
[pipeline] CACHE MISS: entrenando estudiante destilado por primera vez
[destilacion] Iniciando destilación: T=4.0, alpha=0.5, epochs=1, lr=2e-05
...

>>> Progreso: 4/4 - Configuración D1P3Q2
[pipeline] CACHE HIT: reutilizando estudiante destilado
...

======================================================================
PIPELINE COMPLETADO
# Resultados guardados en: resultados/resultados_usuario.csv
# Configuraciones exitosas: 4/4
======================================================================
```

### 14.4. Verificar resultados del demo

```bash
# Ver el CSV generado
cat resultados/resultados_usuario.csv
# En Windows: type resultados\resultados_usuario.csv
```

**Debes ver 4 filas** con las métricas medidas:
```csv
id,nombre,modelo_base,cuantizacion,poda_pct,destilado,vram_pico_mb,latencia_mediana_ms,...
B0,Línea Base BERT-base FP32,BERT-base,FP32,0%,No,1300.5,24.8,...
Q1,BERT-base INT8 (PTQ),BERT-base,INT8,0%,No,720.3,16.5,...
D1,DistilBERT destilado FP32,DistilBERT,FP32,0%,Sí,820.1,14.5,...
D1P3Q2,DistilBERT + Poda 60% + INT4,DistilBERT,INT4,60%,Sí,220.5,4.8,...
```

### 14.5. Verificar que el cache del destilado se creó

```bash
# Linux/Mac
ls cache_modelos/estudiante_destilado/
# Debes ver: config.json, model.safetensors, tokenizer files

# Windows
dir cache_modelos\estudiante_destilado\
```

**Check ✅:**
- [ ] El demo completa sin errores
- [ ] `resultados_usuario.csv` tiene 4 filas
- [ ] `cache_modelos/estudiante_destilado/` existe con archivos del modelo

**❌ Si el demo falla:**
- Error OOM → reduce `BATCH_SIZE_ENTRENAMIENTO` en `config.py` (de 24 a 16)
- Error descarga HuggingFace → ve a sección 17
- Error bitsandbytes → reinstala con `pip install bitsandbytes --upgrade`

---

## 15. Ejecución 4: Pipeline completo (22 configs)

Ahora sí, ejecuta el pipeline completo. Esto tomará **~1.5-2 horas**.

### 15.1. Preparación importante

**Antes de empezar:**
- ✅ Venv activado (`(venv)` en el prompt)
- ✅ Demo funcionó correctamente (paso 14)
- ✅ Cache del destilado creado (acelera las 13 configs D1*)
- ✅ Aplicaciones cerradas (navegador, Steam, etc.)
- ✅ Laptop conectada a corriente (no en modo ahorro)
- ✅ Ventilación adecuada (no sobre superficies blandas)

### 15.2. Opcional: configurar logging a archivo

Para guardar toda la salida en un archivo (útil para depuración posterior):

**Linux/Mac:**
```bash
python main.py completo 2>&1 | tee resultados/log_ejecucion.txt
```

**Windows (PowerShell):**
```powershell
python main.py completo 2>&1 | Tee-Object -FilePath resultados\log_ejecucion.txt
```

### 15.3. Ejecutar el pipeline completo

```bash
python main.py completo
```

### 15.4. Seguimiento del progreso

Verás mensajes como:
```
>>> Progreso: 1/22 - Configuración B0
>>> B0 completado en 32.5s

>>> Progreso: 2/22 - Configuración Q1
>>> Q1 completado en 95.2s

>>> Progreso: 10/22 - Configuración D1
[pipeline] CACHE HIT: reutilizando estudiante destilado
>>> D1 completado en 8.5s   ← ¡Rápido porque usa cache!
```

### 15.5. Estimación de tiempos por configuración

| # | ID | Tiempo aprox | Notas |
|---|---|---|---|
| 1 | B0 | 30 seg | Línea base |
| 2 | Q1 | 1-2 min | INT8 BERT |
| 3 | Q2 | 1-2 min | INT4 BERT |
| 4 | P1 | 2-3 min | Poda 20% + FT |
| 5 | P2 | 2-3 min | Poda 40% + FT |
| 6 | P3 | 3-4 min | Poda 60% + FT |
| 7-9 | P1Q1, P2Q1, P3Q1 | 2-4 min c/u | BERT combinado |
| 10 | D1 | 5-7 min | Destilación (1ra vez, cachea) |
| 11 | D1Q1 | 1-2 min | Cache + cuantización |
| 12 | D1Q2 | 1-2 min | Cache + cuantización |
| 13-15 | D1P1, D1P2, D1P3 | 2-4 min c/u | Cache + poda + FT |
| 16-21 | D1P*Q* | 2-4 min c/u | Triple combinación |
| 22 | D1P3Q2FT | 4-6 min | + FT adicional |

**Total: ~1.5-2 horas**

### 15.6. Si se interrumpe la ejecución

El pipeline guarda resultados incrementalmente tras cada configuración. Si se interrumpe:

```bash
# Los resultados parciales están seguros en resultados_usuario.csv
cat resultados/resultados_usuario.csv  # ver cuántas filas se completaron

# Para retomar solo las configs que faltan:
# Edita src/experimentos/pipeline.py o crea un script que ejecute solo los IDs faltantes
# Ejemplo: ejecutar solo D1Q2 y D1P2 si esas fallaron
python -c "
import sys; sys.path.insert(0, '.')
from src.experimentos.pipeline import ejecutar_todas_configuraciones
ejecutar_todas_configuraciones(ids_configuraciones=['D1Q2', 'D1P2'])
"
```

**Check ✅:**
- [ ] `resultados_usuario.csv` tiene 22 filas al final
- [ ] El pipeline terminó con "PIPELINE COMPLETADO"
- [ ] No hay `[ERROR]` en la salida

---

## 16. Ejecución 5: Visualización final con resultados reales

Con tus resultados reales en `resultados_usuario.csv`, regenera la visualización de Pareto:

```bash
python main.py solo-pareto
```

**Salida esperada:**
```
======================================================================
ANÁLISIS DE LA FRONTERA DE PARETO
======================================================================
[pareto] Usando resultados reales: resultados/resultados_usuario.csv
[pareto] 22 configuraciones cargadas
[pareto] X configuraciones en la Frontera de Pareto:
  ...
[pareto] Codo de la curva: <ID>
[pareto] PNG estático guardado: resultados/frontera_pareto.png
[pareto] HTML interactivo guardado: resultados/frontera_pareto_interactiva.html
```

> Automáticamente usa `resultados_usuario.csv` si existe, si no cae a `resultados_referencia.csv`.

**Tiempo:** 5 segundos.

### 16.1. Abrir y verificar las visualizaciones

```bash
# PNG estático (para incluir en el documento del proyecto)
# Linux
xdg-open resultados/frontera_pareto.png
# macOS
open resultados/frontera_pareto.png
# Windows
start resultados\frontera_pareto.png

# HTML interactivo (para defensa al profesor)
xdg-open resultados/frontera_pareto_interactiva.html
```

**Check ✅:**
- [ ] PNG regenerado con fecha reciente
- [ ] HTML interactivo muestra 22 puntos
- [ ] La frontera y el codo están marcados
- [ ] Pasa el mouse sobre los puntos y se ven los detalles

---

## 16. Checklist final antes de entregar al profesor

Antes de cerrar el proyecto y entregarlo, verifica estos puntos:

### Archivos generados

- [ ] `resultados/resultados_usuario.csv` - 22 filas con métricas reales
- [ ] `resultados/frontera_pareto.png` - imagen PNG para el documento
- [ ] `resultados/frontera_pareto_interactiva.html` - HTML para defensa
- [ ] `docs/Plan_Prueba_Concepto_Local.pdf` - plan técnico (ya estaba)
- [ ] `cache_modelos/estudiante_destilado/` - cache del destilado (referencia)

### Validaciones finales

```bash
# Verificar que el CSV tiene 22 filas
wc -l resultados/resultados_usuario.csv
# Linux: debe mostrar 23 (22 datos + 1 header)

# Verificar IDs completos
python -c "
import csv
with open('resultados/resultados_usuario.csv') as f:
    reader = csv.DictReader(f)
    ids = [row['id'] for row in reader]
print(f'Total configs: {len(ids)}')
print(f'IDs: {ids}')
expected = ['B0','Q1','Q2','P1','P2','P3','P1Q1','P2Q1','P3Q1','D1','D1Q1','D1Q2','D1P1','D1P2','D1P3','D1P1Q1','D1P2Q1','D1P3Q1','D1P1Q2','D1P2Q2','D1P3Q2','D1P3Q2FT']
missing = set(expected) - set(ids)
if missing:
    print(f'FALTAN: {missing}')
else:
    print('OK: 22/22 configuraciones completas')
"
```

### Verificar que el codo de Pareto tiene sentido

```bash
python -c "
import csv
with open('resultados/resultados_usuario.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
# Mostrar top 3 configs que cumplen criterio (ahorro>=50% Y retencion>=98%)
exitosas = [r for r in rows if r['criterio_aceptacion'] == 'Sí']
print(f'Configuraciones que cumplen criterio de éxito rotundo: {len(exitosas)}')
for r in exitosas:
    print(f'  {r[\"id\"]}: ahorro={r[\"ahorro_vram_pct\"]}%, retención={r[\"precision_retenida_pct\"]}%')
"
```

### Para incluir en el documento final del proyecto

1. **Copia `frontera_pareto.png`** a la carpeta de imágenes de tu documento
2. **Copia `resultados_usuario.csv`** como anexo digital
3. **Incluye `frontera_pareto_interactiva.html`** como recurso interactivo
4. **Referencia el código fuente** como anexo digital o repositorio Git

### Estructura final de archivos a entregar

```
entrega_proyecto/
├── documento_principal.docx (o .pdf)  # Tu documento del proyecto
├── codigo/
│   └── po-concepto-local/              # Todo el proyecto Python
│       ├── src/
│       ├── resultados/
│       │   ├── resultados_usuario.csv  # ← TUS RESULTADOS REALES
│       │   ├── frontera_pareto.png     # ← Imagen para el documento
│       │   └── frontera_pareto_interactiva.html
│       ├── docs/
│       │   └── Plan_Prueba_Concepto_Local.pdf
│       ├── README.md
│       ├── requirements.txt
│       └── main.py
└── anexos/
    ├── log_ejecucion.txt               # (opcional, si guardaste log)
    └── capturas_defensa/               # Capturas del HTML interactivo
```

---

## 17. Solución de problemas comunes

### Problema 1: Error "CUDA out of memory" (OOM)

**Síntoma:** `RuntimeError: CUDA out of memory. Tried to allocate X MiB`

**Solución:**
```bash
# 1. Verificar procesos que usan GPU
nvidia-smi

# 2. Cerrar otros procesos (juegos, navegador con GPU accel)

# 3. Reducir batch size en config.py
# Editar config.py y cambiar:
#   BATCH_SIZE_ENTRENAMIENTO = 24  →  BATCH_SIZE_ENTRENAMIENTO = 16
# O reducir max_len:
#   MAX_LEN = 128  →  MAX_LEN = 96

# 4. Si persiste, limpiar cache y reintentar
python main.py limpiar-cache
python main.py completo
```

### Problema 2: PyTorch no detecta CUDA

**Síntoma:** `torch.cuda.is_available()` retorna `False`

**Diagnóstico:**
```bash
python -c "import torch; print(torch.__version__)"
# Si la versión NO contiene '+cu121' o '+cu118', instalaste versión CPU
```

**Solución:**
```bash
# Desinstalar torch CPU
pip uninstall torch torchvision -y

# Reinstalar con CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verificar
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Problema 3: bitsandbytes no detecta GPU

**Síntoma:** Error al cargar bitsandbytes o `RuntimeError: CUDA not detected`

**Solución:**
```bash
# Verificar instalación
pip show bitsandbytes

# Reinstalar versión más reciente
pip install bitsandbytes --upgrade

# Verificar
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

### Problema 4: Error descargando modelo de HuggingFace

**Síntoma:** `OSError: Couldn't reach server` o `ConnectionError`

**Solución 1 - Reintentar (la red puede haber tenido un fallo temporal):**
```bash
python main.py completo  # reintentar
```

**Solución 2 - Usar mirror (si estás detrás de firewall o en China):**
```bash
# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com
python main.py completo

# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python main.py completo
```

**Solución 3 - Descargar manualmente:**
```bash
pip install huggingface-hub
huggingface-cli download bert-base-uncased --local-dir cache_modelos/bert-base-uncased
huggingface-cli download distilbert-base-uncased --local-dir cache_modelos/distilbert-base-uncased
```

### Problema 5: Latencia muy variable entre ejecuciones

**Síntoma:** Las mediciones de latencia cambian ±50% entre ejecuciones del mismo modelo.

**Causas:**
1. Throttling térmico (laptop se calienta)
2. Scheduler del sistema operativo
3. Procesos en background

**Solución:**
- Ejecuta con la laptop en superficie dura y ventilada
- Usa un pad refrigerante si es posible
- Cierra todas las aplicaciones no esenciales
- Conecta la laptop a corriente (no modo batería)
- Repite las mediciones 3 veces y reporta la mediana

### Problema 6: Error "Permission denied" al activar venv en Windows

**Síntoma:** `Activate.ps1 cannot be loaded because running scripts is disabled`

**Solución:**
```powershell
# Cambiar política de ejecución para tu usuario
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Confirmar con 'Y' cuando pregunte

# Reintentar activación
.\venv\Scripts\Activate.ps1
```

### Problema 7: Error "ModuleNotFoundError: No module named 'src'"

**Síntoma:** Al ejecutar `python main.py`, error de import.

**Solución:** Asegúrate de estar ejecutando desde la carpeta correcta:
```bash
# Verifica que estás en la carpeta del proyecto
pwd  # Linux/Mac
# o
echo %cd%  # Windows CMD

# Debe mostrar la ruta a po-concepto-local/
# Si no, cd a esa carpeta:
cd ruta/a/po-concepto-local

# Volver a ejecutar
python main.py verificar
```

### Problema 8: El CSV solo tiene algunas filas (ejecución interrumpida)

**Síntoma:** `resultados_usuario.csv` tiene menos de 22 filas.

**Solución - retomar solo las que faltan:**
```bash
# Ver qué IDs faltan
python -c "
import csv
with open('resultados/resultados_usuario.csv') as f:
    reader = csv.DictReader(f)
    done = set(row['id'] for row in reader)
all_ids = ['B0','Q1','Q2','P1','P2','P3','P1Q1','P2Q1','P3Q1','D1','D1Q1','D1Q2','D1P1','D1P2','D1P3','D1P1Q1','D1P2Q1','D1P3Q1','D1P1Q2','D1P2Q2','D1P3Q2','D1P3Q2FT']
missing = [i for i in all_ids if i not in done]
print(f'Faltan: {missing}')
"

# Ejecutar solo las que faltan (ejemplo)
python -c "
import sys; sys.path.insert(0, '.')
from src.experimentos.pipeline import ejecutar_todas_configuraciones
ejecutar_todas_configuraciones(ids_configuraciones=['D1P3Q1', 'D1P3Q2', 'D1P3Q2FT'])
"
```

> **Nota:** Esto sobrescribe `resultados_usuario.csv`. Si quieres preservar lo existente, renómbralo primero o modifica el parámetro `archivo_salida`.

### Problema 9: Versión de Python incompatibilidad

**Síntoma:** `ImportError` o errores raros al importar bitsandbytes.

**Diagnóstico:**
```bash
python --version
# Si tienes 3.12+, bitsandbytes puede tener problemas
```

**Solución:** Instalar Python 3.10 y recrear venv:
```bash
# Linux
sudo apt install python3.10 python3.10-venv
python3.10 -m venv venv_nuevo
source venv_nuevo/bin/activate
pip install -r requirements.txt

# Windows: descargar Python 3.10 de python.org y reinstalar
```

### Problema 10: Otros errores no listados

Si encuentras un error no documentado:

1. **Lee el stack trace completo** - usualmente indica el archivo y línea del problema
2. **Copia el error exacto** y búscalo en Google o Stack Overflow
3. **Verifica tu entorno con** `python verificar_entorno.py`
4. **Revisa que el venv está activado** (`(venv)` en el prompt)
5. **Revisa que tienes espacio en disco y memoria RAM libre**

---

## Resumen final - Comandos esenciales

```bash
# === SETUP (una sola vez) ===
python -m venv venv                    # Crear venv
source venv/bin/activate              # Activar (Linux/Mac)
# o: .\venv\Scripts\Activate.ps1      # Activar (Windows)

pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# === VERIFICACIÓN ===
python verificar_entorno.py           # Verifica todo
python main.py verificar              # Igual pero vía CLI

# === EJECUCIÓN ===
python main.py solo-pareto --csv resultados/resultados_referencia.csv  # Ver Pareto rápido
python main.py demo                   # Demo 4 configs (~10 min)
python main.py completo               # 22 configs (~1.5-2 horas)
python main.py solo-pareto            # Visualización con resultados reales
python main.py listar                 # Listar las 22 configs
python main.py limpiar-cache          # Borrar cache del destilado

# === DESACTIVAR VENV AL TERMINAR ===
deactivate
```

---

## Tiempos totales estimados

| Etapa | Tiempo |
|---|---|
| Instalación de drivers + Python | 15-20 min |
| Creación de venv + instalación de deps | 15-20 min |
| Verificación del entorno | 1 min |
| Visualización con datos de referencia | 1 min |
| Demo (4 configs) | 8-12 min |
| Pipeline completo (22 configs) | 1.5-2 horas |
| Visualización final | 1 min |
| **Total desde cero hasta entrega** | **~2.5-3 horas** |

---

**Guía generada para el proyecto:** Optimización Computacional de Modelos de Deep Learning
**Autor:** Jerónimo Bazán (2243590) — Universidad Industrial de Santander
**Profesor:** Carlos Niño — Aprendizaje Profundo
**Noviembre 2026**
