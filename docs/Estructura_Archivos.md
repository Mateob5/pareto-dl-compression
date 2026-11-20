# Estructura de Archivos del Proyecto

Documento de referencia para verificar que tienes todos los archivos antes de ejecutar el pipeline.

---

## Estado INICIAL (lo que descargas)

Esta es la estructura que debes tener justo después de descargar/descomprimir el proyecto, **antes de ejecutar cualquier comando**:

```
po-concepto-local/                          ← Carpeta raíz del proyecto
│
├── README.md                               # Documentación principal del proyecto
├── requirements.txt                        # Lista de dependencias Python
├── verificar_entorno.py                    # Script de verificación de GPU/CUDA
├── main.py                                 # Punto de entrada (CLI con argparse)
├── config.py                               # Configuración global del pipeline
│
├── src/                                    # Código fuente modular
│   ├── __init__.py                         # (vacío, hace que src sea un paquete)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── imdb_loader.py                  # Carga del dataset IMDb (subset 1,500 ej.)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── profesor.py                     # Wrapper BERT-base (110M parámetros)
│   │   └── estudiante.py                   # Wrapper DistilBERT (66M parámetros)
│   │
│   ├── tecnicas/
│   │   ├── __init__.py
│   │   ├── cuantizacion.py                 # PTQ INT8/INT4 con bitsandbytes
│   │   ├── poda.py                         # Poda estructurada de cabezas de atención
│   │   └── destilacion.py                  # KL + CrossEntropy + temperatura T=4
│   │
│   ├── metricas/
│   │   ├── __init__.py
│   │   ├── hardware.py                     # VRAM pico, latencia, throughput
│   │   └── precision.py                    # Accuracy, F1, % retención
│   │
│   ├── experimentos/
│   │   ├── __init__.py
│   │   ├── configuraciones.py              # Definición de las 22 configs
│   │   └── pipeline.py                     # Orquestador principal
│   │
│   └── analisis/
│       ├── __init__.py
│       └── pareto.py                       # Frontera de Pareto + visualización
│
├── resultados/                             # Resultados y visualizaciones
│   ├── generar_referencia.py               # Script que genera el CSV de referencia
│   ├── resultados_referencia.csv           # Valores estimados (literatura)
│   ├── frontera_pareto.png                 # PNG estático (Matplotlib)
│   └── frontera_pareto_interactiva.html    # HTML interactivo (Plotly)
│
└── docs/                                   # Documentación técnica
    ├── Plan_Prueba_Concepto_Local.pdf      # Plan técnico (19 páginas)
    └── Guia_Ejecucion_Paso_a_Paso.md       # Guía de instalación y ejecución
```

**Total:** 28 archivos en 8 carpetas
**Tamaño inicial:** ~6 MB (sin dependencias instaladas)

---

## Verificación rápida (antes de empezar)

Ejecuta este comando en la carpeta del proyecto para verificar que tienes todo:

```bash
# Linux / macOS
ls -la
ls -la src/
ls -la resultados/
ls -la docs/

# Windows PowerShell
dir
dir src
dir resultados
dir docs
```

**Debes ver exactamente:**

```bash
# En la raíz del proyecto:
README.md
config.py
main.py
requirements.txt
verificar_entorno.py
src/
resultados/
docs/

# En src/:
__init__.py
analisis/
data/
experimentos/
metricas/
models/
tecnicas/

# En resultados/:
frontera_pareto.png
frontera_pareto_interactiva.html
generar_referencia.py
resultados_referencia.csv

# En docs/:
Guia_Ejecucion_Paso_a_Paso.md
Plan_Prueba_Concepto_Local.pdf
```

---

## Estado DESPUÉS de ejecutar (lo que se genera)

Después de crear el venv, instalar dependencias y ejecutar el pipeline completo, aparecerán estas carpetas/archivos adicionales:

```
po-concepto-local/
│
├── [todos los archivos anteriores...]
│
├── venv/                                   # ⚡ SE CREA AL HACER: python -m venv venv
│   ├── bin/  (Linux/Mac)                   #    ~30 MB inicial
│   ├── Scripts/ (Windows)                  #    ~5 GB después de instalar deps
│   ├── lib/  (Linux/Mac)
│   ├── Lib/  (Windows)
│   └── pyvenv.cfg
│
├── cache_modelos/                          # ⚡ SE CREA AUTOMÁTICAMENTE (config.py)
│   └── estudiante_destilado/               # ⚡ SE CREA EN LA EJECUCIÓN DE D1
│       ├── config.json                     #    Creado al entrenar D1 la primera vez
│       ├── model.safetensors               #    ~270 MB (pesos del DistilBERT destilado)
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.txt
│
├── cache_dataset/                          # ⚡ SE CREA AL CARGAR IMDb
│   └── [archivos cache HuggingFace]        #    ~80 MB (dataset IMDb cacheado)
│
└── resultados/
    └── resultados_usuario.csv              # ⚡ SE CREA AL EJECUTAR el pipeline
                                             #    22 filas con tus métricas reales
```

---

## Tabla explicativa de cada archivo

### Archivos raíz (5)

| Archivo | Propósito | ¿Cuándo se usa? |
|---|---|---|
| `README.md` | Documentación completa del proyecto | Referencia en cualquier momento |
| `requirements.txt` | Lista de dependencias Python | Solo en instalación (`pip install -r`) |
| `verificar_entorno.py` | Verifica GPU, CUDA, deps (6 chequeos) | Antes de ejecutar el pipeline |
| `main.py` | Punto de entrada con CLI argparse | Siempre (`python main.py <modo>`) |
| `config.py` | Configuración global, semillas, rutas | Importado por todos los módulos |

### Carpeta `src/` (código fuente)

| Archivo | Propósito | Líneas aprox. |
|---|---|---|
| `src/data/imdb_loader.py` | Descarga IMDb, tokeniza, crea DataLoaders | ~100 |
| `src/models/profesor.py` | Wrapper BERT-base (110M parámetros) | ~70 |
| `src/models/estudiante.py` | Wrapper DistilBERT (66M parámetros) | ~90 |
| `src/tecnicas/cuantizacion.py` | PTQ INT8/INT4 con bitsandbytes NF4 | ~110 |
| `src/tecnicas/poda.py` | Poda estructurada cabezas + fine-tuning | ~160 |
| `src/tecnicas/destilacion.py` | KL + CrossEntropy + temperatura T=4 | ~170 |
| `src/metricas/hardware.py` | VRAM pico, latencia CUDA events, throughput | ~150 |
| `src/metricas/precision.py` | Accuracy, F1, % retención, criterio aceptación | ~110 |
| `src/experimentos/configuraciones.py` | Definición dataclass de 22 configs | ~190 |
| `src/experimentos/pipeline.py` | Orquestador + persistencia CSV + cache | ~180 |
| `src/analisis/pareto.py` | Puntos no dominados + codo + Matplotlib + Plotly | ~240 |

### Carpeta `resultados/`

| Archivo | Propósito | Tamaño |
|---|---|---|
| `generar_referencia.py` | Genera CSV de referencia con valores literatura | ~5 KB |
| `resultados_referencia.csv` | 22 configs con valores estimados | ~3 KB |
| `frontera_pareto.png` | PNG estático (Matplotlib, 1800x1200 px) | ~155 KB |
| `frontera_pareto_interactiva.html` | HTML interactivo (Plotly) | ~4.7 MB |
| `resultados_usuario.csv` ⚡ | Se crea al ejecutar con tus métricas reales | ~4 KB |

### Carpeta `docs/`

| Archivo | Propósito | Tamaño |
|---|---|---|
| `Plan_Prueba_Concepto_Local.pdf` | Plan técnico (19 páginas) | ~289 KB |
| `Guia_Ejecucion_Paso_a_Paso.md` | Guía de instalación y ejecución | ~25 KB |
| `Estructura_Archivos.md` | Este documento | ~5 KB |

---

## Características importantes

### Archivos que NO debes modificar
- ✋ Todos los archivos `.py` dentro de `src/`
- ✋ `verificar_entorno.py`, `main.py`
- ✋ `resultados/generar_referencia.py`
- ✋ `docs/Plan_Prueba_Concepto_Local.pdf`

### Archivos que SÍ puedes modificar
- ✏️ `config.py` — para ajustar hiperparámetros (batch size, epochs, etc.)
- ✏️ `requirements.txt` — solo si necesitas versiones específicas

### Archivos/carpetas que se generan (puedes borrar y regenerar)
- 🗑️ `venv/` — puedes recrear con `python -m venv venv`
- 🗑️ `cache_modelos/estudiante_destilado/` — borrar con `python main.py limpiar-cache`
- 🗑️ `cache_dataset/` — se regenera al cargar IMDb
- 🗑️ `resultados/resultados_usuario.csv` — se regenera al ejecutar el pipeline
- 🗑️ `resultados/frontera_pareto.png` y `.html` — se regeneran con `python main.py solo-pareto`

### Archivos que debes conservar para la entrega al profesor
- ⭐ `resultados/resultados_usuario.csv` (tus resultados reales)
- ⭐ `resultados/frontera_pareto.png` (imagen para el documento)
- ⭐ `resultados/frontera_pareto_interactiva.html` (HTML para defensa)
- ⭐ `docs/Plan_Prueba_Concepto_Local.pdf` (plan técnico)
- ⭐ Todo el código en `src/` y archivos raíz

---

## Comando para listar y verificar todo de una sola vez

Pega este comando en tu terminal para verificar que tienes la estructura completa:

```bash
# Linux / macOS
echo "=== RAÍZ ===" && ls -la *.py *.txt *.md 2>/dev/null && \
echo "=== src/ ===" && find src -name "*.py" | sort && \
echo "=== resultados/ ===" && ls resultados/ && \
echo "=== docs/ ===" && ls docs/ && \
echo "=== TOTAL ===" && find . -type f -not -path "*/venv/*" -not -path "*/__pycache__/*" -not -path "*/cache_*/*" | wc -l
```

```powershell
# Windows PowerShell
Write-Host "=== RAÍZ ==="
Get-ChildItem -File *.py, *.txt, *.md | Select-Object Name
Write-Host "=== src/ ==="
Get-ChildItem -Path src -Recurse -Filter *.py | Select-Object FullName
Write-Host "=== resultados/ ==="
Get-ChildItem -Path resultados | Select-Object Name
Write-Host "=== docs/ ==="
Get-ChildItem -Path docs | Select-Object Name
Write-Host "=== TOTAL ==="
(Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch 'venv|__pycache__|cache_' }).Count
```

**Salida esperada:** 28 archivos totales (sin contar venv ni caches).

---

## Nota sobre el archivo ZIP que descargaste

Si descargaste el proyecto como ZIP, al descomprimir debes ver **una sola carpeta llamada `po-concepto-local/`** que contiene toda la estructura de arriba.

**Si al descomprimir ves múltiples carpetas sueltas** (README.md, src/, etc. sin carpeta contenedora), es porque el ZIP se creó sin carpeta raíz. En ese caso:

1. Crea una carpeta `po-concepto-local/` manualmente
2. Mueve todos los archivos descomprimidos dentro
3. Verifica la estructura con los comandos de arriba

**Ruta recomendada** (sin espacios ni acentos):
- ✅ Linux/Mac: `~/proyectos/po-concepto-local`
- ✅ Windows: `C:\proyectos\po-concepto-local`
- ❌ Evitar: `C:\Mi Carpeta\Proyecto Final\po-concepto-local` (espacios)
- ❌ Evitar: `C:\Users\Jerónimo\Proyecto\po-concepto-local` (acentos)
