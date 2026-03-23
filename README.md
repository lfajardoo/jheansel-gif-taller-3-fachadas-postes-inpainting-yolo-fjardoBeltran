# Detección de fachadas y eliminación de postes con YOLOv8 + FastAPI

Este proyecto extiende un trabajo previo de detección de casas/fachadas para incorporar la detección de **postes** y su posterior **eliminación automática** en imágenes mediante **inpainting**.

El flujo general del proyecto es:

1. Entrenamiento de un modelo **YOLOv8** con dos clases: `casa` y `poste`.
2. Inferencia sobre imágenes nuevas.
3. Generación de máscaras para los objetos detectados de la clase `poste`.
4. Aplicación de **inpainting** con OpenCV para eliminar visualmente los postes de la imagen.
5. Exposición del pipeline como servicio HTTP usando **FastAPI**.

El proyecto fue construido a partir del notebook experimental `Taller_3.ipynb`, y reorganizado en una estructura reproducible basada en scripts.

---

## Objetivo

Desarrollar un pipeline reproducible que permita:

- detectar fachadas/casas y postes en imágenes,
- generar máscaras para los postes detectados,
- eliminar postes de las imágenes usando inpainting,
- exponer el proceso mediante una API con FastAPI,
- conservar evidencias del entrenamiento, inferencia y resultados visuales.

---

## Estructura del repositorio

```text
src/
├── train_yolo.py
├── inferencia.py
├── utils.py
└── api.py

models/
└── casas_postes_v1/
    ├── weights/
    │   ├── best.pt
    │   ├── last.pt
    │   └── best.onnx
    └── archivos de métricas y visualización del entrenamiento

requirements.txt
data.yaml
README.md

---- 

# Detección de fachadas y eliminación de postes con YOLOv8 + FastAPI

Este proyecto extiende un trabajo previo de detección de casas/fachadas para incorporar la detección de **postes** y su posterior **eliminación automática** en imágenes mediante **inpainting**.

El flujo general del proyecto es:

1. Entrenamiento de un modelo **YOLOv8** con dos clases: `casa` y `poste`.
2. Inferencia sobre imágenes nuevas.
3. Generación de máscaras para los objetos detectados de la clase `poste`.
4. Aplicación de **inpainting** con OpenCV para eliminar visualmente los postes de la imagen.
5. Exposición del pipeline como servicio HTTP usando **FastAPI**.

El proyecto fue construido a partir del notebook experimental `Taller_3.ipynb` y reorganizado en una estructura reproducible basada en scripts.

---

## Objetivo

Desarrollar un pipeline reproducible que permita:

- detectar fachadas/casas y postes en imágenes,
- generar máscaras para los postes detectados,
- eliminar postes de las imágenes usando inpainting,
- exponer el proceso mediante una API con FastAPI,
- conservar evidencias del entrenamiento, inferencia y resultados visuales.

---

## Estructura del repositorio

```text
src/
├── train_yolo.py
├── inferencia.py
├── utils.py
└── api.py

models/
└── casas_postes_v1/
    ├── weights/
    │   ├── best.pt
    │   ├── last.pt
    │   └── best.onnx
    └── archivos de métricas y visualización del entrenamiento

data/
└── casas.yolov8/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    ├── data.yaml
    └── README.roboflow.txt

notebooks/
└── Taller_3.ipynb

requirements.txt
data.yaml
README.md
.gitignore
```

### Estructura generada durante la ejecución

Las siguientes carpetas se crean cuando se ejecuta inferencia por script o por API:

```text
outputs/
├── predicciones/
├── mascaras_postes/
├── resultados_inpainting/
└── api_uploads/
```

---

## Requisitos

Se recomienda usar:

- **Python 3.10** o **Python 3.11**
- `pip`
- entorno virtual (`venv`)
- sistema operativo Linux, macOS o Windows con Python 3 instalado

### Requisitos de hardware

- CPU suficiente para inferencia y pruebas básicas
- GPU recomendada para entrenamiento más rápido
- memoria según el tamaño del dataset y del modelo

Para este proyecto se recomienda **Python 3.10 o 3.11** por compatibilidad con `ultralytics`, `opencv`, `numpy`, `fastapi` y `uvicorn`.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

### 2. Crear y activar un entorno virtual

#### Linux / macOS con Python 3.10

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

#### Linux / macOS con Python 3.11

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Actualizar `pip`

```bash
pip install --upgrade pip
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Dependencias

El archivo `requirements.txt` contiene las dependencias necesarias para:

- entrenamiento e inferencia con YOLOv8,
- lectura y escritura de imágenes,
- generación de máscaras,
- inpainting,
- exportación a ONNX,
- despliegue de la API con FastAPI.

Contenido recomendado de `requirements.txt`:

```txt
ultralytics>=8.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
pillow>=9.5.0
pyyaml>=6.0
onnx>=1.14.0
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.9
```

---

## Dataset

El dataset utilizado proviene de una exportación en formato **YOLOv8** y debe estar organizado así:

```text
data/casas.yolov8/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
└── README.roboflow.txt
```

Las clases del dataset son:

- `casa`
- `poste`

El archivo `data.yaml` de la raíz del proyecto centraliza la configuración del dataset.

Contenido esperado de `data.yaml`:

```yaml
path: data/casas.yolov8
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: casa
  1: poste
```

---

## Archivos principales de `src/`

### `src/train_yolo.py`

Script de entrenamiento del modelo YOLOv8.

Permite:

- cargar un modelo base,
- entrenar con el dataset definido en `data.yaml`,
- guardar resultados y pesos en `models/`.

### `src/inferencia.py`

Script de inferencia sobre imágenes nuevas.

Permite:

- cargar pesos entrenados,
- detectar objetos en una imagen o carpeta,
- generar máscaras para la clase `poste`,
- aplicar inpainting,
- guardar imágenes de predicción, máscaras y resultados finales.

### `src/utils.py`

Funciones auxiliares para:

- manejo de rutas,
- lectura y guardado de imágenes,
- dibujo de bounding boxes,
- generación de máscaras,
- dilatación de máscara,
- aplicación de inpainting con OpenCV.

### `src/api.py`

Servicio web con FastAPI.

Permite:

- recibir imágenes mediante HTTP,
- ejecutar detección,
- generar máscaras,
- aplicar inpainting,
- devolver rutas de salida o el archivo resultante.

---

## Entrenamiento del modelo

Para entrenar el modelo desde cero usando el dataset configurado en `data.yaml`:

```bash
python src/train_yolo.py --data data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16
```

### Parámetros principales

- `--model`: modelo base de YOLO
- `--data`: ruta al archivo `data.yaml`
- `--epochs`: número de épocas
- `--imgsz`: tamaño de imagen
- `--batch`: tamaño de lote
- `--project`: carpeta de salida
- `--name`: nombre del experimento
- `--device`: dispositivo de ejecución (`cpu`, `0`, etc.)

### Ejemplo con nombre de experimento explícito

```bash
python src/train_yolo.py --data data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --project models --name casas_postes_v1
```

Los resultados del entrenamiento se guardan en una carpeta como:

```text
models/casas_postes_v1/
```

Incluyendo:

- pesos del modelo,
- curvas de precisión y recall,
- matrices de confusión,
- imágenes de lotes de entrenamiento y validación,
- archivo `results.csv`.

---

## Uso de pesos ya entrenados

Si ya cuentas con pesos generados previamente, puedes usar directamente:

```text
models/casas_postes_v1/weights/best.pt
```

También puede existir una exportación ONNX en:

```text
models/casas_postes_v1/weights/best.onnx
```

---

## Inferencia por script

### Solo detección

```bash
python src/inferencia.py --weights models/casas_postes_v1/weights/best.pt --source data/casas.yolov8/test/images --modo detectar
```

### Generación de máscaras

```bash
python src/inferencia.py --weights models/casas_postes_v1/weights/best.pt --source data/casas.yolov8/test/images --modo mascaras
```

### Inpainting

```bash
python src/inferencia.py --weights models/casas_postes_v1/weights/best.pt --source data/casas.yolov8/test/images --modo inpaint
```

### Pipeline completo

```bash
python src/inferencia.py --weights models/casas_postes_v1/weights/best.pt --source data/casas.yolov8/test/images --modo pipeline
```

El modo `pipeline` genera detección, máscara e inpainting en una sola ejecución.

---

## Parámetros útiles de inferencia

El script `src/inferencia.py` permite modificar varios parámetros:

- `--weights`: ruta al archivo `.pt`
- `--source`: imagen o carpeta de entrada
- `--modo`: `detectar`, `mascaras`, `inpaint` o `pipeline`
- `--conf`: umbral de confianza
- `--target-class`: clase a eliminar, por defecto `poste`
- `--mask-kernel`: tamaño del kernel para dilatar la máscara
- `--mask-iterations`: iteraciones de dilatación
- `--inpaint-radius`: radio del inpainting
- `--inpaint-method`: `telea` o `ns`

### Ejemplo completo

```bash
python src/inferencia.py \
  --weights models/casas_postes_v1/weights/best.pt \
  --source data/casas.yolov8/test/images \
  --modo pipeline \
  --conf 0.25 \
  --target-class poste \
  --mask-kernel 7 \
  --mask-iterations 1 \
  --inpaint-radius 3 \
  --inpaint-method telea
```

---

## API con FastAPI

El proyecto también puede ejecutarse como servicio web usando FastAPI, sin cambiar la estructura actual del repositorio.

### Archivo requerido

Debes añadir:

```text
src/api.py
```

### Ejecutar la API

Desde la raíz del proyecto:

```bash
uvicorn src.api:app --reload
```

### Documentación interactiva

Una vez levantado el servicio, abre:

```text
http://127.0.0.1:8000/docs
```

### Endpoints principales

- `GET /`
- `GET /health`
- `POST /predict`
- `POST /mask`
- `POST /inpaint`
- `POST /inpaint/file`

### Comportamiento esperado

#### `POST /predict`

- recibe una imagen,
- ejecuta detección,
- guarda una imagen con bounding boxes,
- devuelve detecciones y ruta del archivo generado.

#### `POST /mask`

- recibe una imagen,
- detecta postes,
- genera la máscara correspondiente,
- devuelve la ruta de la máscara.

#### `POST /inpaint`

- recibe una imagen,
- detecta postes,
- genera máscara,
- aplica inpainting,
- devuelve la ruta del resultado final.

#### `POST /inpaint/file`

- recibe una imagen,
- ejecuta el pipeline de inpainting,
- devuelve directamente la imagen procesada.

---

## Carpetas de salida

Durante la inferencia por script o por API se generan salidas como:

```text
outputs/predicciones/
outputs/mascaras_postes/
outputs/resultados_inpainting/
outputs/api_uploads/
```

Estas carpetas pueden ignorarse en Git porque son regenerables.

---

## Ejemplos visuales

Se recomienda incluir una pequeña muestra con tres tipos de imágenes:

- imagen original,
- máscara generada,
- imagen final tras inpainting.

Por ejemplo:

```text
examples/
├── input/
├── masks/
└── output/
```

Estas imágenes pueden usarse en este README para mostrar el comportamiento del pipeline.

---

## Resultados esperados

El proyecto genera tres tipos principales de resultados:

### 1. Predicciones

Imágenes con cajas delimitadoras sobre casas y postes detectados.

### 2. Máscaras

Imágenes binarias donde los postes detectados aparecen en blanco para ser usados por el proceso de inpainting.

### 3. Inpainting

Imágenes finales donde se intenta eliminar visualmente el poste detectado.

---

## Reproducibilidad

Este repositorio permite reproducir el experimento porque reúne:

- scripts de entrenamiento,
- scripts de inferencia,
- utilidades de máscara e inpainting,
- descriptor del dataset (`data.yaml`),
- pesos entrenados,
- servicio HTTP para probar el modelo sin escribir código adicional,
- notebook original del experimento como referencia.

Para repetir el flujo completo:

1. instalar dependencias,
2. verificar la estructura del dataset,
3. entrenar o reutilizar `best.pt`,
4. ejecutar inferencia por script o mediante FastAPI.

---

## Verificación recomendada

Después de crear los archivos de `src/`, se recomienda probar:

- `train_yolo.py`
- `inferencia.py`
- `utils.py`
- `api.py`

especialmente en un entorno con **Python 3.10 o 3.11**.

---

## Limitaciones actuales

- La máscara se genera a partir de **bounding boxes**, por lo que la eliminación del poste puede afectar zonas cercanas.
- El método de inpainting usado es el de **OpenCV**, adecuado para una primera versión reproducible, pero no necesariamente el más avanzado visualmente.
- La calidad del resultado final depende de:
  - la calidad de las detecciones,
  - la precisión de la máscara,
  - el tamaño y ubicación del poste,
  - el fondo de la imagen.

---

## Trabajo futuro

Posibles mejoras:

- usar segmentación en lugar de solo bounding boxes,
- incorporar modelos más avanzados de inpainting,
- agregar script separado de evaluación,
- generar comparativas automáticas lado a lado,
- contenedorización con Docker,
- despliegue en nube de la API.

---

## Relación con el proyecto base

Este proyecto toma como referencia el repositorio base de detección de casas/fachadas y lo amplía para soportar:

- detección de la clase `poste`,
- generación automática de máscaras,
- eliminación visual de postes usando inpainting,
- exposición del pipeline como API con FastAPI.

La nueva versión conserva la idea del proyecto base de trabajar con:

- `src/`
- `data.yaml`
- `requirements.txt`
- documentación en `README.md`

---

## Créditos

Este repositorio fué construido con ayuda de LLMs comerciales con el objetivo de generar el taller 3, Aplicaciones de aprendizajes de máquina 2026-1.