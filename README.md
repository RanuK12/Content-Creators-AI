# 🤖 Virtual Content Pipeline
## Pipeline Automatizado de Producción de Arte Digital con IA

---

## 📁 Estructura del Proyecto

```
virtual_content_pipeline/
├── config/
│   ├── comfyui_workflows/     # Workflows JSON exportados
│   ├── kling_api/             # Configuración Kling AI
│   └── models/                # Modelos base (manual)
├── src/
│   ├── api_wrappers/          # Wrappers de APIs (Kling, ComfyUI)
│   ├── batch_processing/      # Generación batch
│   ├── dataset_prep/          # Preparación datasets Kohya_ss
│   ├── post_production/       # Watermark, resize, FFmpeg
│   ├── publishing/            # Publicación automatizada (placeholder)
│   └── utils/                 # Utilidades
├── workflows/
│   ├── workflow_a_character_consistency/   # LoRA + PuLID + ControlNet
│   ├── workflow_b_outfit_variation/        # Inpainting + Depth
│   └── workflow_c_motion_reel/             # 9:16 para Reels/TikTok
├── data/
│   ├── raw_images/            # Imágenes fuente
│   ├── processed_datasets/    # Datasets listos para entrenar
│   ├── training_data/         # Datos de entrenamiento
│   └── output/                # Output final
├── scripts/
│   ├── setup/                 # Scripts de setup
│   ├── batch_generate/        # Scripts de generación batch
│   ├── post_process/          # Scripts de post-producción
│   └── publish/               # Scripts de publicación
├── tests/                     # Tests
├── docs/                      # Documentación
├── requirements.txt           # Dependencias Python
├── config.yaml                # Configuración global
└── .env.example               # Variables de entorno
```

---

## 🚀 Quick Start

### 1. Setup Inicial

```bash
# Clonar y entrar
cd virtual_content_pipeline

# Setup automático
bash scripts/setup/setup.sh

# O manual:
mkdir -p data/{raw_images,processed_datasets,training_data,output}
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configurar Variables de Entorno

Edita `.env`:
```bash
KLING_ACCESS_KEY=tu_access_key
KLING_SECRET_KEY=tu_secret_key
COMFYUI_HOST=127.0.0.1
COMFYUI_PORT=8188
```

### 3. Descargar Modelos (Manual)

Coloca en `config/models/`:
- `flux1-dev-fp8.safetensors` (o flux1-dev.safetensors)
- `flux-vae.safetensors`
- `pulid_flux_v0.9.1.safetensors`
- `controlnet-union-sdxl-promax.safetensors`
- `controlnet-depth-sdxl.safetensors`
- `controlnet-openpose-sdxl.safetensors`

### 4. Preparar Dataset para Entrenar LoRA

```bash
# Coloca imágenes en data/raw_images/nova/
# Ejecuta preparación
python src/dataset_prep/kohya_prep.py
```

### 5. Generar Contenido Batch

```bash
# ComfyUI batch
python src/batch_processing/comfyui_batch.py

# Kling API video
python src/api_wrappers/kling_api.py
```

### 6. Post-producción

```bash
python src/post_production/post_prod.py
```

---

## 🧩 Workflows ComfyUI

### Workflow A: Character Consistency
**Archivo:** `workflows/workflow_a_character_consistency/workflow_api.json`

**Componentes:**
- Checkpoint: Flux.1 [dev] fp8
- LoRA: `nova_character_flux_lora.safetensors` (weight 0.7)
- PuLID: Face analysis + embedding preservation
- ControlNet: OpenPose para control de pose
- Output: 1024x1024, batch configurable

**Uso:** Genera retratos consistentes del personaje con diferentes poses.

### Workflow B: Outfit Variation
**Archivo:** `workflows/workflow_b_outfit_variation/workflow_api.json`

**Componentes:**
- Checkpoint: Flux.1 [dev] fp8
- LoRA: `nova_character_flux_lora.safetensors` (weight 0.6)
- Inpainting: Mask sobre área de ropa
- ControlNet: Depth para preservar forma corporal
- Denoise: 0.75 (balance entre consistencia y variación)

**Uso:** Cambia outfit del personaje manteniendo identidad facial.

### Workflow C: Motion Reel
**Archivo:** `workflows/workflow_c_motion_reel/workflow_api.json`

**Componentes:**
- Checkpoint: Flux.1 [dev] fp8
- LoRA: `nova_character_flux_lora.safetensors` (weight 0.7)
- PuLID: Face preservation
- Resolución: 1080x1920 (9:16)
- ControlNet: OpenPose para pose dinámica

**Uso:** Genera frames para reels/TikTok. Luego usar Kling I2V para animar.

---

## 📦 Scripts Principales

### `src/api_wrappers/kling_api.py`
Wrapper async de Kling AI API v2.0 con:
- Text-to-Video
- Image-to-Video
- Polling automático con reintentos
- Descarga automática de resultados
- Rate limiting handling

### `src/batch_processing/comfyui_batch.py`
Generador batch para ComfyUI:
- Carga workflows JSON exportados
- Itera sobre listas de prompts
- Actualiza seeds dinámicamente
- Descarga y organiza outputs

### `src/dataset_prep/kohya_prep.py`
Preparador de datasets para entrenamiento LoRA:
- Auto-detección de rostros (OpenCV Haar/DNN)
- Face crop centrado con padding
- Auto-tagging básico (placeholder para WD14)
- Generación de imágenes de regularización
- Configuración automática Kohya_ss

### `src/post_production/post_prod.py`
Post-producción automatizada:
- Watermark con texto configurable
- Resize para múltiples plataformas
- FFmpeg para procesamiento de video
- Batch processing completo

---

## ⚙️ Configuración

Edita `config.yaml` para personalizar:

```yaml
character:
  name: "Nova"                    # Nombre del personaje
  style: "photorealistic"         # Estilo visual
  lora_trigger: "nova_character"  # Trigger word para LoRA

watermark:
  text: "AI Generated • Virtual Character"
  position: "bottom_right"
  opacity: 0.7

output:
  resolutions:
    instagram_feed: [1080, 1080]
    instagram_reel: [1080, 1920]
    tiktok: [1080, 1920]
    twitter: [1200, 675]

training:
  resolution: 1024
  batch_size: 2
  max_train_steps: 2000
  learning_rate: 1e-4
```

---

## 🎓 Entrenamiento de LoRA (Kohya_ss)

1. Prepara imágenes en `data/raw_images/{character}/`
2. Ejecuta `python src/dataset_prep/kohya_prep.py`
3. Abre Kohya_ss GUI y carga la config generada en `data/processed_datasets/{character}_training/kohya_config.toml`
4. Entrena con los parámetros optimizados
5. Copia el LoRA resultante a `config/models/`

---

## 🎬 Pipeline de Video (Kling AI)

1. Genera imagen base con Workflow C (1080x1920)
2. Usa Kling I2V para animar:
   ```python
   async with KlingAPI() as api:
       result = await api.image_to_video(
           image_path="./data/output/nova_reel_0001.png",
           prompt="Gentle walking motion, subtle head turn, wind in hair",
           duration="5",
           mode="pro"
       )
   ```
3. Post-procesa con FFmpeg (watermark, resize)

---

## 📝 Notas Importantes

- **Transparencia:** Todo el contenido generado debe incluir watermark de "AI Generated"
- **Hardware recomendado:** RTX 4090 24GB VRAM mínimo para Flux.1
- **VRAM optimization:** Usa `--normalvram` o `--lowvram` en ComfyUI si es necesario
- **Modelos:** Los modelos base deben descargarse manualmente (licencias variadas)

---

## 🔧 Troubleshooting

### ComfyUI Out of Memory
```bash
python main.py --normalvram --fp8_e4m3fn --disable-xformers
```

### Kling API Rate Limited
El wrapper maneja rate limits automáticamente con backoff exponencial.

### Face Detection Fallida
Asegúrate de que las imágenes de entrada tengan rostros visibles y bien iluminados.

---

## 📄 Licencia

Este proyecto es para uso personal y educativo. Respeta las licencias de los modelos utilizados (Flux.1, ControlNet, etc.)

---

**Built with ❤️ for AI Art & Virtual Content Creation**
