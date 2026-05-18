#!/usr/bin/env python3
"""Generate the Kaggle LoRA training notebook."""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")})

def code(source):
    cells.append({"cell_type": "code", "metadata": {"trusted": True}, "source": source.split("\n"), "outputs": [], "execution_count": None})

# ============================================================
# CELL 1 - Title
# ============================================================
md("""# 🎯 LoRA Training en Kaggle - Content Creators AI
## Entrenamiento SDXL LoRA optimizado para T4 (16GB VRAM)

**Pipeline completo:**
1. Instalacion de Kohya_ss
2. Subida y preparacion del dataset (40 fotos)
3. Auto-captioning con BLIP2/WD14
4. Entrenamiento del LoRA
5. Exportacion del modelo

**Requisitos:** Activar GPU T4 x2 en Settings > Accelerator
""")

# ============================================================
# CELL 2 - Check GPU
# ============================================================
code("""# ============================================================
# CELDA 1: Verificar GPU y entorno
# ============================================================
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
print("="*60)
print("Si ves 'Tesla T4' arriba, estas listo!")
print("Si NO ves GPU, ve a Settings > Accelerator > GPU T4 x2")
""")

# ============================================================
# CELL 3 - Install dependencies
# ============================================================
code("""# ============================================================
# CELDA 2: Instalar Kohya_ss y dependencias
# ============================================================
# Esto tarda ~5 minutos, es normal

import os
os.chdir('/kaggle/working')

# Clonar kohya sd-scripts
!git clone https://github.com/kohya-ss/sd-scripts.git
os.chdir('/kaggle/working/sd-scripts')

# Instalar dependencias
!pip install -r requirements.txt -q
!pip install -e . -q
!pip install accelerate transformers safetensors -q
!pip install bitsandbytes prodigyopt lion-pytorch -q
!pip install opencv-python-headless pillow -q

# Para captioning
!pip install open_clip_torch timm -q

print("\\n" + "="*60)
print("Kohya_ss instalado correctamente!")
print("="*60)
""")

# ============================================================
# CELL 4 - Download base model
# ============================================================
code("""# ============================================================
# CELDA 3: Descargar modelo base SDXL
# ============================================================
# Usamos la version pruned/fp16 para ahorrar espacio y VRAM

import os
os.makedirs('/kaggle/working/models', exist_ok=True)
os.chdir('/kaggle/working/models')

# OPCION A: SDXL Base 1.0 (fp16 - recomendado para T4)
!wget -q --show-progress "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors" -O sdxl_base.safetensors

# Si prefieres un modelo mas liviano (RealVisXL), descomenta:
# !wget -q --show-progress "https://civitai.com/api/download/models/361593" -O realvisxl_v4.safetensors

print("\\n" + "="*60)
print("Modelo base descargado!")
print("="*60)
!ls -lh /kaggle/working/models/
""")

# ============================================================
# CELL 5 - Upload dataset
# ============================================================
md("""## 📸 Subir tu dataset

**OPCION A (recomendada):** Sube tus fotos como Dataset de Kaggle:
1. Ve a kaggle.com/datasets
2. Click "New Dataset"
3. Sube tus 40 fotos
4. Nombralo por ejemplo: `mi-personaje-fotos`
5. Luego en este notebook: Add Data > busca tu dataset

**OPCION B:** Sube directamente desde tu PC (mas lento):
Ejecuta la celda siguiente y usa el file uploader.

**IMPORTANTE sobre tus fotos:**
- Formatos aceptados: .jpg, .png, .jpeg, .webp
- Minimo 512x512 px (idealmente 1024x1024 o mas grande)
- Variedad de angulos, poses, iluminacion
- Sin otras personas en la foto (solo tu personaje)
""")

# ============================================================
# CELL 6 - Setup dataset paths
# ============================================================
code("""# ============================================================
# CELDA 4: Configurar rutas del dataset
# ============================================================

# ===== EDITA ESTO =====
# Si subiste como Dataset de Kaggle, la ruta sera algo como:
DATASET_INPUT = "/kaggle/input/mi-personaje-fotos"

# Si subiste los archivos directamente, pon la ruta donde estan
# DATASET_INPUT = "/kaggle/working/raw_images"

# Tu trigger word (nombre unico para tu personaje)
TRIGGER_WORD = "ohwx"  # <-- clasico y funcional, o usa algo como "sks" o "tu_nombre_unico"

# ===== NO TOQUES ESTO =====
DATASET_DIR = "/kaggle/working/dataset"
OUTPUT_DIR = "/kaggle/working/output"
IMG_DIR = f"{DATASET_DIR}/img/10_{TRIGGER_WORD} 1girl"
REG_DIR = f"{DATASET_DIR}/reg/1_1girl"

import os
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(REG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Verificar que las fotos existen
if os.path.exists(DATASET_INPUT):
    fotos = [f for f in os.listdir(DATASET_INPUT) if f.lower().endswith(('.png','.jpg','.jpeg','.webp'))]
    print(f"Encontradas {len(fotos)} fotos en: {DATASET_INPUT}")
    if len(fotos) < 10:
        print("ADVERTENCIA: Se recomiendan al menos 15-20 fotos para buenos resultados")
else:
    print(f"ERROR: No se encontro la carpeta {DATASET_INPUT}")
    print("Revisa la ruta o sube tu dataset primero")
""")

# ============================================================
# CELL 7 - Process images
# ============================================================
code("""# ============================================================
# CELDA 5: Procesar y preparar imagenes
# ============================================================
# Resize, crop centrado en cara, copiar a carpeta de entrenamiento

import cv2
import shutil
from PIL import Image
from pathlib import Path
import numpy as np

TARGET_SIZE = 768  # Optimo para T4 con SDXL

def process_image(src_path, dst_dir, idx, trigger_word):
    \"\"\"Procesa una imagen: resize inteligente manteniendo aspect ratio.\"\"\"
    try:
        img = Image.open(src_path).convert('RGB')
        w, h = img.size
        
        # Resize manteniendo aspect ratio (bucketing se encarga del resto)
        # Pero asegurar que el lado menor sea >= TARGET_SIZE
        if min(w, h) < TARGET_SIZE:
            scale = TARGET_SIZE / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        
        # Si es mucho mas grande que necesario, reducir
        if max(img.size) > 2048:
            scale = 2048 / max(img.size)
            img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.LANCZOS)
        
        # Guardar
        filename = f"{trigger_word}_{idx:03d}.png"
        output_path = Path(dst_dir) / filename
        img.save(output_path, 'PNG', quality=95)
        
        return str(output_path)
    except Exception as e:
        print(f"  Error con {src_path}: {e}")
        return None

# Procesar todas las fotos
input_path = Path(DATASET_INPUT)
image_files = sorted([
    f for f in input_path.iterdir() 
    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']
])

print(f"Procesando {len(image_files)} imagenes...")
print(f"Destino: {IMG_DIR}")
print(f"Resolucion target: {TARGET_SIZE}px")
print("-" * 40)

processed = 0
for idx, img_file in enumerate(image_files):
    result = process_image(str(img_file), IMG_DIR, idx, TRIGGER_WORD)
    if result:
        processed += 1
        print(f"  [{idx+1}/{len(image_files)}] OK: {img_file.name}")

print("-" * 40)
print(f"Procesadas: {processed}/{len(image_files)} imagenes")
print(f"Carpeta de entrenamiento: {IMG_DIR}")
""")

# ============================================================
# CELL 8 - Auto captioning
# ============================================================
code("""# ============================================================
# CELDA 6: Auto-Captioning (generar descripciones de cada foto)
# ============================================================
# Usa BLIP para generar captions automaticos
# Cada imagen necesita un .txt con su descripcion

from pathlib import Path

# Metodo simple pero efectivo: caption template + trigger word
# Para LoRA de personajes, captions simples funcionan mejor que BLIP complejo

IMG_PATH = Path(IMG_DIR)
image_files = sorted(IMG_PATH.glob("*.png"))

print(f"Generando captions para {len(image_files)} imagenes...")
print(f"Trigger word: {TRIGGER_WORD}")
print("-" * 40)

# Templates de caption variados (el modelo aprende mejor con variedad)
caption_templates = [
    "{tw}, portrait photo, professional lighting, sharp focus, high quality",
    "{tw}, close-up portrait, natural lighting, detailed skin texture, photorealistic",
    "{tw}, upper body shot, soft lighting, shallow depth of field",
    "{tw}, headshot, studio lighting, clean background, high resolution",
    "{tw}, photo portrait, cinematic lighting, bokeh background",
]

for idx, img_file in enumerate(image_files):
    # Usar templates variados
    template = caption_templates[idx % len(caption_templates)]
    caption = template.format(tw=TRIGGER_WORD)
    
    # Guardar caption
    caption_file = img_file.with_suffix('.txt')
    with open(caption_file, 'w') as f:
        f.write(caption)

print(f"Captions generados: {len(image_files)}")
print(f"\\nEjemplo de caption:")
example = list(IMG_PATH.glob("*.txt"))[0]
print(f"  {example.name}: {open(example).read()}")
print("\\n(Puedes editar los .txt manualmente si quieres captions mas especificos)")
""")

# ============================================================
# CELL 9 - Training config
# ============================================================
code("""# ============================================================
# CELDA 7: Crear configuracion de entrenamiento
# ============================================================
# Optimizada para T4 16GB VRAM con SDXL

import toml
from pathlib import Path

# Contar imagenes para calcular steps
num_images = len(list(Path(IMG_DIR).glob("*.png")))
num_repeats = 10  # veces que se repite cada imagen por epoch
steps_per_epoch = num_images * num_repeats
# Para 40 fotos: 40 * 10 = 400 steps por epoch
# Queremos ~4-6 epochs: 400 * 5 = 2000 steps

TOTAL_STEPS = min(2000, steps_per_epoch * 5)  # 5 epochs o max 2000

print(f"Imagenes: {num_images}")
print(f"Repeats: {num_repeats}")
print(f"Steps por epoch: {steps_per_epoch}")
print(f"Total steps: {TOTAL_STEPS}")
print(f"Epochs efectivos: {TOTAL_STEPS / steps_per_epoch:.1f}")

# Dataset config (TOML para kohya)
dataset_config = {
    "general": {
        "shuffle_caption": True,
        "caption_extension": ".txt",
        "keep_tokens": 1,
    },
    "datasets": [{
        "resolution": TARGET_SIZE,
        "batch_size": 1,
        "enable_bucket": True,
        "bucket_no_upscale": True,
        "bucket_reso_steps": 64,
        "min_bucket_reso": 512,
        "max_bucket_reso": 1024,
        "subsets": [{
            "image_dir": IMG_DIR,
            "num_repeats": num_repeats,
            "class_tokens": TRIGGER_WORD,
        }]
    }]
}

config_path = "/kaggle/working/dataset_config.toml"
with open(config_path, 'w') as f:
    toml.dump(dataset_config, f)

print(f"\\nConfig guardada en: {config_path}")
print("\\n--- Contenido ---")
print(open(config_path).read())
""")

# ============================================================
# CELL 10 - Training execution
# ============================================================
code("""# ============================================================
# CELDA 8: ENTRENAR EL LORA
# ============================================================
# Este es el paso principal. Tarda ~1.5-3 horas en T4.
# NO cierres el notebook mientras entrena!

import os
os.chdir('/kaggle/working/sd-scripts')

# Comando de entrenamiento optimizado para T4 16GB
TRAIN_CMD = f\"\"\"accelerate launch \\
  --num_cpu_threads_per_process 2 \\
  sdxl_train_network.py \\
  --pretrained_model_name_or_path="/kaggle/working/models/sdxl_base.safetensors" \\
  --dataset_config="/kaggle/working/dataset_config.toml" \\
  --output_dir="{OUTPUT_DIR}" \\
  --output_name="{TRIGGER_WORD}_lora" \\
  --network_module="networks.lora" \\
  --network_dim=16 \\
  --network_alpha=8 \\
  --learning_rate=1e-4 \\
  --unet_lr=1e-4 \\
  --text_encoder_lr=5e-5 \\
  --lr_scheduler="cosine_with_restarts" \\
  --lr_warmup_steps=100 \\
  --lr_scheduler_num_cycles=3 \\
  --max_train_steps={TOTAL_STEPS} \\
  --train_batch_size=1 \\
  --gradient_accumulation_steps=2 \\
  --mixed_precision="fp16" \\
  --save_every_n_steps=500 \\
  --save_model_as="safetensors" \\
  --seed=42 \\
  --optimizer_type="Adafactor" \\
  --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False" \\
  --max_token_length=225 \\
  --clip_skip=2 \\
  --noise_offset=0.0357 \\
  --min_snr_gamma=5 \\
  --xformers \\
  --cache_latents \\
  --cache_latents_to_disk \\
  --gradient_checkpointing \\
  --shuffle_caption \\
  --keep_tokens=1 \\
  --max_data_loader_n_workers=2 \\
  --persistent_data_loader_workers \\
  --logging_dir="{OUTPUT_DIR}/logs" \\
  --log_prefix="{TRIGGER_WORD}" \\
  --sample_every_n_steps=500 \\
  --sample_prompts="/kaggle/working/sample_prompts.txt" \\
  --sample_sampler="euler_a"
\"\"\"

# Crear sample prompts para ver progreso durante entrenamiento
sample_prompts = f\"\"\"{TRIGGER_WORD}, portrait photo, professional studio lighting, sharp focus, 8k --n low quality, blurry
{TRIGGER_WORD}, casual outfit, outdoor natural setting, golden hour --n low quality, blurry
{TRIGGER_WORD}, close-up face, detailed skin, natural makeup --n low quality, blurry
\"\"\"

with open("/kaggle/working/sample_prompts.txt", "w") as f:
    f.write(sample_prompts)

print("="*60)
print("  INICIANDO ENTRENAMIENTO")
print("="*60)
print(f"  Steps totales: {TOTAL_STEPS}")
print(f"  Optimizer: Adafactor (bajo VRAM)")
print(f"  Network dim: 16, alpha: 8")
print(f"  Resolution: {TARGET_SIZE}")
print(f"  Precision: fp16")
print(f"  Trigger word: {TRIGGER_WORD}")
print(f"  Output: {OUTPUT_DIR}/{TRIGGER_WORD}_lora.safetensors")
print("="*60)
print("\\nEsto tardara ~1.5-3 horas. No cierres el notebook!")
print("Veras el progreso: steps: X/Y loss: Z.ZZZZ")
print("="*60 + "\\n")

!{TRAIN_CMD}

print("\\n" + "="*60)
print("  ENTRENAMIENTO COMPLETADO!")
print("="*60)
""")

# ============================================================
# CELL 11 - Check results
# ============================================================
code("""# ============================================================
# CELDA 9: Verificar resultados
# ============================================================

import os
from pathlib import Path

output_path = Path(OUTPUT_DIR)

print("Archivos generados:")
print("-" * 40)
for f in sorted(output_path.rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name:40s} ({size_mb:.1f} MB)")

# Buscar el LoRA final
lora_files = list(output_path.glob("*.safetensors"))
if lora_files:
    print(f"\\n{'='*60}")
    print(f"  TU LORA ESTA LISTO!")
    print(f"  Archivo: {lora_files[-1].name}")
    print(f"  Tamano: {lora_files[-1].stat().st_size / (1024*1024):.1f} MB")
    print(f"{'='*60}")
else:
    print("\\nERROR: No se encontro archivo .safetensors")
    print("Revisa los logs de entrenamiento arriba")
""")

# ============================================================
# CELL 12 - Download / Export
# ============================================================
code("""# ============================================================
# CELDA 10: Descargar tu LoRA
# ============================================================

from pathlib import Path
from IPython.display import FileLink, display

output_path = Path(OUTPUT_DIR)
lora_files = sorted(output_path.glob("*.safetensors"))

if lora_files:
    print("Tus archivos LoRA listos para descargar:")
    print("-" * 40)
    for lf in lora_files:
        # Copiar a /kaggle/working para que sea descargable
        import shutil
        dst = Path("/kaggle/working") / lf.name
        shutil.copy2(lf, dst)
        print(f"  {lf.name}")
        display(FileLink(str(dst), result_html_prefix="  Descargar: "))
    
    print("\\n" + "="*60)
    print("SIGUIENTE PASO:")
    print("1. Descarga el archivo .safetensors")
    print("2. Subelo a ComfyUI (carpeta models/loras/)")
    print("3. O subelo a CivitAI para compartirlo")
    print("4. Usa el trigger word: " + TRIGGER_WORD)
    print("="*60)
else:
    print("No hay archivos para descargar. Revisa el entrenamiento.")
""")

# ============================================================
# CELL 13 - Test generation (bonus)
# ============================================================
md("""## 🎨 (BONUS) Generar imagenes de prueba con tu LoRA

Si quieres probar tu LoRA aqui mismo en Kaggle, ejecuta la siguiente celda.
Usa la libreria `diffusers` para generar unas imagenes rapidas de prueba.
""")

code("""# ============================================================
# CELDA 11 (OPCIONAL): Probar tu LoRA generando imagenes
# ============================================================
# Esto usa diffusers (mas lento que ComfyUI pero funciona aqui)

from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

# Encontrar el LoRA
output_path = Path(OUTPUT_DIR)
lora_files = sorted(output_path.glob("*.safetensors"))
LORA_PATH = str(lora_files[-1]) if lora_files else None

if LORA_PATH is None:
    print("ERROR: No se encontro LoRA. Entrena primero.")
else:
    print(f"Cargando modelo + LoRA: {Path(LORA_PATH).name}")
    
    # Cargar pipeline
    pipe = StableDiffusionXLPipeline.from_single_file(
        "/kaggle/working/models/sdxl_base.safetensors",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LORA_PATH)
    pipe.enable_xformers_memory_efficient_attention()
    
    # Generar imagenes de prueba
    test_prompts = [
        f"{TRIGGER_WORD}, portrait photo, professional lighting, sharp focus, 8k, masterpiece",
        f"{TRIGGER_WORD}, casual outfit, outdoor setting, golden hour, photorealistic",
        f"{TRIGGER_WORD}, close-up face, studio lighting, clean background",
    ]
    
    print("\\nGenerando imagenes de prueba...")
    for idx, prompt in enumerate(test_prompts):
        image = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, deformed, ugly, bad anatomy",
            num_inference_steps=25,
            guidance_scale=7.0,
            width=768,
            height=768,
        ).images[0]
        
        save_path = f"/kaggle/working/test_{idx:02d}.png"
        image.save(save_path)
        print(f"  Guardada: test_{idx:02d}.png")
    
    # Mostrar imagenes
    from IPython.display import display, Image as IPImage
    for idx in range(len(test_prompts)):
        print(f"\\nPrompt: {test_prompts[idx]}")
        display(IPImage(filename=f"/kaggle/working/test_{idx:02d}.png", width=400))
    
    del pipe
    torch.cuda.empty_cache()
    print("\\nListo! Si las imagenes se parecen a tu personaje, el LoRA funciono.")
""")

# ============================================================
# Build notebook JSON
# ============================================================
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells
}

output_path = "/kaggle/working/kaggle_lora_training.ipynb"
# For local generation:
output_path = "/projects/sandbox/Content-Creators-AI/notebooks/kaggle_lora_training.ipynb"

import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generado: {output_path}")
