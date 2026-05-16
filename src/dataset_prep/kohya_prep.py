"""
Dataset Preparation para Kohya_ss
Auto-crop, auto-tag, face crop, regularization images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
from loguru import logger
import subprocess

class DatasetPreparator:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        target_size: int = 1024,
        face_crop: bool = True,
        auto_tag: bool = True
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.face_crop = face_crop
        self.auto_tag = auto_tag

        # Crear estructura de salida
        self.img_dir = self.output_dir / "img"
        self.reg_dir = self.output_dir / "reg"
        self.log_dir = self.output_dir / "logs"

        for d in [self.img_dir, self.reg_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def detect_faces(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """Detecta rostros usando OpenCV DNN."""
        image = cv2.imread(str(image_path))
        if image is None:
            return []

        # Usar detector DNN de OpenCV (más preciso que Haar)
        model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "deploy.prototxt"

        # Fallback a Haar si no está el modelo DNN
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        return [(x, y, w, h) for x, y, w, h in faces]

    def crop_face_centered(
        self,
        image: Image.Image,
        face_box: Tuple[int, int, int, int],
        target_size: int = 1024
    ) -> Image.Image:
        """Recorta centrado en el rostro con padding."""
        x, y, w, h = face_box

        # Centro del rostro
        cx, cy = x + w // 2, y + h // 2

        # Calcular crop cuadrado
        crop_size = max(w, h) * 1.5  # 1.5x el tamaño del rostro
        crop_size = min(crop_size, min(image.size))

        left = max(0, cx - int(crop_size // 2))
        top = max(0, cy - int(crop_size // 2))
        right = min(image.width, left + int(crop_size))
        bottom = min(image.height, top + int(crop_size))

        # Ajustar si se sale de los límites
        if right - left < crop_size:
            left = max(0, right - int(crop_size))
        if bottom - top < crop_size:
            top = max(0, bottom - int(crop_size))

        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((target_size, target_size), Image.LANCZOS)

    def process_image(self, image_path: str, trigger_word: str = "nova_character") -> Optional[str]:
        """Procesa una imagen: crop, resize, tag."""
        try:
            image = Image.open(image_path)

            # Convertir a RGB si es necesario
            if image.mode != "RGB":
                image = image.convert("RGB")

            filename = Path(image_path).stem

            if self.face_crop:
                faces = self.detect_faces(image_path)
                if faces:
                    # Usar el rostro más grande
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    image = self.crop_face_centered(image, largest_face, self.target_size)
                else:
                    # Fallback: resize centrado
                    image = image.resize((self.target_size, self.target_size), Image.LANCZOS)
            else:
                image = image.resize((self.target_size, self.target_size), Image.LANCZOS)

            # Guardar imagen procesada
            output_name = f"{trigger_word}_{filename}.png"
            output_path = self.img_dir / output_name
            image.save(output_path, "PNG")

            # Generar caption/tag
            if self.auto_tag:
                caption = self.generate_caption(image_path, trigger_word)
                caption_path = self.img_dir / f"{trigger_word}_{filename}.txt"
                with open(caption_path, "w") as f:
                    f.write(caption)

            logger.success(f"Procesado: {output_name}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error procesando {image_path}: {e}")
            return None

    def generate_caption(self, image_path: str, trigger_word: str) -> str:
        """Genera caption básico (placeholder para WD14 tagger)."""
        # En producción, usar WD14 tagger o BLIP
        return f"{trigger_word}, 1girl, solo, looking at viewer, realistic, photorealistic, 8k, detailed skin, professional photography"

    def create_regularization_images(
        self,
        prompt: str = "1girl, solo, realistic, photorealistic",
        num_images: int = 100,
        model: str = "flux1-dev.safetensors"
    ):
        """Genera imágenes de regularización usando ComfyUI o SD webui."""
        logger.info(f"Generando {num_images} imágenes de regularización...")

        # Placeholder: en producción, usar API de ComfyUI para generar
        # imágenes genéricas del mismo estilo pero sin el personaje específico

        reg_prompt = f"{prompt}, generic woman, no specific identity"

        for i in range(num_images):
            reg_filename = f"reg_{i:04d}.png"
            reg_path = self.reg_dir / reg_filename

            # Aquí iría la llamada a ComfyUI API para generar
            # Por ahora, copiamos una imagen placeholder
            logger.info(f"Regularization {i+1}/{num_images}: {reg_filename}")

    def prepare_dataset(
        self,
        trigger_word: str = "nova_character",
        reg_prompt: str = "1girl, solo, realistic, photorealistic",
        num_reg_images: int = 100
    ) -> Dict[str, str]:
        """Pipeline completo de preparación de dataset."""
        logger.info("Iniciando preparación de dataset...")

        # Procesar imágenes de entrenamiento
        input_files = list(self.input_dir.glob("*.png")) +                      list(self.input_dir.glob("*.jpg")) +                      list(self.input_dir.glob("*.jpeg"))

        processed = []
        for file in input_files:
            result = self.process_image(str(file), trigger_word)
            if result:
                processed.append(result)

        # Generar imágenes de regularización
        self.create_regularization_images(reg_prompt, num_reg_images)

        # Crear archivo de configuración para Kohya_ss
        config = self.generate_kohya_config(trigger_word)

        logger.success(f"Dataset preparado: {len(processed)} imágenes de entrenamiento")
        return {
            "train_dir": str(self.img_dir),
            "reg_dir": str(self.reg_dir),
            "config": config,
            "num_train_images": len(processed)
        }

    def generate_kohya_config(self, trigger_word: str) -> str:
        """Genera configuración para Kohya_ss GUI/CLI."""
        config = f"""# Kohya_ss Config - Generado automáticamente
pretrained_model_name_or_path = "./models/flux1-dev.safetensors"
output_dir = "./output/{trigger_word}_lora"
logging_dir = "./logs"

# Dataset
train_data_dir = "{self.img_dir}"
reg_data_dir = "{self.reg_dir}"
resolution = 1024
batch_size = 2
max_train_steps = 2000
save_every_n_epochs = 1

# Learning
learning_rate = 1e-4
lr_scheduler = "cosine_with_restarts"
optimizer_type = "AdamW8bit"

# LoRA
network_module = "networks.lora_flux"
network_dim = 32
network_alpha = 16

# Trigger word
keep_tokens = 1
"""
        config_path = self.output_dir / "kohya_config.toml"
        with open(config_path, "w") as f:
            f.write(config)
        return str(config_path)

# Ejemplo de uso
def main():
    prep = DatasetPreparator(
        input_dir="./data/raw_images/nova",
        output_dir="./data/processed_datasets/nova_training",
        target_size=1024,
        face_crop=True,
        auto_tag=True
    )

    result = prep.prepare_dataset(
        trigger_word="nova_character",
        num_reg_images=50
    )

    print(f"Dataset listo en: {result['train_dir']}")
    print(f"Config Kohya: {result['config']}")

if __name__ == "__main__":
    main()
