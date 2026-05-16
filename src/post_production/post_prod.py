"""
Post-producción automatizada
Watermark, resize para plataformas, metadata, FFmpeg.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2
from loguru import logger
import yaml

class PostProduction:
    def __init__(self, config_path: str = "./config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.watermark_config = self.config["watermark"]
        self.output_config = self.config["output"]

    def add_watermark(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        custom_text: Optional[str] = None
    ) -> str:
        """Añade watermark a una imagen."""
        image = Image.open(image_path)

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Crear capa de watermark
        watermark = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark)

        # Texto
        text = custom_text or self.watermark_config["text"]
        font_size = self.watermark_config["font_size"]

        # Intentar cargar fuente, fallback a default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Calcular posición
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        position = self.watermark_config["position"]
        padding = 20

        if position == "bottom_right":
            x = image.width - text_width - padding
            y = image.height - text_height - padding
        elif position == "bottom_left":
            x = padding
            y = image.height - text_height - padding
        elif position == "top_right":
            x = image.width - text_width - padding
            y = padding
        elif position == "top_left":
            x = padding
            y = padding
        else:
            x = (image.width - text_width) // 2
            y = (image.height - text_height) // 2

        # Dibujar texto con sombra
        opacity = int(255 * self.watermark_config["opacity"])
        draw.text((x+2, y+2), text, font=font, fill=(0, 0, 0, opacity//2))
        draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity))

        # Combinar
        result = Image.alpha_composite(image, watermark)

        if output_path is None:
            output_path = str(Path(image_path).parent / f"wm_{Path(image_path).name}")

        result.convert("RGB").save(output_path, "PNG")
        logger.success(f"Watermarked: {output_path}")
        return output_path

    def resize_for_platform(
        self,
        image_path: str,
        platform: str,
        output_dir: Optional[str] = None
    ) -> str:
        """Resize imagen para una plataforma específica."""
        image = Image.open(image_path)

        resolution = self.output_config["resolutions"].get(platform)
        if not resolution:
            raise ValueError(f"Plataforma no soportada: {platform}")

        width, height = resolution

        # Resize manteniendo aspect ratio con crop/fill
        img_ratio = image.width / image.height
        target_ratio = width / height

        if img_ratio > target_ratio:
            # Imagen más ancha, crop horizontal
            new_height = height
            new_width = int(new_height * img_ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            left = (new_width - width) // 2
            image = image.crop((left, 0, left + width, height))
        else:
            # Imagen más alta, crop vertical
            new_width = width
            new_height = int(new_width / img_ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            top = (new_height - height) // 2
            image = image.crop((0, top, width, top + height))

        if output_dir is None:
            output_dir = Path(image_path).parent / platform
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{platform}_{Path(image_path).name}"

        image.save(output_path, "JPEG", quality=95)
        logger.success(f"Resized for {platform}: {output_path}")
        return str(output_path)

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        watermark_text: Optional[str] = None,
        platform: str = "instagram_reel"
    ) -> str:
        """Post-producción de video con FFmpeg."""
        resolution = self.output_config["resolutions"].get(platform, [1080, 1920])
        width, height = resolution

        text = watermark_text or self.watermark_config["text"]

        # Comando FFmpeg
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,drawtext=text='{text}':fontsize=24:fontcolor=white@0.7:x=w-tw-20:y=h-th-20",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p"
        ]

        if output_path is None:
            output_path = str(Path(video_path).parent / f"processed_{Path(video_path).name}")

        cmd.append(output_path)

        logger.info(f"Procesando video: {video_path}")
        subprocess.run(cmd, check=True)
        logger.success(f"Video procesado: {output_path}")
        return output_path

    def batch_process(
        self,
        input_dir: str,
        output_dir: str,
        platforms: List[str] = None,
        add_watermark: bool = True
    ) -> Dict[str, List[str]]:
        """Procesa batch de imágenes para múltiples plataformas."""
        if platforms is None:
            platforms = ["instagram_feed", "instagram_reel", "twitter"]

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {platform: [] for platform in platforms}

        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))

        for img_path in image_files:
            # Añadir watermark
            if add_watermark:
                wm_path = self.add_watermark(
                    str(img_path),
                    str(output_dir / f"wm_{img_path.name}")
                )
            else:
                wm_path = str(img_path)

            # Resize para cada plataforma
            for platform in platforms:
                try:
                    resized = self.resize_for_platform(wm_path, platform, str(output_dir / platform))
                    results[platform].append(resized)
                except Exception as e:
                    logger.error(f"Error resize {platform} para {img_path}: {e}")

        return results

# Ejemplo de uso
def main():
    post = PostProduction()

    # Procesar imagen individual
    post.add_watermark("./data/output/test.png")

    # Batch process
    results = post.batch_process(
        input_dir="./data/output/nova_batch_001",
        output_dir="./data/output/nova_processed",
        platforms=["instagram_reel", "instagram_feed", "twitter"]
    )

    print(f"Procesadas {sum(len(v) for v in results.values())} imágenes")

if __name__ == "__main__":
    main()
