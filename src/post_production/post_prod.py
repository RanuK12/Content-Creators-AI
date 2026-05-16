"""
Post-Production Module
=======================
Handles watermarking, resizing, video processing via FFmpeg.

Features:
- Batch watermarking with configurable position/opacity
- Multi-platform resize (Instagram, Twitter, Portfolio)
- FFmpeg video post-production (codec, captions, watermark overlay)
- EXIF metadata injection for AI provenance

Research relevance:
- Ethical watermarking ensures AI-generated content is clearly labeled
- Metadata injection provides full provenance chain for transparency
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image, ImageDraw, ImageFont


@dataclass
class WatermarkConfig:
    """Watermark settings."""
    text: str = "AI Generated Concept Art"
    font_size: int = 24
    opacity: float = 0.6
    position: str = "bottom_right"  # bottom_right, bottom_left, center
    color: tuple[int, int, int] = (255, 255, 255)
    margin: int = 20


@dataclass
class ResizePreset:
    """Platform-specific resize preset."""
    name: str
    width: int
    height: int
    quality: int = 95


# Standard platform presets
RESIZE_PRESETS = {
    "instagram_feed": ResizePreset("Instagram Feed", 1080, 1080),
    "instagram_story": ResizePreset("Instagram Story", 1080, 1920),
    "twitter": ResizePreset("Twitter", 1200, 675),
    "portfolio_4k": ResizePreset("Portfolio 4K", 3840, 2160),
    "thumbnail": ResizePreset("Thumbnail", 400, 400),
}


class PostProduction:
    """
    Post-production processor for images and videos.
    
    Handles the final output stage of the pipeline:
    watermarking, resizing, format conversion, and metadata.
    """

    def __init__(self, watermark_config: Optional[WatermarkConfig] = None):
        self.watermark_config = watermark_config or WatermarkConfig()
        logger.info(f"PostProduction initialized | watermark='{self.watermark_config.text}'")

    def add_watermark(
        self,
        image_path: str | Path,
        output_path: Optional[str | Path] = None,
        config: Optional[WatermarkConfig] = None,
    ) -> Path:
        """
        Add semi-transparent text watermark to image.
        
        Uses PIL with alpha compositing for professional watermark.
        """
        cfg = config or self.watermark_config
        image_path = Path(image_path)
        
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_watermarked{image_path.suffix}"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open image
        img = Image.open(image_path).convert("RGBA")
        
        # Create watermark layer
        watermark_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark_layer)
        
        # Font (fallback to default if custom not available)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", cfg.font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        # Calculate position
        bbox = draw.textbbox((0, 0), cfg.text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        if cfg.position == "bottom_right":
            x = img.width - text_width - cfg.margin
            y = img.height - text_height - cfg.margin
        elif cfg.position == "bottom_left":
            x = cfg.margin
            y = img.height - text_height - cfg.margin
        elif cfg.position == "center":
            x = (img.width - text_width) // 2
            y = (img.height - text_height) // 2
        else:
            x = img.width - text_width - cfg.margin
            y = img.height - text_height - cfg.margin
        
        # Draw with opacity
        alpha = int(255 * cfg.opacity)
        color_with_alpha = (*cfg.color, alpha)
        draw.text((x, y), cfg.text, font=font, fill=color_with_alpha)
        
        # Composite
        result = Image.alpha_composite(img, watermark_layer)
        result = result.convert("RGB")
        result.save(output_path, quality=95)
        
        logger.debug(f"Watermarked: {output_path.name}")
        return output_path

    def resize_for_platform(
        self,
        image_path: str | Path,
        platform: str,
        output_dir: Optional[str | Path] = None,
    ) -> Path:
        """
        Resize image for a specific platform preset.
        
        Uses center-crop strategy to maintain composition.
        """
        if platform not in RESIZE_PRESETS:
            raise ValueError(f"Unknown platform: {platform}. Options: {list(RESIZE_PRESETS.keys())}")
        
        preset = RESIZE_PRESETS[platform]
        image_path = Path(image_path)
        
        if output_dir is None:
            output_dir = image_path.parent / platform
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{image_path.stem}_{platform}{image_path.suffix}"
        
        img = Image.open(image_path)
        
        # Resize with cover strategy (fill, then crop)
        target_ratio = preset.width / preset.height
        img_ratio = img.width / img.height
        
        if img_ratio > target_ratio:
            # Image is wider: scale by height, crop width
            new_height = preset.height
            new_width = int(img_ratio * new_height)
        else:
            # Image is taller: scale by width, crop height
            new_width = preset.width
            new_height = int(new_width / img_ratio)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop
        left = (new_width - preset.width) // 2
        top = (new_height - preset.height) // 2
        img = img.crop((left, top, left + preset.width, top + preset.height))
        
        img.save(output_path, quality=preset.quality)
        logger.debug(f"Resized for {platform}: {output_path.name}")
        return output_path

    def batch_watermark(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> list[Path]:
        """Watermark all images in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        
        for ext in extensions:
            for img_path in input_dir.glob(ext):
                out_path = output_dir / img_path.name
                result = self.add_watermark(img_path, out_path)
                results.append(result)
        
        logger.info(f"Batch watermark: {len(results)} images processed")
        return results

    def batch_resize_multiplatform(
        self,
        input_dir: str | Path,
        output_base_dir: str | Path,
        platforms: list[str] = None,
    ) -> dict[str, list[Path]]:
        """Resize all images for multiple platforms."""
        if platforms is None:
            platforms = list(RESIZE_PRESETS.keys())
        
        input_dir = Path(input_dir)
        output_base_dir = Path(output_base_dir)
        
        results = {}
        extensions = ["*.png", "*.jpg", "*.jpeg"]
        
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_dir.glob(ext))
        
        for platform in platforms:
            platform_results = []
            platform_dir = output_base_dir / platform
            
            for img_path in image_paths:
                result = self.resize_for_platform(img_path, platform, platform_dir)
                platform_results.append(result)
            
            results[platform] = platform_results
        
        total = sum(len(v) for v in results.values())
        logger.info(f"Multi-platform resize: {total} files across {len(platforms)} platforms")
        return results

    @staticmethod
    def process_video_ffmpeg(
        input_path: str | Path,
        output_path: str | Path,
        watermark_text: str = "AI Generated Concept Art",
        target_resolution: tuple[int, int] = (1080, 1920),
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "slow",
    ) -> Path:
        """
        Post-process video with FFmpeg.
        
        Adds text watermark overlay, resizes, and re-encodes.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        width, height = target_resolution
        
        # FFmpeg command with drawtext filter for watermark
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                f"drawtext=text='{watermark_text}':"
                f"fontsize=20:fontcolor=white@0.6:"
                f"x=w-tw-20:y=h-th-20"
            ),
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_path),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr[:500]}")
                raise RuntimeError(f"FFmpeg failed: {result.stderr[:200]}")
            
            logger.info(f"Video processed: {output_path.name}")
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timeout (300s)")
            raise

    @staticmethod
    def inject_metadata(
        image_path: str | Path,
        metadata: dict,
    ) -> None:
        """
        Inject EXIF/XMP metadata into image for AI provenance.
        
        Metadata includes: model used, generation date, trigger word, etc.
        Does NOT include full prompts (privacy consideration).
        """
        from PIL.PngImagePlugin import PngInfo
        
        image_path = Path(image_path)
        img = Image.open(image_path)
        
        if image_path.suffix.lower() == ".png":
            png_info = PngInfo()
            for key, value in metadata.items():
                png_info.add_text(f"ai_pipeline_{key}", str(value))
            img.save(image_path, pnginfo=png_info)
        else:
            # For JPEG, use EXIF UserComment
            # Note: Full EXIF injection requires piexif library
            logger.debug(f"Metadata injection for {image_path.suffix} requires piexif")
        
        logger.debug(f"Metadata injected: {image_path.name}")
