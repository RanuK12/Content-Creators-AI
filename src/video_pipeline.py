"""
Video Pipeline - End-to-end video generation from still images.

Flow: ComfyUI keyframes → Kling Image-to-Video → FFmpeg post-production.
Handles batch animation, compilation, thumbnail extraction, and reporting.
"""

import asyncio
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from src.api_wrappers.kling_api import KlingAPIClient
from src.batch_processing.comfyui_batch import ComfyUIBatchProcessor


@dataclass
class VideoConfig:
    """Configuration for a single video generation task."""
    source_image: str
    output_name: str
    prompt: str = "Cinematic subtle motion, photorealistic"
    duration: float = 5.0
    aspect_ratio: str = "16:9"
    motion_strength: float = 0.6
    camera_movement: str = "static"
    seed: int = 42


@dataclass
class VideoDeliverable:
    """Result of a video generation and processing task."""
    raw_video_path: Optional[Path] = None
    processed_video_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)
    generation_time: float = 0.0
    success: bool = False
    error: Optional[str] = None

    @property
    def summary(self) -> dict:
        return {
            "raw": str(self.raw_video_path) if self.raw_video_path else None,
            "processed": str(self.processed_video_path) if self.processed_video_path else None,
            "thumbnail": str(self.thumbnail_path) if self.thumbnail_path else None,
            "generation_time": round(self.generation_time, 2),
            "success": self.success,
        }


class VideoPipeline:
    """
    End-to-end video pipeline from still images to deliverable videos.

    Stages:
    1. Keyframe generation (ComfyUI interpolation frames)
    2. Image-to-Video animation (Kling API)
    3. FFmpeg post-production (encoding, compilation, thumbnails)
    4. Reporting and metrics
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize clients
        comfyui_cfg = self.config["comfyui"]
        kling_cfg = self.config["kling"]

        self.comfyui = ComfyUIBatchProcessor(
            host=comfyui_cfg["host"],
            port=comfyui_cfg["port"],
        )
        self.kling = KlingAPIClient(
            base_url=kling_cfg["base_url"],
            polling_interval=kling_cfg["polling"]["interval"],
            max_wait=kling_cfg["polling"]["max_wait"],
        )

        # FFmpeg settings
        self.ffmpeg_settings = self.config["post_production"]["ffmpeg"]
        self.output_base = Path(self.config["paths"]["output"]) / "videos"
        self.output_base.mkdir(parents=True, exist_ok=True)

        logger.info("VideoPipeline initialized")

    def _load_config(self) -> dict:
        """Load pipeline configuration."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    async def generate_keyframes(
        self,
        source_image: str,
        num_frames: int = 4,
        workflow_path: str = "workflows/workflow_c_motion_reel/workflow_api.json",
        output_dir: Optional[str] = None,
    ) -> list[Path]:
        """
        Generate interpolation keyframes from a source image using ComfyUI.

        Args:
            source_image: Path to the source image.
            num_frames: Number of keyframes to generate.
            workflow_path: ComfyUI workflow for frame interpolation.
            output_dir: Output directory for keyframes.

        Returns:
            List of paths to generated keyframe images.
        """
        output = Path(output_dir) if output_dir else self.output_base / "keyframes"
        output.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {num_frames} keyframes from {source_image}")

        frames = await self.comfyui.generate_batch(
            workflow_path=workflow_path,
            prompts=[f"keyframe_{i}" for i in range(num_frames)],
            lora_name="",
            lora_weight=0.0,
            seeds=[42 + i * 100 for i in range(num_frames)],
            output_dir=str(output),
            source_image=source_image,
        )

        keyframe_paths = [Path(f) for f in frames]
        logger.info(f"Generated {len(keyframe_paths)} keyframes")
        return keyframe_paths

    async def animate_image(self, config: VideoConfig) -> VideoDeliverable:
        """
        Animate a single image to video using Kling API.

        Args:
            config: VideoConfig with generation parameters.

        Returns:
            VideoDeliverable with paths and metadata.
        """
        deliverable = VideoDeliverable()
        start_time = time.time()

        try:
            logger.info(f"Animating: {config.output_name} ({config.duration}s, {config.camera_movement})")

            # Generate raw video via Kling
            raw_dir = self.output_base / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_path = await self.kling.image_to_video(
                image_path=config.source_image,
                prompt=config.prompt,
                duration=config.duration,
                camera_movement=config.camera_movement,
                output_dir=str(raw_dir),
            )

            if not raw_path:
                deliverable.error = "Kling API returned no video"
                return deliverable

            deliverable.raw_video_path = Path(raw_path)

            # Post-process with FFmpeg
            processed_dir = self.output_base / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            processed_path = processed_dir / f"{config.output_name}.mp4"

            await self._ffmpeg_process(
                input_path=deliverable.raw_video_path,
                output_path=processed_path,
            )
            deliverable.processed_video_path = processed_path

            # Extract thumbnail
            thumbnail_path = await self._extract_thumbnail(
                video_path=processed_path,
                output_name=config.output_name,
            )
            deliverable.thumbnail_path = thumbnail_path

            # Metadata
            deliverable.metadata = {
                "prompt": config.prompt,
                "duration": config.duration,
                "aspect_ratio": config.aspect_ratio,
                "motion_strength": config.motion_strength,
                "camera_movement": config.camera_movement,
                "seed": config.seed,
                "source_image": config.source_image,
            }

            deliverable.success = True
            logger.success(f"Video complete: {config.output_name}")

        except Exception as e:
            deliverable.error = str(e)
            logger.error(f"Animation failed for {config.output_name}: {e}")

        finally:
            deliverable.generation_time = time.time() - start_time

        return deliverable

    async def batch_animate(self, configs: list[VideoConfig]) -> list[VideoDeliverable]:
        """
        Animate multiple images sequentially.

        Args:
            configs: List of VideoConfig objects.

        Returns:
            List of VideoDeliverable results.
        """
        logger.info(f"Starting batch animation: {len(configs)} videos")
        deliverables = []

        for i, config in enumerate(configs, 1):
            logger.info(f"Animating {i}/{len(configs)}: {config.output_name}")
            deliverable = await self.animate_image(config)
            deliverables.append(deliverable)

            # Rate limiting between API calls
            if i < len(configs):
                await asyncio.sleep(3.0)

        successful = sum(1 for d in deliverables if d.success)
        logger.info(f"Batch complete: {successful}/{len(configs)} successful")

        return deliverables

    async def full_pipeline(
        self,
        source_images: list[str],
        character_name: str,
        base_prompt: str = "Cinematic motion, photorealistic character",
        duration: float = 5.0,
        camera_movement: str = "slow_zoom_in",
        compile_reel: bool = True,
    ) -> dict:
        """
        Execute the full video pipeline for a set of source images.

        Args:
            source_images: List of source image paths.
            character_name: Character identifier for output naming.
            base_prompt: Base prompt for video generation.
            duration: Video duration in seconds.
            camera_movement: Camera movement type.
            compile_reel: Whether to create a compilation reel.

        Returns:
            Dictionary with pipeline results and paths.
        """
        logger.info(f"Full video pipeline: {character_name} ({len(source_images)} images)")

        # Build configs
        configs = [
            VideoConfig(
                source_image=img,
                output_name=f"{character_name}_{i:03d}",
                prompt=base_prompt,
                duration=duration,
                camera_movement=camera_movement,
                seed=42 + i * 111,
            )
            for i, img in enumerate(source_images)
        ]

        # Animate all images
        deliverables = await self.batch_animate(configs)

        # Compile reel if requested
        reel_path = None
        if compile_reel:
            successful_videos = [
                d.processed_video_path for d in deliverables
                if d.success and d.processed_video_path
            ]
            if len(successful_videos) >= 2:
                reel_path = await self._create_compilation(
                    videos=successful_videos,
                    output_name=f"{character_name}_reel",
                )

        # Generate report
        report = self.generate_report(deliverables, character_name)

        return {
            "deliverables": deliverables,
            "reel_path": reel_path,
            "report": report,
            "character": character_name,
        }

    async def _create_compilation(
        self, videos: list[Path], output_name: str
    ) -> Optional[Path]:
        """
        Create a compilation reel from multiple video clips using FFmpeg.

        Args:
            videos: List of video paths to concatenate.
            output_name: Name for the output compilation.

        Returns:
            Path to compilation video, or None on failure.
        """
        try:
            reel_dir = self.output_base / "reels"
            reel_dir.mkdir(parents=True, exist_ok=True)

            # Create concat file list
            concat_file = reel_dir / f"{output_name}_list.txt"
            with open(concat_file, "w") as f:
                for video in videos:
                    f.write(f"file '{video.resolve()}'\n")

            output_path = reel_dir / f"{output_name}.mp4"

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", self.ffmpeg_settings.get("codec", "libx264"),
                "-crf", str(self.ffmpeg_settings.get("crf", 18)),
                "-preset", self.ffmpeg_settings.get("preset", "slow"),
                "-pix_fmt", "yuv420p",
                str(output_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()

            if process.returncode == 0:
                logger.success(f"Compilation created: {output_path}")
                # Clean up concat list
                concat_file.unlink(missing_ok=True)
                return output_path
            else:
                logger.error(f"FFmpeg compilation failed: {stderr.decode()}")
                return None

        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return None

    async def _extract_thumbnail(
        self, video_path: Path, output_name: str, timestamp: float = 1.0
    ) -> Optional[Path]:
        """
        Extract a thumbnail frame from a video.

        Args:
            video_path: Path to the video file.
            output_name: Base name for thumbnail.
            timestamp: Time position to extract frame (seconds).

        Returns:
            Path to thumbnail image.
        """
        try:
            thumb_dir = self.output_base / "thumbnails"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            thumb_path = thumb_dir / f"{output_name}_thumb.jpg"

            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-ss", str(timestamp),
                "-vframes", "1",
                "-q:v", "2",
                str(thumb_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

            if process.returncode == 0 and thumb_path.exists():
                return thumb_path
            return None

        except Exception as e:
            logger.warning(f"Thumbnail extraction failed: {e}")
            return None

    async def _ffmpeg_process(self, input_path: Path, output_path: Path) -> None:
        """Apply FFmpeg post-processing to a video."""
        codec = self.ffmpeg_settings.get("codec", "libx264")
        crf = self.ffmpeg_settings.get("crf", 18)
        preset = self.ffmpeg_settings.get("preset", "slow")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg processing failed: {stderr.decode()[:500]}")

    def generate_report(
        self, deliverables: list[VideoDeliverable], character_name: str
    ) -> dict:
        """
        Generate a summary report for video pipeline execution.

        Args:
            deliverables: List of VideoDeliverable results.
            character_name: Character name for report context.

        Returns:
            Report dictionary with statistics and details.
        """
        successful = [d for d in deliverables if d.success]
        failed = [d for d in deliverables if not d.success]
        total_time = sum(d.generation_time for d in deliverables)

        report = {
            "character": character_name,
            "total_videos": len(deliverables),
            "successful": len(successful),
            "failed": len(failed),
            "total_generation_time": round(total_time, 2),
            "avg_time_per_video": round(total_time / max(len(deliverables), 1), 2),
            "success_rate": round(len(successful) / max(len(deliverables), 1), 4),
            "errors": [d.error for d in failed if d.error],
            "deliverables": [d.summary for d in deliverables],
        }

        logger.info(
            f"Video Report [{character_name}]: "
            f"{report['successful']}/{report['total_videos']} successful, "
            f"avg {report['avg_time_per_video']}s/video"
        )

        return report
