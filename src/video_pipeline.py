"""
Video Pipeline Module
======================
End-to-end video generation: ComfyUI frames → Kling I2V → FFmpeg post-prod.

Pipeline stages:
1. Generate key frames via ComfyUI (character-consistent)
2. Animate best frame with Kling AI Image-to-Video
3. Post-process with FFmpeg (watermark, resize, captions)
4. Export deliverables (MP4 + metadata + generation report)

Features:
- Camera movement control (dolly, pan, zoom)
- Temporal consistency metrics
- Batch video generation with concurrency control
- Automatic retry with exponential backoff
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

from src.api_wrappers.kling_api import CameraMovement, KlingAPIClient, VideoTask
from src.batch_processing.comfyui_batch import ComfyUIBatchGenerator
from src.post_production.post_prod import PostProduction


@dataclass
class VideoConfig:
    """Configuration for a single video generation."""
    source_image: str | Path
    output_name: str
    prompt: str = "subtle motion, cinematic, high quality"
    negative_prompt: str = "blurry, distorted, low quality, glitch"
    duration: float = 5.0
    aspect_ratio: str = "9:16"
    motion_strength: float = 0.7
    camera_movement: CameraMovement = CameraMovement.NONE
    camera_strength: float = 0.5
    seed: Optional[int] = None


@dataclass
class VideoDeliverable:
    """Final video deliverable with metadata."""
    raw_video_path: Optional[Path] = None
    processed_video_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    metadata: dict = field(default_factory=dict)
    generation_time_seconds: float = 0.0
    success: bool = False
    error: Optional[str] = None


class VideoPipeline:
    """
    Orchestrates the complete video generation pipeline.
    
    Integrates ComfyUI for frame generation, Kling AI for animation,
    and FFmpeg for post-production.
    """

    def __init__(
        self,
        kling_client: KlingAPIClient,
        comfyui_client: Optional[ComfyUIBatchGenerator] = None,
        output_dir: str | Path = "output/videos/",
        watermark_text: str = "AI Generated Concept Art",
    ):
        self.kling = kling_client
        self.comfyui = comfyui_client
        self.output_dir = Path(output_dir)
        self.watermark_text = watermark_text
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VideoPipeline initialized | output={output_dir}")

    async def generate_keyframes(
        self,
        workflow_path: str | Path,
        prompts: list[str],
        lora_name: str,
        lora_weight: float = 0.8,
        output_dir: Optional[Path] = None,
    ) -> list[Path]:
        """
        Generate key frames using ComfyUI for later animation.
        
        Selects the best frames based on quality/composition.
        """
        if self.comfyui is None:
            raise RuntimeError("ComfyUI client not initialized")
        
        if output_dir is None:
            output_dir = self.output_dir / "keyframes"
        
        results = await self.comfyui.generate_character_batch(
            workflow_path=workflow_path,
            character_lora=lora_name,
            lora_weight=lora_weight,
            prompts=prompts,
            output_dir=str(output_dir),
        )
        
        # Flatten
        frames = [p for batch in results for p in batch]
        logger.info(f"Generated {len(frames)} keyframes")
        return frames

    async def animate_image(self, config: VideoConfig) -> VideoDeliverable:
        """
        Animate a single image through the full pipeline.
        
        Steps: Kling I2V → FFmpeg post-production → Thumbnail extraction
        """
        deliverable = VideoDeliverable()
        start_time = time.time()
        
        try:
            # Create output directories
            raw_dir = self.output_dir / "raw"
            processed_dir = self.output_dir / "processed"
            thumb_dir = self.output_dir / "thumbnails"
            raw_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)
            thumb_dir.mkdir(parents=True, exist_ok=True)
            
            raw_path = raw_dir / f"{config.output_name}_raw.mp4"
            
            # --- Stage 1: Kling I2V ---
            logger.info(f"Animating: {Path(config.source_image).name} → {config.output_name}")
            
            task = await self.kling.generate_video(
                image_path=config.source_image,
                output_path=raw_path,
                prompt=config.prompt,
                duration=config.duration,
                aspect_ratio=config.aspect_ratio,
                motion_strength=config.motion_strength,
                camera_movement=config.camera_movement,
                camera_strength=config.camera_strength,
            )
            
            if task is None or task.local_path is None:
                deliverable.error = "Kling AI generation failed"
                return deliverable
            
            deliverable.raw_video_path = Path(task.local_path)
            
            # --- Stage 2: FFmpeg Post-Production ---
            processed_path = processed_dir / f"{config.output_name}.mp4"
            
            PostProduction.process_video_ffmpeg(
                input_path=deliverable.raw_video_path,
                output_path=processed_path,
                watermark_text=self.watermark_text,
                target_resolution=(1080, 1920) if config.aspect_ratio == "9:16" else (1920, 1080),
            )
            
            deliverable.processed_video_path = processed_path
            
            # --- Stage 3: Extract Thumbnail ---
            thumb_path = thumb_dir / f"{config.output_name}_thumb.jpg"
            self._extract_thumbnail(processed_path, thumb_path)
            deliverable.thumbnail_path = thumb_path
            
            # --- Metadata ---
            deliverable.metadata = {
                "source_image": str(config.source_image),
                "prompt": config.prompt,
                "duration": config.duration,
                "camera_movement": config.camera_movement.value,
                "motion_strength": config.motion_strength,
                "aspect_ratio": config.aspect_ratio,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "ai_generated": True,
            }
            
            deliverable.success = True
            
        except Exception as e:
            deliverable.error = str(e)
            logger.error(f"Video pipeline error: {e}")
        
        finally:
            deliverable.generation_time_seconds = time.time() - start_time
        
        return deliverable

    async def batch_animate(
        self,
        configs: list[VideoConfig],
        concurrency: int = 2,
    ) -> list[VideoDeliverable]:
        """
        Batch animate multiple images with controlled concurrency.
        
        Kling API has rate limits, so concurrency should be conservative.
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process(config: VideoConfig) -> VideoDeliverable:
            async with semaphore:
                return await self.animate_image(config)
        
        tasks = [process(c) for c in configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        deliverables = []
        for r in results:
            if isinstance(r, VideoDeliverable):
                deliverables.append(r)
            elif isinstance(r, Exception):
                d = VideoDeliverable(error=str(r))
                deliverables.append(d)
        
        success_count = sum(1 for d in deliverables if d.success)
        logger.info(f"Batch animation: {success_count}/{len(configs)} successful")
        
        return deliverables

    async def full_pipeline(
        self,
        workflow_path: str | Path,
        prompts: list[str],
        lora_name: str,
        lora_weight: float = 0.8,
        video_prompt: str = "subtle motion, cinematic",
        camera_movement: CameraMovement = CameraMovement.DOLLY_IN,
        duration: float = 5.0,
    ) -> list[VideoDeliverable]:
        """
        Complete pipeline: generate frames → animate → post-produce.
        
        Args:
            workflow_path: ComfyUI workflow for frame generation
            prompts: Prompts for keyframe generation
            lora_name: Character LoRA
            lora_weight: LoRA strength
            video_prompt: Motion prompt for Kling
            camera_movement: Camera type
            duration: Video duration
            
        Returns:
            List of VideoDeliverable objects
        """
        # Step 1: Generate keyframes
        frames = await self.generate_keyframes(
            workflow_path=workflow_path,
            prompts=prompts,
            lora_name=lora_name,
            lora_weight=lora_weight,
        )
        
        if not frames:
            logger.error("No keyframes generated")
            return []
        
        # Step 2: Create video configs for each frame
        configs = []
        for i, frame in enumerate(frames):
            configs.append(VideoConfig(
                source_image=frame,
                output_name=f"clip_{i:03d}",
                prompt=video_prompt,
                duration=duration,
                camera_movement=camera_movement,
            ))
        
        # Step 3: Animate all frames
        deliverables = await self.batch_animate(configs)
        
        # Step 4: Generate compilation if multiple clips
        if len([d for d in deliverables if d.success]) > 1:
            await self._create_compilation(deliverables)
        
        return deliverables

    async def _create_compilation(self, deliverables: list[VideoDeliverable]) -> Optional[Path]:
        """Concatenate successful videos into a compilation reel."""
        successful = [d for d in deliverables if d.success and d.processed_video_path]
        
        if len(successful) < 2:
            return None
        
        # Create file list for FFmpeg concat
        list_path = self.output_dir / "concat_list.txt"
        with open(list_path, "w") as f:
            for d in successful:
                f.write(f"file '{d.processed_video_path.absolute()}'\n")
        
        output_path = self.output_dir / "compilation_reel.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output_path),
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
            logger.info(f"Compilation reel created: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Compilation failed: {e.stderr[:200]}")
            return None
        finally:
            list_path.unlink(missing_ok=True)

    @staticmethod
    def _extract_thumbnail(video_path: Path, output_path: Path, time_offset: str = "00:00:01"):
        """Extract a single frame as thumbnail."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", time_offset,
            "-vframes", "1",
            "-q:v", "2",
            str(output_path),
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        except subprocess.CalledProcessError:
            logger.warning(f"Thumbnail extraction failed for {video_path.name}")

    def generate_report(self, deliverables: list[VideoDeliverable]) -> Path:
        """Generate JSON report for video batch."""
        report_path = self.output_dir / "video_pipeline_report.json"
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_videos": len(deliverables),
            "successful": sum(1 for d in deliverables if d.success),
            "failed": sum(1 for d in deliverables if not d.success),
            "total_time_seconds": sum(d.generation_time_seconds for d in deliverables),
            "deliverables": [
                {
                    "success": d.success,
                    "processed_path": str(d.processed_video_path) if d.processed_video_path else None,
                    "time_seconds": round(d.generation_time_seconds, 1),
                    "error": d.error,
                    "metadata": d.metadata,
                }
                for d in deliverables
            ],
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Video report: {report_path}")
        return report_path
