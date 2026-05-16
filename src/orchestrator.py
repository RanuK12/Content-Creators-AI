"""
Pipeline Orchestrator - Coordinates the full AI art generation pipeline.

Manages the end-to-end workflow: image generation → evaluation → video creation
→ post-production → platform-specific output organization.
"""

import asyncio
import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from src.api_wrappers.kling_api import KlingAPIClient
from src.batch_processing.comfyui_batch import ComfyUIBatchProcessor
from src.evaluation.facial_consistency import FacialConsistencyEvaluator
from src.post_production.post_prod import PostProductionPipeline


@dataclass
class PipelineTask:
    """Represents a single pipeline execution task."""
    character_name: str
    workflow_path: str
    prompts: list[str]
    lora_name: str
    lora_weight: float = 0.85
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789])
    generate_video: bool = False
    video_prompt: str = ""
    video_duration: float = 5.0
    camera_movement: str = "static"
    platforms: list[str] = field(default_factory=lambda: ["instagram", "twitter", "portfolio"])


@dataclass
class PipelineResult:
    """Result of a pipeline task execution."""
    task: PipelineTask
    generated_images: list[Path] = field(default_factory=list)
    filtered_images: list[Path] = field(default_factory=list)
    consistency_score: float = 0.0
    videos_generated: list[Path] = field(default_factory=list)
    watermarked_images: list[Path] = field(default_factory=list)
    platform_outputs: dict[str, list[Path]] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = False
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def summary(self) -> dict:
        return {
            "character": self.task.character_name,
            "images_generated": len(self.generated_images),
            "images_filtered": len(self.filtered_images),
            "consistency_score": round(self.consistency_score, 4),
            "videos_generated": len(self.videos_generated),
            "platforms_served": list(self.platform_outputs.keys()),
            "duration_seconds": round(self.duration, 2),
            "success": self.success,
        }


class PipelineOrchestrator:
    """
    Coordinates the full generation pipeline from image creation to delivery.

    Pipeline stages:
    1. Image Generation (ComfyUI batch processing)
    2. Quality Evaluation (facial consistency scoring)
    3. Filtering (threshold-based selection)
    4. Video Generation (Kling API image-to-video)
    5. Post-Production (watermarking, resizing)
    6. Output Organization (platform-specific deliverables)
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize pipeline components
        self.comfyui = ComfyUIBatchProcessor(
            host=self.config["comfyui"]["host"],
            port=self.config["comfyui"]["port"],
        )
        self.evaluator = FacialConsistencyEvaluator(
            threshold=self.config["evaluation"]["facial_consistency"]["threshold"]
        )
        self.post_prod = PostProductionPipeline(
            watermark_config=self.config["post_production"]["watermark"],
            resize_presets=self.config["post_production"]["resize_presets"],
        )
        self.kling_client = KlingAPIClient(
            base_url=self.config["kling"]["base_url"],
            polling_interval=self.config["kling"]["polling"]["interval"],
            max_wait=self.config["kling"]["polling"]["max_wait"],
        )

        # Pipeline settings
        self.consistency_threshold = self.config["evaluation"]["facial_consistency"]["threshold"]
        self.output_base = Path(self.config["paths"]["output"])
        self.output_base.mkdir(parents=True, exist_ok=True)

        logger.info(f"PipelineOrchestrator initialized with config: {config_path}")
        logger.info(f"Consistency threshold: {self.consistency_threshold}")
        logger.info(f"Output directory: {self.output_base}")

    def _load_config(self) -> dict:
        """Load and validate configuration from YAML."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.debug(f"Configuration loaded from {self.config_path}")
        return config

    async def execute_task(self, task: PipelineTask) -> PipelineResult:
        """
        Execute a single pipeline task through all stages.

        Args:
            task: PipelineTask defining the generation parameters.

        Returns:
            PipelineResult with all outputs and metrics.
        """
        result = PipelineResult(task=task, start_time=time.time())
        logger.info(f"Starting pipeline for character: {task.character_name}")

        try:
            # Stage 1: Generate images via ComfyUI
            logger.info(f"[Stage 1/6] Generating images - {len(task.prompts)} prompts × {len(task.seeds)} seeds")
            generated = await self._generate_images(task)
            result.generated_images = generated
            logger.info(f"Generated {len(generated)} images")

            # Stage 2: Evaluate facial consistency
            logger.info("[Stage 2/6] Evaluating facial consistency")
            scores = await self._evaluate_consistency(generated, task.character_name)
            result.consistency_score = sum(scores.values()) / max(len(scores), 1)
            logger.info(f"Average consistency score: {result.consistency_score:.4f}")

            # Stage 3: Filter by threshold
            logger.info(f"[Stage 3/6] Filtering (threshold={self.consistency_threshold})")
            filtered = [
                img for img, score in scores.items()
                if score >= self.consistency_threshold
            ]
            result.filtered_images = filtered
            logger.info(f"Passed filter: {len(filtered)}/{len(generated)} images")

            # Stage 4: Generate videos (if enabled)
            if task.generate_video and filtered:
                logger.info("[Stage 4/6] Generating videos via Kling API")
                videos = await self._generate_videos(task, filtered)
                result.videos_generated = videos
                logger.info(f"Generated {len(videos)} videos")
            else:
                logger.info("[Stage 4/6] Video generation skipped")

            # Stage 5: Post-production (watermark + resize)
            logger.info("[Stage 5/6] Applying post-production")
            watermarked = await self._apply_post_production(filtered, task.character_name)
            result.watermarked_images = watermarked

            # Stage 6: Organize platform-specific outputs
            logger.info("[Stage 6/6] Organizing platform outputs")
            platform_outputs = await self._organize_outputs(
                watermarked, task.platforms, task.character_name
            )
            result.platform_outputs = platform_outputs

            # Compile metrics
            result.metrics = {
                "total_generated": len(generated),
                "passed_filter": len(filtered),
                "filter_rate": len(filtered) / max(len(generated), 1),
                "avg_consistency": result.consistency_score,
                "videos_created": len(result.videos_generated),
                "platforms_served": len(platform_outputs),
            }

            result.success = True
            logger.success(f"Pipeline complete for {task.character_name}: {result.summary}")

        except Exception as e:
            result.error = str(e)
            logger.error(f"Pipeline failed for {task.character_name}: {e}")

        finally:
            result.end_time = time.time()

        return result

    async def execute_batch(self, tasks: list[PipelineTask]) -> list[PipelineResult]:
        """
        Execute multiple pipeline tasks sequentially.

        Args:
            tasks: List of PipelineTask objects.

        Returns:
            List of PipelineResult objects.
        """
        logger.info(f"Starting batch execution: {len(tasks)} tasks")
        results = []

        for i, task in enumerate(tasks, 1):
            logger.info(f"--- Task {i}/{len(tasks)}: {task.character_name} ---")
            result = await self.execute_task(task)
            results.append(result)

            # Brief pause between tasks to avoid overloading
            if i < len(tasks):
                await asyncio.sleep(2.0)

        # Batch summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {successful}/{len(tasks)} succeeded")

        return results

    async def _generate_images(self, task: PipelineTask) -> list[Path]:
        """Generate images using ComfyUI batch processor."""
        output_dir = self.output_base / task.character_name / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        images = await self.comfyui.generate_batch(
            workflow_path=task.workflow_path,
            prompts=task.prompts,
            lora_name=task.lora_name,
            lora_weight=task.lora_weight,
            seeds=task.seeds,
            output_dir=str(output_dir),
        )

        return [Path(img) for img in images]

    async def _evaluate_consistency(
        self, images: list[Path], character_name: str
    ) -> dict[Path, float]:
        """Evaluate facial consistency for all generated images."""
        reference_dir = Path(self.config["paths"]["references"]) / character_name

        scores = {}
        for image_path in images:
            score = await self.evaluator.evaluate(
                generated_image=str(image_path),
                reference_dir=str(reference_dir),
            )
            scores[image_path] = score

        return scores

    async def _generate_videos(
        self, task: PipelineTask, source_images: list[Path]
    ) -> list[Path]:
        """Generate videos from filtered images using Kling API."""
        video_dir = self.output_base / task.character_name / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        videos = []
        for img in source_images[:4]:  # Limit to top 4 images for video
            video_path = await self.kling_client.image_to_video(
                image_path=str(img),
                prompt=task.video_prompt or f"Cinematic motion of {task.character_name}",
                duration=task.video_duration,
                camera_movement=task.camera_movement,
                output_dir=str(video_dir),
            )
            if video_path:
                videos.append(Path(video_path))

        return videos

    async def _apply_post_production(
        self, images: list[Path], character_name: str
    ) -> list[Path]:
        """Apply watermarking and basic post-production."""
        output_dir = self.output_base / character_name / "watermarked"
        output_dir.mkdir(parents=True, exist_ok=True)

        watermarked = await self.post_prod.batch_watermark(
            images=[str(img) for img in images],
            output_dir=str(output_dir),
        )

        return [Path(w) for w in watermarked]

    async def _organize_outputs(
        self, images: list[Path], platforms: list[str], character_name: str
    ) -> dict[str, list[Path]]:
        """Resize and organize outputs per platform."""
        platform_outputs = {}

        for platform in platforms:
            platform_dir = self.output_base / character_name / "platforms" / platform
            platform_dir.mkdir(parents=True, exist_ok=True)

            resized = await self.post_prod.batch_resize(
                images=[str(img) for img in images],
                platform=platform,
                output_dir=str(platform_dir),
            )
            platform_outputs[platform] = [Path(r) for r in resized]

        return platform_outputs

    @classmethod
    def from_task_file(cls, task_file: str, config_path: str = "config.yaml") -> tuple:
        """
        Load pipeline tasks from a JSON task file.

        Args:
            task_file: Path to JSON file containing task definitions.
            config_path: Path to config.yaml.

        Returns:
            Tuple of (PipelineOrchestrator, list[PipelineTask]).
        """
        with open(task_file, "r") as f:
            task_data = json.load(f)

        tasks = []
        for entry in task_data.get("tasks", []):
            task = PipelineTask(
                character_name=entry["character_name"],
                workflow_path=entry["workflow_path"],
                prompts=entry["prompts"],
                lora_name=entry["lora_name"],
                lora_weight=entry.get("lora_weight", 0.85),
                seeds=entry.get("seeds", [42, 123, 456, 789]),
                generate_video=entry.get("generate_video", False),
                video_prompt=entry.get("video_prompt", ""),
                video_duration=entry.get("video_duration", 5.0),
                camera_movement=entry.get("camera_movement", "static"),
                platforms=entry.get("platforms", ["instagram", "twitter", "portfolio"]),
            )
            tasks.append(task)

        orchestrator = cls(config_path=config_path)
        logger.info(f"Loaded {len(tasks)} tasks from {task_file}")

        return orchestrator, tasks


async def main():
    """CLI entry point for pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="AI Art Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.orchestrator --task-file tasks.json
  python -m src.orchestrator --task-file tasks.json --config custom_config.yaml
  python -m src.orchestrator --task-file tasks.json --dry-run
        """,
    )

    parser.add_argument(
        "--task-file",
        type=str,
        required=True,
        help="Path to JSON task definition file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate tasks without executing pipeline",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.add("logs/pipeline_{time}.log", level="DEBUG", rotation="50 MB")
    else:
        logger.add("logs/pipeline_{time}.log", level="INFO", rotation="50 MB")

    logger.info("=" * 60)
    logger.info("AI Art Pipeline Orchestrator")
    logger.info("=" * 60)

    # Load orchestrator and tasks
    orchestrator, tasks = PipelineOrchestrator.from_task_file(
        task_file=args.task_file,
        config_path=args.config,
    )

    # Override output directory if specified
    if args.output_dir:
        orchestrator.output_base = Path(args.output_dir)
        orchestrator.output_base.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("DRY RUN - Validating tasks only")
        for i, task in enumerate(tasks, 1):
            logger.info(f"  Task {i}: {task.character_name} ({len(task.prompts)} prompts)")
        logger.info("Validation complete. All tasks are valid.")
        return

    # Execute batch
    results = await orchestrator.execute_batch(tasks)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE RESULTS SUMMARY")
    logger.info("=" * 60)

    for result in results:
        status = "OK" if result.success else "FAILED"
        logger.info(
            f"  [{status}] {result.task.character_name}: "
            f"{len(result.filtered_images)} images, "
            f"{len(result.videos_generated)} videos, "
            f"score={result.consistency_score:.3f}, "
            f"time={result.duration:.1f}s"
        )

    total_time = sum(r.duration for r in results)
    logger.info(f"\nTotal execution time: {total_time:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
