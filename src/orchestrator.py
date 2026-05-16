"""
Pipeline Orchestrator
======================
Central coordinator for the complete AI art generation pipeline.

Flow:
    Config YAML + Prompt List
    → Generate images (ComfyUI batch)
    → Evaluate consistency (FaceNet/InsightFace)
    → Filter best (threshold > 0.85)
    → Generate videos (Kling I2V)
    → Post-produce (FFmpeg + watermark)
    → Organize output structure
    → Generate report JSON with metrics

Research relevance:
- End-to-end automation reduces human intervention to config editing
- Metrics-driven filtering ensures only high-quality outputs are published
- Full provenance tracking for reproducibility
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from src.api_wrappers.kling_api import CameraMovement, KlingAPIClient
from src.batch_processing.comfyui_batch import ComfyUIBatchGenerator
from src.evaluation.facial_consistency import FacialConsistencyEvaluator
from src.post_production.post_prod import PostProduction, WatermarkConfig


@dataclass
class PipelineTask:
    """A single pipeline execution task."""
    character_name: str
    workflow_path: str
    prompts: list[str]
    lora_name: str
    lora_weight: float = 0.8
    seeds: list[int] = field(default_factory=list)
    generate_video: bool = True
    video_prompt: str = ""
    video_duration: float = 5.0
    camera_movement: str = "none"
    platforms: list[str] = field(default_factory=lambda: ["instagram_feed", "twitter"])


@dataclass
class PipelineResult:
    """Results from a pipeline execution."""
    task: PipelineTask
    generated_images: list[Path] = field(default_factory=list)
    filtered_images: list[Path] = field(default_factory=list)
    consistency_score: float = 0.0
    videos_generated: list[Path] = field(default_factory=list)
    watermarked_images: list[Path] = field(default_factory=list)
    platform_outputs: dict[str, list[Path]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    success: bool = False
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "character": self.task.character_name,
            "success": self.success,
            "duration_seconds": round(self.duration_seconds, 1),
            "images_generated": len(self.generated_images),
            "images_filtered": len(self.filtered_images),
            "consistency_score": round(self.consistency_score, 4),
            "videos_generated": len(self.videos_generated),
            "platforms": list(self.platform_outputs.keys()),
            "error": self.error,
            "metrics": self.metrics,
        }


class PipelineOrchestrator:
    """
    Main orchestrator that coordinates the full generation pipeline.
    
    Connects all modules:
    - ComfyUI batch generation
    - Facial consistency evaluation
    - Kling AI video generation
    - Post-production (watermark, resize)
    - Output organization
    """

    def __init__(self, config_path: str | Path = "config.yaml"):
        """
        Initialize orchestrator from config file.
        
        Args:
            config_path: Path to global config.yaml
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize modules
        self.comfyui = ComfyUIBatchGenerator(
            host=self.config["comfyui"]["host"],
            port=self.config["comfyui"]["port"],
            timeout=self.config["comfyui"]["timeout_seconds"],
            max_retries=self.config["comfyui"]["max_retries"],
        )
        
        self.evaluator = FacialConsistencyEvaluator(
            similarity_threshold=self.config["evaluation"]["facial_consistency"]["similarity_threshold"],
            min_face_size=self.config["evaluation"]["facial_consistency"]["min_face_size"],
        )
        
        self.post_prod = PostProduction(
            watermark_config=WatermarkConfig(
                text=self.config["post_production"]["watermark"]["text"],
                font_size=self.config["post_production"]["watermark"]["font_size"],
                opacity=self.config["post_production"]["watermark"]["opacity"],
                position=self.config["post_production"]["watermark"]["position"],
            )
        )
        
        # Kling client (initialized lazily when needed)
        self._kling_client: Optional[KlingAPIClient] = None
        
        # Output paths
        self.output_base = Path(self.config["paths"]["output_base"])
        self.reports_dir = Path(self.config["paths"]["reports"])
        
        logger.info("PipelineOrchestrator initialized")

    def _load_config(self) -> dict:
        """Load and validate config.yaml."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _get_kling_client(self) -> KlingAPIClient:
        """Lazy-initialize Kling API client."""
        if self._kling_client is None:
            import os
            self._kling_client = KlingAPIClient(
                access_key=os.getenv("KLING_API_KEY", ""),
                secret_key=os.getenv("KLING_API_SECRET", ""),
                base_url=self.config["kling"]["base_url"],
                polling_interval=self.config["kling"]["polling_interval_seconds"],
                max_polling_attempts=self.config["kling"]["max_polling_attempts"],
            )
        return self._kling_client

    def _organize_output(
        self,
        character_name: str,
        content_type: str = "images",
    ) -> Path:
        """
        Create organized output directory.
        
        Structure: output/portfolio/{character}/{date}/{type}/
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.output_base / "portfolio" / character_name / date_str / content_type
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    async def execute_task(self, task: PipelineTask) -> PipelineResult:
        """
        Execute a single pipeline task end-to-end.
        
        Steps:
        1. Generate images via ComfyUI
        2. Evaluate facial consistency
        3. Filter by threshold
        4. Generate videos (optional)
        5. Post-produce (watermark + resize)
        6. Organize outputs
        
        Args:
            task: PipelineTask configuration
            
        Returns:
            PipelineResult with all outputs and metrics
        """
        result = PipelineResult(task=task, start_time=time.time())
        
        try:
            logger.info(f"{'='*60}")
            logger.info(f"PIPELINE START: {task.character_name}")
            logger.info(f"{'='*60}")
            
            # --- STEP 1: Generate Images ---
            logger.info("[1/5] Generating images via ComfyUI...")
            
            output_dir = self._organize_output(task.character_name, "raw")
            
            generated = await self.comfyui.generate_character_batch(
                workflow_path=task.workflow_path,
                character_lora=task.lora_name,
                lora_weight=task.lora_weight,
                prompts=task.prompts,
                seeds=task.seeds if task.seeds else None,
                output_dir=str(output_dir),
            )
            
            # Flatten results
            result.generated_images = [p for batch in generated for p in batch]
            logger.info(f"  Generated {len(result.generated_images)} images")
            
            if not result.generated_images:
                result.error = "No images generated"
                result.end_time = time.time()
                return result
            
            # --- STEP 2: Evaluate Consistency ---
            logger.info("[2/5] Evaluating facial consistency...")
            
            image_paths = [str(p) for p in result.generated_images]
            consistency_report = self.evaluator.evaluate_consistency(
                image_paths, technique=f"LoRA({task.lora_name})"
            )
            
            if consistency_report:
                result.consistency_score = consistency_report.mean_similarity
                result.metrics["consistency"] = consistency_report.to_dict()
                logger.info(f"  Mean similarity: {result.consistency_score:.4f}")
            
            # --- STEP 3: Filter by Threshold ---
            logger.info("[3/5] Filtering images by quality threshold...")
            
            threshold = self.config["evaluation"]["facial_consistency"]["similarity_threshold"]
            
            if consistency_report and len(result.generated_images) > 1:
                # Keep images with above-threshold similarity to reference (first image)
                reference_embedding = self.evaluator.extract_embedding(
                    result.generated_images[0]
                )
                
                if reference_embedding:
                    for img_path in result.generated_images:
                        emb = self.evaluator.extract_embedding(img_path)
                        if emb:
                            sim = self.evaluator.compute_cosine_similarity(
                                reference_embedding.embedding, emb.embedding
                            )
                            if sim >= threshold:
                                result.filtered_images.append(img_path)
                        else:
                            # Keep images where face isn't detected (full body, etc.)
                            result.filtered_images.append(img_path)
                else:
                    result.filtered_images = result.generated_images.copy()
            else:
                result.filtered_images = result.generated_images.copy()
            
            logger.info(
                f"  Filtered: {len(result.filtered_images)}/{len(result.generated_images)} "
                f"passed (threshold={threshold})"
            )
            
            # --- STEP 4: Generate Videos (optional) ---
            if task.generate_video and result.filtered_images:
                logger.info("[4/5] Generating videos via Kling AI...")
                
                video_dir = self._organize_output(task.character_name, "videos")
                kling = self._get_kling_client()
                
                # Use top image for video
                best_image = result.filtered_images[0]
                
                video_task = await kling.generate_video(
                    image_path=best_image,
                    output_path=video_dir / f"{task.character_name}_motion.mp4",
                    prompt=task.video_prompt,
                    duration=task.video_duration,
                    camera_movement=CameraMovement(task.camera_movement),
                )
                
                if video_task and video_task.local_path:
                    result.videos_generated.append(Path(video_task.local_path))
                    logger.info(f"  Video generated: {video_task.local_path}")
                else:
                    logger.warning("  Video generation failed")
            else:
                logger.info("[4/5] Skipping video generation")
            
            # --- STEP 5: Post-Production ---
            logger.info("[5/5] Post-production (watermark + resize)...")
            
            # Watermark
            watermark_dir = self._organize_output(task.character_name, "watermarked")
            for img_path in result.filtered_images:
                wm_path = self.post_prod.add_watermark(
                    img_path, watermark_dir / img_path.name
                )
                result.watermarked_images.append(wm_path)
                
                # Inject metadata
                PostProduction.inject_metadata(wm_path, {
                    "generator": "AI Art Pipeline v1.0",
                    "model": self.config["models"]["primary"]["name"],
                    "lora": task.lora_name,
                    "character": task.character_name,
                    "date": datetime.now().isoformat(),
                    "ai_generated": "true",
                })
            
            # Multi-platform resize
            platform_dir = self._organize_output(task.character_name, "platforms")
            result.platform_outputs = self.post_prod.batch_resize_multiplatform(
                watermark_dir, platform_dir, platforms=task.platforms
            )
            
            # Video post-production
            if result.videos_generated:
                processed_dir = self._organize_output(task.character_name, "videos_final")
                for video_path in result.videos_generated:
                    PostProduction.process_video_ffmpeg(
                        video_path,
                        processed_dir / f"{video_path.stem}_final.mp4",
                        watermark_text=self.config["post_production"]["watermark"]["text"],
                    )
            
            result.success = True
            logger.info(f"Pipeline complete for {task.character_name}")
            
        except Exception as e:
            result.error = str(e)
            logger.error(f"Pipeline error: {e}")
        
        finally:
            result.end_time = time.time()
            result.metrics["total_time_seconds"] = round(result.duration_seconds, 1)
        
        return result

    async def execute_batch(
        self, tasks: list[PipelineTask], sequential: bool = True
    ) -> list[PipelineResult]:
        """
        Execute multiple pipeline tasks.
        
        Args:
            tasks: List of pipeline tasks
            sequential: If True, process one at a time (VRAM safe)
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        logger.info(f"Batch execution: {len(tasks)} tasks (sequential={sequential})")
        
        if sequential:
            for i, task in enumerate(tasks):
                logger.info(f"\n{'#'*60}")
                logger.info(f"# TASK {i+1}/{len(tasks)}: {task.character_name}")
                logger.info(f"{'#'*60}\n")
                result = await self.execute_task(task)
                results.append(result)
        else:
            # Parallel (use with caution - VRAM constraints)
            coros = [self.execute_task(t) for t in tasks]
            results = await asyncio.gather(*coros)
        
        # Generate batch report
        await self._generate_batch_report(results)
        
        return results

    async def _generate_batch_report(self, results: list[PipelineResult]) -> Path:
        """Generate comprehensive JSON report for the batch."""
        report_path = self.reports_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tasks": len(results),
                "successful": sum(1 for r in results if r.success),
                "failed": sum(1 for r in results if not r.success),
                "total_time_seconds": sum(r.duration_seconds for r in results),
            },
            "tasks": [r.to_dict() for r in results],
            "summary": {
                "total_images_generated": sum(len(r.generated_images) for r in results),
                "total_images_filtered": sum(len(r.filtered_images) for r in results),
                "total_videos": sum(len(r.videos_generated) for r in results),
                "avg_consistency": (
                    sum(r.consistency_score for r in results if r.consistency_score > 0)
                    / max(1, sum(1 for r in results if r.consistency_score > 0))
                ),
            },
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Batch report saved: {report_path}")
        return report_path

    @classmethod
    def from_task_file(
        cls, task_file: str | Path, config_path: str | Path = "config.yaml"
    ) -> tuple["PipelineOrchestrator", list[PipelineTask]]:
        """
        Load orchestrator and tasks from a JSON task file.
        
        Task file format:
        {
            "tasks": [
                {
                    "character_name": "character_01",
                    "workflow_path": "workflows/workflow_a.../workflow_api.json",
                    "prompts": ["prompt 1", "prompt 2"],
                    "lora_name": "char_01.safetensors",
                    "lora_weight": 0.8,
                    "generate_video": true,
                    "video_prompt": "gentle smile, slight head turn",
                    "camera_movement": "dolly_in"
                }
            ]
        }
        """
        with open(task_file) as f:
            data = json.load(f)
        
        orchestrator = cls(config_path=config_path)
        
        tasks = []
        for task_data in data.get("tasks", []):
            tasks.append(PipelineTask(**task_data))
        
        return orchestrator, tasks

    async def close(self):
        """Cleanup resources."""
        await self.comfyui.close()
        if self._kling_client:
            await self._kling_client.close()


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def _async_main():
    """Async CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Art Pipeline Orchestrator")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config path")
    parser.add_argument("--tasks", "-t", required=True, help="Tasks JSON file")
    parser.add_argument("--sequential", action="store_true", default=True, help="Sequential execution")
    
    args = parser.parse_args()
    
    orchestrator, tasks = PipelineOrchestrator.from_task_file(args.tasks, args.config)
    
    try:
        results = await orchestrator.execute_batch(tasks, sequential=args.sequential)
        
        # Print summary
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION SUMMARY")
        print(f"{'='*60}")
        for r in results:
            status = "OK" if r.success else "FAIL"
            print(
                f"  [{status}] {r.task.character_name} | "
                f"imgs={len(r.filtered_images)} | "
                f"vids={len(r.videos_generated)} | "
                f"sim={r.consistency_score:.3f} | "
                f"time={r.duration_seconds:.0f}s"
            )
        print(f"{'='*60}\n")
        
    finally:
        await orchestrator.close()


def main():
    """CLI entry point."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
