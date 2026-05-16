"""
Metrics & Benchmarking Module
==============================
Comprehensive evaluation framework for AI-generated content quality,
temporal consistency, and influencer pipeline performance metrics.

Provides quantitative benchmarks using established perceptual quality metrics
(LPIPS, PSNR, SSIM, FID) alongside custom temporal consistency metrics for
video content and business KPIs for the influencer automation pipeline.

References:
- Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric" (CVPR 2018)
- Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (NeurIPS 2017)
- Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (IEEE TIP 2004)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class ImageQualityMetrics:
    """
    Standard image quality metrics for evaluating generated content.

    Attributes:
        lpips: Learned Perceptual Image Patch Similarity (lower = better, 0-1).
        psnr: Peak Signal-to-Noise Ratio in dB (higher = better).
        ssim: Structural Similarity Index (higher = better, 0-1).
        fid: Frechet Inception Distance (lower = better).
    """

    lpips: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    fid: float = 0.0

    def to_dict(self) -> dict:
        return {
            "lpips": round(self.lpips, 4),
            "psnr": round(self.psnr, 2),
            "ssim": round(self.ssim, 4),
            "fid": round(self.fid, 2),
        }


@dataclass
class TemporalMetrics:
    """
    Temporal consistency metrics for video/animation content.

    Evaluates frame-to-frame coherence to detect flickering, temporal
    artifacts, and identity drift across video frames.

    Attributes:
        frame_count: Total number of frames analyzed.
        avg_frame_similarity: Mean cosine similarity between consecutive frames.
        flicker_score: Measure of frame-to-frame luminance instability (lower = better).
        temporal_consistency: Overall temporal coherence score (0-1, higher = better).
    """

    frame_count: int = 0
    avg_frame_similarity: float = 0.0
    flicker_score: float = 0.0
    temporal_consistency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame_count": self.frame_count,
            "avg_frame_similarity": round(self.avg_frame_similarity, 4),
            "flicker_score": round(self.flicker_score, 4),
            "temporal_consistency": round(self.temporal_consistency, 4),
        }


@dataclass
class InfluencerKPIs:
    """
    Business and pipeline KPIs for the AI influencer system.

    Tracks operational metrics that determine the viability and efficiency
    of the automated content pipeline.

    Attributes:
        total_posts: Total number of posts generated and published.
        posts_per_week: Average posting frequency.
        content_types: Distribution of content types (image, reel, carousel).
        avg_consistency: Average facial consistency score across all posts.
        success_rate: Percentage of posts successfully published.
        gen_time_seconds: Average generation time per content piece.
        cost_per_video: Estimated cost per video generation (API costs).
        platform_metrics: Per-platform engagement metrics.
        character_distribution: Posts per character.
    """

    total_posts: int = 0
    posts_per_week: float = 0.0
    content_types: dict[str, int] = field(default_factory=dict)
    avg_consistency: float = 0.0
    success_rate: float = 0.0
    gen_time_seconds: float = 0.0
    cost_per_video: float = 0.0
    platform_metrics: dict[str, dict] = field(default_factory=dict)
    character_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_posts": self.total_posts,
            "posts_per_week": round(self.posts_per_week, 1),
            "content_types": self.content_types,
            "avg_consistency": round(self.avg_consistency, 4),
            "success_rate": round(self.success_rate, 4),
            "gen_time_seconds": round(self.gen_time_seconds, 2),
            "cost_per_video": round(self.cost_per_video, 4),
            "platform_metrics": self.platform_metrics,
            "character_distribution": self.character_distribution,
        }



class MetricsDashboard:
    """
    Comprehensive metrics computation and reporting dashboard.

    Provides methods for computing all quality metrics, running technique
    benchmarks, and generating publication-ready reports, LaTeX tables,
    and matplotlib visualization charts.

    Example:
        >>> dashboard = MetricsDashboard(output_dir="output/metrics")
        >>> quality = dashboard.compute_lpips(generated_images, reference_images)
        >>> report = dashboard.generate_full_report(results)
    """

    def __init__(self, output_dir: str | Path = "output/metrics", device: str = "cuda"):
        """
        Initialize the metrics dashboard.

        Args:
            output_dir: Directory for saving reports and charts.
            device: Computation device ("cuda" or "cpu").
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._lpips_model = None
        logger.info(f"MetricsDashboard initialized | output_dir={output_dir} | device={device}")

    def _load_lpips_model(self):
        """Lazy-load the LPIPS model."""
        if self._lpips_model is None:
            import lpips
            self._lpips_model = lpips.LPIPS(net="alex").to(self.device)
            logger.info("LPIPS model (AlexNet) loaded")

    def compute_lpips(
        self,
        generated_paths: list[str | Path],
        reference_paths: list[str | Path],
    ) -> list[float]:
        """
        Compute LPIPS perceptual distance between generated and reference images.

        LPIPS measures perceptual similarity using deep features from a pretrained
        network. Lower values indicate more perceptually similar images.

        Args:
            generated_paths: Paths to generated images.
            reference_paths: Paths to reference images (must match length).

        Returns:
            List of LPIPS scores for each image pair.

        Raises:
            ValueError: If input lists have different lengths.
        """
        if len(generated_paths) != len(reference_paths):
            raise ValueError(
                f"Mismatched lengths: {len(generated_paths)} generated vs {len(reference_paths)} reference"
            )

        self._load_lpips_model()
        import torch
        from torchvision import transforms
        from PIL import Image

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        scores = []
        for gen_path, ref_path in zip(generated_paths, reference_paths):
            try:
                gen_img = transform(Image.open(gen_path).convert("RGB")).unsqueeze(0).to(self.device)
                ref_img = transform(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    score = self._lpips_model(gen_img, ref_img).item()
                scores.append(score)
            except Exception as e:
                logger.warning(f"LPIPS computation failed for {gen_path}: {e}")
                scores.append(float("nan"))

        logger.info(f"LPIPS computed: mean={np.nanmean(scores):.4f} over {len(scores)} pairs")
        return scores

    def compute_ssim_batch(
        self,
        generated_paths: list[str | Path],
        reference_paths: list[str | Path],
    ) -> list[float]:
        """
        Compute SSIM (Structural Similarity Index) for a batch of image pairs.

        SSIM measures structural similarity considering luminance, contrast,
        and structure. Values range from -1 to 1, with 1 being identical.

        Args:
            generated_paths: Paths to generated images.
            reference_paths: Paths to reference images.

        Returns:
            List of SSIM scores for each pair.
        """
        from skimage.metrics import structural_similarity
        import cv2

        scores = []
        for gen_path, ref_path in zip(generated_paths, reference_paths):
            try:
                gen_img = cv2.imread(str(gen_path), cv2.IMREAD_GRAYSCALE)
                ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

                if gen_img is None or ref_img is None:
                    scores.append(float("nan"))
                    continue

                # Resize to match
                h, w = min(gen_img.shape[0], ref_img.shape[0]), min(gen_img.shape[1], ref_img.shape[1])
                gen_img = cv2.resize(gen_img, (w, h))
                ref_img = cv2.resize(ref_img, (w, h))

                score = structural_similarity(gen_img, ref_img)
                scores.append(float(score))
            except Exception as e:
                logger.warning(f"SSIM computation failed for {gen_path}: {e}")
                scores.append(float("nan"))

        logger.info(f"SSIM computed: mean={np.nanmean(scores):.4f} over {len(scores)} pairs")
        return scores

    def compute_temporal_consistency(
        self, frame_dir: str | Path, method: str = "ssim"
    ) -> TemporalMetrics:
        """
        Compute temporal consistency metrics for a sequence of video frames.

        Analyzes frame-to-frame similarity to detect flickering and temporal
        artifacts common in AI-generated video content.

        Args:
            frame_dir: Directory containing sequential frame images.
            method: Similarity method ("ssim" or "histogram").

        Returns:
            TemporalMetrics with all temporal consistency measurements.
        """
        import cv2
        from glob import glob

        frame_dir = Path(frame_dir)
        frame_paths = sorted(glob(str(frame_dir / "*.png")) + glob(str(frame_dir / "*.jpg")))

        if len(frame_paths) < 2:
            logger.warning(f"Not enough frames in {frame_dir} for temporal analysis")
            return TemporalMetrics(frame_count=len(frame_paths))

        similarities = []
        luminance_diffs = []

        for i in range(len(frame_paths) - 1):
            frame_a = cv2.imread(frame_paths[i])
            frame_b = cv2.imread(frame_paths[i + 1])

            if frame_a is None or frame_b is None:
                continue

            # Resize to match
            h = min(frame_a.shape[0], frame_b.shape[0])
            w = min(frame_a.shape[1], frame_b.shape[1])
            frame_a = cv2.resize(frame_a, (w, h))
            frame_b = cv2.resize(frame_b, (w, h))

            if method == "ssim":
                from skimage.metrics import structural_similarity
                gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
                gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
                sim = structural_similarity(gray_a, gray_b)
            else:
                # Histogram comparison
                hist_a = cv2.calcHist([frame_a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist_b = cv2.calcHist([frame_b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                cv2.normalize(hist_a, hist_a)
                cv2.normalize(hist_b, hist_b)
                sim = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

            similarities.append(float(sim))

            # Luminance difference for flicker detection
            lum_a = np.mean(cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY))
            lum_b = np.mean(cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY))
            luminance_diffs.append(abs(lum_a - lum_b))

        avg_similarity = float(np.mean(similarities)) if similarities else 0.0
        flicker_score = float(np.std(luminance_diffs)) if luminance_diffs else 0.0
        temporal_consistency = max(0.0, min(1.0, avg_similarity * (1.0 - flicker_score / 50.0)))

        metrics = TemporalMetrics(
            frame_count=len(frame_paths),
            avg_frame_similarity=avg_similarity,
            flicker_score=flicker_score,
            temporal_consistency=temporal_consistency,
        )

        logger.info(
            f"Temporal metrics: frames={metrics.frame_count} | "
            f"consistency={metrics.temporal_consistency:.4f} | "
            f"flicker={metrics.flicker_score:.4f}"
        )
        return metrics

    def run_technique_benchmark(
        self,
        techniques: dict[str, dict[str, list[str | Path]]],
        reference_paths: Optional[list[str | Path]] = None,
    ) -> dict[str, ImageQualityMetrics]:
        """
        Run a full quality benchmark comparing multiple generation techniques.

        Args:
            techniques: Dict mapping technique names to dicts with "generated"
                and optionally "reference" image path lists.
            reference_paths: Global reference paths (used if technique-specific
                references are not provided).

        Returns:
            Dictionary mapping technique names to their ImageQualityMetrics.
        """
        results: dict[str, ImageQualityMetrics] = {}

        for technique_name, paths in techniques.items():
            generated = paths.get("generated", [])
            refs = paths.get("reference", reference_paths or [])

            if not generated:
                logger.warning(f"No generated images for technique: {technique_name}")
                continue

            metrics = ImageQualityMetrics()

            # Compute LPIPS if references available
            if refs and len(refs) == len(generated):
                try:
                    lpips_scores = self.compute_lpips(generated, refs)
                    metrics.lpips = float(np.nanmean(lpips_scores))
                except Exception as e:
                    logger.warning(f"LPIPS failed for {technique_name}: {e}")

            # Compute SSIM if references available
            if refs and len(refs) == len(generated):
                try:
                    ssim_scores = self.compute_ssim_batch(generated, refs)
                    metrics.ssim = float(np.nanmean(ssim_scores))
                except Exception as e:
                    logger.warning(f"SSIM failed for {technique_name}: {e}")

            # Compute PSNR
            if refs and len(refs) == len(generated):
                try:
                    psnr_scores = self._compute_psnr_batch(generated, refs)
                    metrics.psnr = float(np.nanmean(psnr_scores))
                except Exception as e:
                    logger.warning(f"PSNR failed for {technique_name}: {e}")

            results[technique_name] = metrics
            logger.info(
                f"Benchmark [{technique_name}]: LPIPS={metrics.lpips:.4f} | "
                f"SSIM={metrics.ssim:.4f} | PSNR={metrics.psnr:.2f}"
            )

        return results

    def _compute_psnr_batch(
        self, generated_paths: list[str | Path], reference_paths: list[str | Path]
    ) -> list[float]:
        """Compute PSNR for a batch of image pairs."""
        import cv2

        scores = []
        for gen_path, ref_path in zip(generated_paths, reference_paths):
            try:
                gen_img = cv2.imread(str(gen_path))
                ref_img = cv2.imread(str(ref_path))

                if gen_img is None or ref_img is None:
                    scores.append(float("nan"))
                    continue

                h = min(gen_img.shape[0], ref_img.shape[0])
                w = min(gen_img.shape[1], ref_img.shape[1])
                gen_img = cv2.resize(gen_img, (w, h))
                ref_img = cv2.resize(ref_img, (w, h))

                mse = np.mean((gen_img.astype(float) - ref_img.astype(float)) ** 2)
                if mse == 0:
                    scores.append(float("inf"))
                else:
                    psnr = 10 * np.log10(255.0**2 / mse)
                    scores.append(float(psnr))
            except Exception as e:
                logger.warning(f"PSNR computation failed: {e}")
                scores.append(float("nan"))

        return scores

    def compute_influencer_kpis(
        self,
        publishing_history: list[dict],
        consistency_scores: list[float],
        generation_times: list[float],
        video_costs: Optional[list[float]] = None,
    ) -> InfluencerKPIs:
        """
        Compute business KPIs for the AI influencer pipeline.

        Args:
            publishing_history: List of publishing result dictionaries.
            consistency_scores: Facial consistency scores per post.
            generation_times: Generation time per content piece (seconds).
            video_costs: API costs per video generation.

        Returns:
            InfluencerKPIs with all business metrics.
        """
        total_posts = len(publishing_history)
        successes = sum(1 for p in publishing_history if p.get("success", False))

        # Content type distribution
        content_types: dict[str, int] = {}
        platform_metrics: dict[str, dict] = {}
        character_dist: dict[str, int] = {}

        for post in publishing_history:
            # Content types
            ct = post.get("content_type", "unknown")
            content_types[ct] = content_types.get(ct, 0) + 1

            # Platform
            platform = post.get("platform", "unknown")
            if platform not in platform_metrics:
                platform_metrics[platform] = {"total": 0, "success": 0}
            platform_metrics[platform]["total"] += 1
            if post.get("success", False):
                platform_metrics[platform]["success"] += 1

            # Character
            char = post.get("character", "unknown")
            character_dist[char] = character_dist.get(char, 0) + 1

        # Calculate time span for posts/week
        timestamps = [p.get("timestamp", 0) for p in publishing_history if p.get("timestamp")]
        if len(timestamps) >= 2:
            time_span_weeks = max((max(timestamps) - min(timestamps)) / (7 * 24 * 3600), 1)
            posts_per_week = total_posts / time_span_weeks
        else:
            posts_per_week = total_posts

        kpis = InfluencerKPIs(
            total_posts=total_posts,
            posts_per_week=round(posts_per_week, 1),
            content_types=content_types,
            avg_consistency=float(np.mean(consistency_scores)) if consistency_scores else 0.0,
            success_rate=successes / max(total_posts, 1),
            gen_time_seconds=float(np.mean(generation_times)) if generation_times else 0.0,
            cost_per_video=float(np.mean(video_costs)) if video_costs else 0.0,
            platform_metrics=platform_metrics,
            character_distribution=character_dist,
        )

        logger.info(
            f"KPIs computed: posts={kpis.total_posts} | "
            f"success={kpis.success_rate:.2%} | "
            f"consistency={kpis.avg_consistency:.4f}"
        )
        return kpis



    def generate_full_report(
        self,
        quality_results: dict[str, ImageQualityMetrics],
        temporal_results: Optional[dict[str, TemporalMetrics]] = None,
        kpis: Optional[InfluencerKPIs] = None,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Generate a comprehensive JSON report combining all metrics.

        Args:
            quality_results: Image quality metrics per technique.
            temporal_results: Temporal metrics per technique (optional).
            kpis: Influencer pipeline KPIs (optional).
            output_path: Report output path (defaults to output_dir/full_report.json).

        Returns:
            Path to the saved report file.
        """
        if output_path is None:
            output_path = self.output_dir / "full_report.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "metadata": {
                "report_type": "comprehensive_benchmark",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": self.device,
            },
            "image_quality": {
                name: metrics.to_dict() for name, metrics in quality_results.items()
            },
        }

        if temporal_results:
            report["temporal_consistency"] = {
                name: metrics.to_dict() for name, metrics in temporal_results.items()
            }

        if kpis:
            report["influencer_kpis"] = kpis.to_dict()

        # Rankings
        if quality_results:
            report["rankings"] = {
                "by_lpips": sorted(quality_results.keys(), key=lambda k: quality_results[k].lpips),
                "by_ssim": sorted(quality_results.keys(), key=lambda k: quality_results[k].ssim, reverse=True),
                "by_psnr": sorted(quality_results.keys(), key=lambda k: quality_results[k].psnr, reverse=True),
            }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Full report saved: {output_path}")
        return output_path

    def generate_latex_tables(
        self,
        quality_results: dict[str, ImageQualityMetrics],
        temporal_results: Optional[dict[str, TemporalMetrics]] = None,
        kpis: Optional[InfluencerKPIs] = None,
    ) -> str:
        """
        Generate LaTeX tables for academic paper inclusion.

        Produces publication-ready tables comparing techniques across
        all metrics, suitable for direct inclusion in a research paper.

        Args:
            quality_results: Image quality metrics per technique.
            temporal_results: Temporal metrics (optional).
            kpis: Pipeline KPIs (optional).

        Returns:
            Complete LaTeX table string.
        """
        latex = ""

        # Image Quality Table
        latex += "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Image Quality Metrics Comparison}\n"
        latex += "\\label{tab:image_quality}\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\hline\n"
        latex += "\\textbf{Technique} & \\textbf{LPIPS}$\\downarrow$ & \\textbf{SSIM}$\\uparrow$ & \\textbf{PSNR}$\\uparrow$ & \\textbf{FID}$\\downarrow$ \\\\\n"
        latex += "\\hline\n"

        for name, m in sorted(quality_results.items(), key=lambda x: x[1].ssim, reverse=True):
            latex += f"{name} & {m.lpips:.4f} & {m.ssim:.4f} & {m.psnr:.2f} & {m.fid:.2f} \\\\\n"

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n\n"

        # Temporal Consistency Table
        if temporal_results:
            latex += "\\begin{table}[h]\n"
            latex += "\\centering\n"
            latex += "\\caption{Temporal Consistency Metrics}\n"
            latex += "\\label{tab:temporal}\n"
            latex += "\\begin{tabular}{lcccc}\n"
            latex += "\\hline\n"
            latex += "\\textbf{Technique} & \\textbf{Frames} & \\textbf{Avg Sim.} & \\textbf{Flicker}$\\downarrow$ & \\textbf{Consistency}$\\uparrow$ \\\\\n"
            latex += "\\hline\n"

            for name, m in sorted(temporal_results.items(), key=lambda x: x[1].temporal_consistency, reverse=True):
                latex += f"{name} & {m.frame_count} & {m.avg_frame_similarity:.4f} & {m.flicker_score:.4f} & {m.temporal_consistency:.4f} \\\\\n"

            latex += "\\hline\n"
            latex += "\\end{tabular}\n"
            latex += "\\end{table}\n\n"

        # KPI Table
        if kpis:
            latex += "\\begin{table}[h]\n"
            latex += "\\centering\n"
            latex += "\\caption{AI Influencer Pipeline KPIs}\n"
            latex += "\\label{tab:kpis}\n"
            latex += "\\begin{tabular}{lc}\n"
            latex += "\\hline\n"
            latex += "\\textbf{Metric} & \\textbf{Value} \\\\\n"
            latex += "\\hline\n"
            latex += f"Total Posts & {kpis.total_posts} \\\\\n"
            latex += f"Posts/Week & {kpis.posts_per_week:.1f} \\\\\n"
            latex += f"Avg. Consistency & {kpis.avg_consistency:.4f} \\\\\n"
            latex += f"Success Rate & {kpis.success_rate:.2%} \\\\\n"
            latex += f"Avg. Gen Time (s) & {kpis.gen_time_seconds:.2f} \\\\\n"
            latex += f"Cost/Video (\\$) & {kpis.cost_per_video:.4f} \\\\\n"
            latex += "\\hline\n"
            latex += "\\end{tabular}\n"
            latex += "\\end{table}\n"

        return latex

    def generate_matplotlib_charts(
        self,
        quality_results: dict[str, ImageQualityMetrics],
        temporal_results: Optional[dict[str, TemporalMetrics]] = None,
        output_dir: Optional[str | Path] = None,
    ) -> list[Path]:
        """
        Generate matplotlib visualization charts for metrics comparison.

        Creates bar charts and radar plots comparing techniques across
        all quality and temporal metrics.

        Args:
            quality_results: Image quality metrics per technique.
            temporal_results: Temporal metrics (optional).
            output_dir: Directory to save chart images.

        Returns:
            List of paths to generated chart images.
        """
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        if output_dir is None:
            output_dir = self.output_dir / "charts"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_charts: list[Path] = []
        techniques = list(quality_results.keys())

        # Chart 1: SSIM Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ssim_values = [quality_results[t].ssim for t in techniques]
        bars = ax.bar(techniques, ssim_values, color="steelblue", edgecolor="navy", alpha=0.8)
        ax.set_xlabel("Technique", fontsize=12)
        ax.set_ylabel("SSIM Score", fontsize=12)
        ax.set_title("Structural Similarity (SSIM) Comparison", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="Target (0.9)")
        ax.legend()
        for bar, val in zip(bars, ssim_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}", ha="center", fontsize=9)
        plt.tight_layout()
        chart_path = output_dir / "ssim_comparison.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        saved_charts.append(chart_path)

        # Chart 2: LPIPS Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        lpips_values = [quality_results[t].lpips for t in techniques]
        bars = ax.bar(techniques, lpips_values, color="coral", edgecolor="darkred", alpha=0.8)
        ax.set_xlabel("Technique", fontsize=12)
        ax.set_ylabel("LPIPS Score (lower is better)", fontsize=12)
        ax.set_title("Perceptual Similarity (LPIPS) Comparison", fontsize=14, fontweight="bold")
        ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5, label="Target (<0.1)")
        ax.legend()
        for bar, val in zip(bars, lpips_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.4f}", ha="center", fontsize=9)
        plt.tight_layout()
        chart_path = output_dir / "lpips_comparison.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        saved_charts.append(chart_path)

        # Chart 3: Multi-metric grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(techniques))
        width = 0.25
        # Normalize metrics to 0-1 range for comparison
        ssim_norm = [quality_results[t].ssim for t in techniques]
        lpips_norm = [1.0 - quality_results[t].lpips for t in techniques]  # Invert LPIPS
        psnr_norm = [min(quality_results[t].psnr / 50.0, 1.0) for t in techniques]

        ax.bar(x - width, ssim_norm, width, label="SSIM", color="steelblue", alpha=0.8)
        ax.bar(x, lpips_norm, width, label="1-LPIPS", color="coral", alpha=0.8)
        ax.bar(x + width, psnr_norm, width, label="PSNR/50", color="forestgreen", alpha=0.8)

        ax.set_xlabel("Technique", fontsize=12)
        ax.set_ylabel("Normalized Score (higher is better)", fontsize=12)
        ax.set_title("Multi-Metric Quality Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(techniques, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        chart_path = output_dir / "multi_metric_comparison.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        saved_charts.append(chart_path)

        # Chart 4: Temporal consistency (if available)
        if temporal_results:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            temp_techniques = list(temporal_results.keys())

            # Consistency scores
            consistency_values = [temporal_results[t].temporal_consistency for t in temp_techniques]
            axes[0].bar(temp_techniques, consistency_values, color="mediumpurple", edgecolor="indigo", alpha=0.8)
            axes[0].set_ylabel("Temporal Consistency")
            axes[0].set_title("Temporal Consistency Score")
            axes[0].set_ylim(0, 1)

            # Flicker scores
            flicker_values = [temporal_results[t].flicker_score for t in temp_techniques]
            axes[1].bar(temp_techniques, flicker_values, color="orange", edgecolor="darkorange", alpha=0.8)
            axes[1].set_ylabel("Flicker Score (lower is better)")
            axes[1].set_title("Flicker Detection Score")

            plt.tight_layout()
            chart_path = output_dir / "temporal_metrics.png"
            plt.savefig(chart_path, dpi=150)
            plt.close()
            saved_charts.append(chart_path)

        logger.info(f"Generated {len(saved_charts)} charts in {output_dir}")
        return saved_charts

    def generate_training_loss_chart(
        self,
        loss_data: dict[str, dict[str, list]],
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """
        Generate training loss curve chart for LoRA training experiments.

        Creates a multi-line chart showing loss trajectories for different
        training configurations, useful for comparing hyperparameter choices.

        Args:
            loss_data: Dict mapping experiment names to dicts with "steps" and "losses" lists.
            output_path: Output path for the chart image.

        Returns:
            Path to the saved chart image.
        """
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")

        if output_path is None:
            output_path = self.output_dir / "charts" / "training_loss.png"
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(loss_data)))

        for (name, data), color in zip(loss_data.items(), colors):
            steps = data.get("steps", [])
            losses = data.get("losses", [])

            if not steps or not losses:
                continue

            # Plot raw loss (faded)
            ax.plot(steps, losses, alpha=0.2, color=color, linewidth=0.5)

            # Plot smoothed loss
            window = max(1, len(losses) // 50)
            if window > 1:
                smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
                smoothed_steps = steps[window - 1:]
                ax.plot(smoothed_steps, smoothed, color=color, linewidth=2, label=name)
            else:
                ax.plot(steps, losses, color=color, linewidth=2, label=name)

        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("LoRA Training Loss Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Training loss chart saved: {output_path}")
        return output_path


# ==============================================================================
# CLI Interface
# ==============================================================================


def main():
    """CLI entry point for metrics and benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Content Metrics & Benchmarking Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with charts
  python benchmark.py --input-dir output/generated --reference-dir datasets/reference --charts

  # Generate LaTeX tables
  python benchmark.py --input-dir output/generated --latex --output report.tex

  # Temporal analysis on video frames
  python benchmark.py --frames-dir output/frames --temporal
        """,
    )

    parser.add_argument("--input-dir", "-i", type=str, help="Directory with generated images")
    parser.add_argument("--reference-dir", "-r", type=str, help="Directory with reference images")
    parser.add_argument("--frames-dir", type=str, help="Directory with video frames for temporal analysis")
    parser.add_argument("--output-dir", "-o", type=str, default="output/metrics", help="Output directory")
    parser.add_argument("--charts", action="store_true", help="Generate matplotlib charts")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Computation device")
    parser.add_argument("--report", type=str, default=None, help="Custom report output path")

    args = parser.parse_args()

    dashboard = MetricsDashboard(output_dir=args.output_dir, device=args.device)

    quality_results: dict[str, ImageQualityMetrics] = {}
    temporal_results: dict[str, TemporalMetrics] = {}

    # Run image quality benchmark
    if args.input_dir:
        from glob import glob

        input_dir = Path(args.input_dir)

        # Look for technique subdirectories
        technique_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

        if technique_dirs:
            techniques = {}
            for tech_dir in technique_dirs:
                images = sorted(glob(str(tech_dir / "*.png")) + glob(str(tech_dir / "*.jpg")))
                if images:
                    techniques[tech_dir.name] = {"generated": images}

                    # Match with references if available
                    if args.reference_dir:
                        ref_dir = Path(args.reference_dir)
                        refs = sorted(glob(str(ref_dir / "*.png")) + glob(str(ref_dir / "*.jpg")))
                        if refs:
                            techniques[tech_dir.name]["reference"] = refs[:len(images)]

            if techniques:
                quality_results = dashboard.run_technique_benchmark(techniques)
        else:
            logger.info("No technique subdirectories found, treating input as single technique")
            images = sorted(glob(str(input_dir / "*.png")) + glob(str(input_dir / "*.jpg")))
            if images and args.reference_dir:
                refs = sorted(glob(str(Path(args.reference_dir) / "*.png")) + glob(str(Path(args.reference_dir) / "*.jpg")))
                if refs:
                    quality_results = dashboard.run_technique_benchmark(
                        {"default": {"generated": images, "reference": refs[:len(images)]}}
                    )

    # Run temporal analysis
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        if frames_dir.is_dir():
            # Check for technique subdirectories
            subdirs = [d for d in frames_dir.iterdir() if d.is_dir()]
            if subdirs:
                for subdir in subdirs:
                    temporal_results[subdir.name] = dashboard.compute_temporal_consistency(subdir)
            else:
                temporal_results["default"] = dashboard.compute_temporal_consistency(frames_dir)

    # Generate outputs
    if quality_results or temporal_results:
        # JSON report
        report_path = args.report or str(Path(args.output_dir) / "full_report.json")
        dashboard.generate_full_report(quality_results, temporal_results, output_path=report_path)

        # Charts
        if args.charts and quality_results:
            dashboard.generate_matplotlib_charts(quality_results, temporal_results)
            print("  Charts generated in output/metrics/charts/")

        # LaTeX
        if args.latex:
            latex_content = dashboard.generate_latex_tables(quality_results, temporal_results)
            latex_path = Path(args.output_dir) / "tables.tex"
            with open(latex_path, "w") as f:
                f.write(latex_content)
            print(f"  LaTeX tables saved: {latex_path}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"  BENCHMARK RESULTS")
        print(f"{'='*60}")
        for name, m in quality_results.items():
            print(f"\n  [{name}]")
            print(f"    LPIPS: {m.lpips:.4f}  |  SSIM: {m.ssim:.4f}  |  PSNR: {m.psnr:.2f} dB")
        if temporal_results:
            print(f"\n  --- Temporal Metrics ---")
            for name, m in temporal_results.items():
                print(f"  [{name}] Consistency: {m.temporal_consistency:.4f} | Flicker: {m.flicker_score:.4f}")
        print(f"\n{'='*60}")
        print(f"  Report: {report_path}")
        print(f"{'='*60}\n")
    else:
        logger.warning("No results to report. Check input paths.")
        print("No images found. Use --input-dir to specify generated images directory.")


if __name__ == "__main__":
    main()
