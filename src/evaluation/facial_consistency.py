"""
Facial Consistency Evaluation Module
=====================================
Academic Research: Evaluates identity preservation across generated images using
face embedding similarity metrics.

Methodology:
- Extracts facial embeddings using InsightFace (ArcFace backbone)
- Computes pairwise cosine similarity between all faces in a batch
- Generates statistical reports (mean, std, min, max similarity)
- Compares techniques: LoRA-only vs LoRA+PuLID vs LoRA+IP-Adapter

References:
- Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
- Guo et al., "PuLID: Pure and Lightning ID Customization via Contrastive Alignment" (2024)
- Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter" (2023)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from loguru import logger
from scipy.spatial.distance import cosine


@dataclass
class FaceEmbedding:
    """Container for a detected face and its embedding vector."""
    image_path: str
    embedding: np.ndarray
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmark: Optional[np.ndarray] = None


@dataclass
class ConsistencyReport:
    """Statistical report of facial consistency across a batch of images."""
    technique: str
    num_images: int
    num_faces_detected: int
    mean_similarity: float
    std_similarity: float
    min_similarity: float
    max_similarity: float
    median_similarity: float
    pairwise_matrix: np.ndarray = field(repr=False)
    threshold_pass_rate: float = 0.0  # % of pairs above threshold
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Serialize report to dictionary (JSON-compatible)."""
        return {
            "technique": self.technique,
            "num_images": self.num_images,
            "num_faces_detected": self.num_faces_detected,
            "mean_similarity": round(self.mean_similarity, 4),
            "std_similarity": round(self.std_similarity, 4),
            "min_similarity": round(self.min_similarity, 4),
            "max_similarity": round(self.max_similarity, 4),
            "median_similarity": round(self.median_similarity, 4),
            "threshold_pass_rate": round(self.threshold_pass_rate, 4),
            "processing_time_seconds": round(self.processing_time_seconds, 2),
        }


class FacialConsistencyEvaluator:
    """
    Evaluates facial identity consistency across generated images.
    
    Uses InsightFace with ArcFace backbone to extract 512-d embeddings,
    then computes pairwise cosine similarity to measure identity preservation.
    
    Research relevance:
    - Quantifies how well different techniques (LoRA, PuLID, IP-Adapter)
      preserve facial identity across varied prompts and poses.
    - Provides reproducible metrics for academic comparison.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        similarity_threshold: float = 0.85,
        min_face_size: int = 64,
        device: str = "cuda",
    ):
        """
        Initialize the evaluator with InsightFace model.
        
        Args:
            model_name: InsightFace model pack name (buffalo_l recommended for accuracy)
            similarity_threshold: Minimum cosine similarity to consider "same identity"
            min_face_size: Minimum face bbox size in pixels to filter low-quality detections
            device: Compute device ('cuda' or 'cpu')
        """
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        self.device = device
        self._model = None
        self._model_name = model_name
        logger.info(
            f"FacialConsistencyEvaluator initialized | "
            f"model={model_name} | threshold={similarity_threshold} | "
            f"min_face={min_face_size}px"
        )

    def _load_model(self):
        """Lazy-load InsightFace model to avoid VRAM allocation until needed."""
        if self._model is None:
            try:
                import insightface
                from insightface.app import FaceAnalysis
                
                self._model = FaceAnalysis(
                    name=self._model_name,
                    providers=[
                        "CUDAExecutionProvider" if self.device == "cuda"
                        else "CPUExecutionProvider"
                    ],
                )
                self._model.prepare(ctx_id=0 if self.device == "cuda" else -1)
                logger.info(f"InsightFace model '{self._model_name}' loaded successfully")
            except ImportError:
                logger.error("insightface not installed. Run: pip install insightface onnxruntime-gpu")
                raise
            except Exception as e:
                logger.error(f"Failed to load InsightFace model: {e}")
                raise

    def extract_embedding(self, image_path: str | Path) -> Optional[FaceEmbedding]:
        """
        Extract face embedding from a single image.
        
        Uses the largest detected face (by bbox area) if multiple faces present.
        Returns None if no face detected or face too small.
        
        Args:
            image_path: Path to image file
            
        Returns:
            FaceEmbedding or None if no valid face found
        """
        self._load_model()
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to read image: {image_path}")
            return None

        faces = self._model.get(img)
        
        if not faces:
            logger.debug(f"No face detected in: {image_path.name}")
            return None

        # Select largest face by bbox area
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Filter by minimum size
        bbox = largest_face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        if face_width < self.min_face_size or face_height < self.min_face_size:
            logger.debug(
                f"Face too small ({face_width}x{face_height}px) in: {image_path.name}"
            )
            return None

        return FaceEmbedding(
            image_path=str(image_path),
            embedding=largest_face.embedding,
            bbox=tuple(bbox),
            confidence=float(largest_face.det_score),
            landmark=largest_face.landmark_2d_106 if hasattr(largest_face, 'landmark_2d_106') else None,
        )

    def extract_batch_embeddings(
        self, image_paths: list[str | Path]
    ) -> list[FaceEmbedding]:
        """
        Extract embeddings from a batch of images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of FaceEmbedding objects (only for images with valid faces)
        """
        embeddings = []
        for path in image_paths:
            emb = self.extract_embedding(path)
            if emb is not None:
                embeddings.append(emb)

        logger.info(
            f"Extracted {len(embeddings)}/{len(image_paths)} face embeddings"
        )
        return embeddings

    @staticmethod
    def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        
        Returns value in [0, 1] where 1 = identical identity.
        Note: scipy.cosine returns distance, so similarity = 1 - distance.
        """
        return 1.0 - cosine(emb1, emb2)

    def compute_pairwise_matrix(
        self, embeddings: list[FaceEmbedding]
    ) -> np.ndarray:
        """
        Compute NxN pairwise cosine similarity matrix.
        
        Research note: This matrix reveals which image pairs maintain
        identity consistency and which deviate. Useful for identifying
        failure modes (e.g., extreme poses that break identity).
        
        Args:
            embeddings: List of FaceEmbedding objects
            
        Returns:
            NxN numpy array of similarities
        """
        n = len(embeddings)
        matrix = np.ones((n, n), dtype=np.float64)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_cosine_similarity(
                    embeddings[i].embedding,
                    embeddings[j].embedding,
                )
                matrix[i, j] = sim
                matrix[j, i] = sim

        return matrix

    def evaluate_consistency(
        self,
        image_paths: list[str | Path],
        technique: str = "unknown",
    ) -> Optional[ConsistencyReport]:
        """
        Full evaluation pipeline: extract → pairwise similarity → report.
        
        This is the main entry point for evaluating a batch of generated images.
        
        Args:
            image_paths: List of image paths to evaluate
            technique: Label for the generation technique (e.g., "LoRA+PuLID")
            
        Returns:
            ConsistencyReport with statistical metrics, or None if < 2 faces
        """
        start_time = time.time()
        
        embeddings = self.extract_batch_embeddings(image_paths)
        
        if len(embeddings) < 2:
            logger.warning(
                f"Need at least 2 faces for consistency evaluation. "
                f"Got {len(embeddings)} from {len(image_paths)} images."
            )
            return None

        # Compute pairwise matrix
        matrix = self.compute_pairwise_matrix(embeddings)
        
        # Extract upper triangle (excluding diagonal)
        upper_indices = np.triu_indices(len(embeddings), k=1)
        similarities = matrix[upper_indices]
        
        # Compute threshold pass rate
        pass_rate = float(np.mean(similarities >= self.similarity_threshold))
        
        processing_time = time.time() - start_time
        
        report = ConsistencyReport(
            technique=technique,
            num_images=len(image_paths),
            num_faces_detected=len(embeddings),
            mean_similarity=float(np.mean(similarities)),
            std_similarity=float(np.std(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
            median_similarity=float(np.median(similarities)),
            pairwise_matrix=matrix,
            threshold_pass_rate=pass_rate,
            processing_time_seconds=processing_time,
        )
        
        logger.info(
            f"Consistency evaluation complete | technique={technique} | "
            f"mean_sim={report.mean_similarity:.4f} | "
            f"pass_rate={report.threshold_pass_rate:.2%} | "
            f"time={processing_time:.1f}s"
        )
        
        return report

    def compare_techniques(
        self,
        technique_batches: dict[str, list[str | Path]],
    ) -> dict[str, ConsistencyReport]:
        """
        Compare facial consistency across multiple generation techniques.
        
        Research application: Generates a comparative table for the paper
        showing which technique best preserves identity.
        
        Args:
            technique_batches: Dict mapping technique name → list of image paths
            
        Returns:
            Dict mapping technique name → ConsistencyReport
            
        Example:
            results = evaluator.compare_techniques({
                "LoRA_only": glob("output/lora_only/*.png"),
                "LoRA+PuLID": glob("output/lora_pulid/*.png"),
                "LoRA+IP-Adapter": glob("output/lora_ipadapter/*.png"),
            })
        """
        results = {}
        
        for technique, paths in technique_batches.items():
            logger.info(f"Evaluating technique: {technique} ({len(paths)} images)")
            report = self.evaluate_consistency(paths, technique=technique)
            if report:
                results[technique] = report

        # Log comparative summary
        if results:
            logger.info("=" * 60)
            logger.info("COMPARATIVE SUMMARY")
            logger.info("=" * 60)
            for name, report in sorted(
                results.items(), key=lambda x: x[1].mean_similarity, reverse=True
            ):
                logger.info(
                    f"  {name:20s} | mean={report.mean_similarity:.4f} | "
                    f"pass={report.threshold_pass_rate:.2%}"
                )
            logger.info("=" * 60)

        return results

    def generate_report_json(
        self,
        results: dict[str, ConsistencyReport],
        output_path: str | Path,
    ) -> Path:
        """
        Export comparative results as JSON report.
        
        Args:
            results: Dict of technique → ConsistencyReport
            output_path: Path for the output JSON file
            
        Returns:
            Path to the generated report file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "metadata": {
                "evaluation_type": "facial_consistency",
                "model": self._model_name,
                "similarity_threshold": self.similarity_threshold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": {name: r.to_dict() for name, r in results.items()},
            "ranking": sorted(
                results.keys(),
                key=lambda k: results[k].mean_similarity,
                reverse=True,
            ),
        }
        
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Report saved to: {output_path}")
        return output_path

    def generate_latex_table(
        self, results: dict[str, ConsistencyReport]
    ) -> str:
        """
        Generate LaTeX table for academic paper.
        
        Output format suitable for direct inclusion in a research paper.
        """
        header = (
            "\\begin{table}[h]\n"
            "\\centering\n"
            "\\caption{Facial Consistency Comparison Across Techniques}\n"
            "\\label{tab:facial_consistency}\n"
            "\\begin{tabular}{lcccccc}\n"
            "\\hline\n"
            "\\textbf{Technique} & \\textbf{Images} & \\textbf{Mean Sim.} & "
            "\\textbf{Std} & \\textbf{Min} & \\textbf{Max} & "
            "\\textbf{Pass Rate} \\\\\n"
            "\\hline\n"
        )
        
        rows = []
        for name, r in sorted(
            results.items(), key=lambda x: x[1].mean_similarity, reverse=True
        ):
            rows.append(
                f"{name} & {r.num_faces_detected} & "
                f"{r.mean_similarity:.4f} & {r.std_similarity:.4f} & "
                f"{r.min_similarity:.4f} & {r.max_similarity:.4f} & "
                f"{r.threshold_pass_rate:.2%} \\\\"
            )
        
        footer = (
            "\n\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}"
        )
        
        return header + "\n".join(rows) + footer


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for facial consistency evaluation."""
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(
        description="Evaluate facial consistency across generated images"
    )
    parser.add_argument(
        "--input-dir", "-i", required=True,
        help="Directory containing generated images"
    )
    parser.add_argument(
        "--technique", "-t", default="unknown",
        help="Label for the generation technique"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85,
        help="Similarity threshold for pass/fail (default: 0.85)"
    )
    parser.add_argument(
        "--output", "-o", default="output/reports/consistency_report.json",
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default="cuda",
        help="Compute device"
    )
    
    args = parser.parse_args()
    
    # Gather images
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(str(Path(args.input_dir) / ext)))
    
    if not image_paths:
        logger.error(f"No images found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(image_paths)} images in {args.input_dir}")
    
    # Evaluate
    evaluator = FacialConsistencyEvaluator(
        similarity_threshold=args.threshold,
        device=args.device,
    )
    
    report = evaluator.evaluate_consistency(image_paths, technique=args.technique)
    
    if report:
        # Save report
        evaluator.generate_report_json(
            {args.technique: report}, args.output
        )
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"FACIAL CONSISTENCY REPORT: {args.technique}")
        print(f"{'='*50}")
        print(f"  Images processed: {report.num_images}")
        print(f"  Faces detected:   {report.num_faces_detected}")
        print(f"  Mean similarity:  {report.mean_similarity:.4f}")
        print(f"  Std deviation:    {report.std_similarity:.4f}")
        print(f"  Min similarity:   {report.min_similarity:.4f}")
        print(f"  Max similarity:   {report.max_similarity:.4f}")
        print(f"  Pass rate (>{args.threshold}): {report.threshold_pass_rate:.2%}")
        print(f"  Processing time:  {report.processing_time_seconds:.1f}s")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
