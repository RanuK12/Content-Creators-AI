"""
Facial Consistency Evaluation Module
=====================================
Academic Research: Evaluates identity preservation across generated images using
face embedding similarity metrics.

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
    bbox: tuple[int, int, int, int]
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
    threshold_pass_rate: float = 0.0
    processing_time_seconds: float = 0.0

    def to_dict(self) -> dict:
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
    then computes pairwise cosine similarity.
    """

    def __init__(self, model_name: str = "buffalo_l", similarity_threshold: float = 0.85, min_face_size: int = 64, device: str = "cuda"):
        self.similarity_threshold = similarity_threshold
        self.min_face_size = min_face_size
        self.device = device
        self._model = None
        self._model_name = model_name
        logger.info(f"FacialConsistencyEvaluator initialized | model={model_name} | threshold={similarity_threshold}")

    def _load_model(self):
        if self._model is None:
            import insightface
            from insightface.app import FaceAnalysis
            self._model = FaceAnalysis(name=self._model_name, providers=["CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"])
            self._model.prepare(ctx_id=0 if self.device == "cuda" else -1)
            logger.info(f"InsightFace model '{self._model_name}' loaded")

    def extract_embedding(self, image_path: str | Path) -> Optional[FaceEmbedding]:
        self._load_model()
        image_path = Path(image_path)
        if not image_path.exists():
            return None
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        faces = self._model.get(img)
        if not faces:
            return None
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = largest_face.bbox.astype(int)
        if (bbox[2] - bbox[0]) < self.min_face_size or (bbox[3] - bbox[1]) < self.min_face_size:
            return None
        return FaceEmbedding(image_path=str(image_path), embedding=largest_face.embedding, bbox=tuple(bbox), confidence=float(largest_face.det_score))

    def extract_batch_embeddings(self, image_paths: list[str | Path]) -> list[FaceEmbedding]:
        embeddings = []
        for path in image_paths:
            emb = self.extract_embedding(path)
            if emb is not None:
                embeddings.append(emb)
        logger.info(f"Extracted {len(embeddings)}/{len(image_paths)} face embeddings")
        return embeddings

    @staticmethod
    def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        return 1.0 - cosine(emb1, emb2)

    def compute_pairwise_matrix(self, embeddings: list[FaceEmbedding]) -> np.ndarray:
        n = len(embeddings)
        matrix = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.compute_cosine_similarity(embeddings[i].embedding, embeddings[j].embedding)
                matrix[i, j] = sim
                matrix[j, i] = sim
        return matrix

    def evaluate_consistency(self, image_paths: list[str | Path], technique: str = "unknown") -> Optional[ConsistencyReport]:
        start_time = time.time()
        embeddings = self.extract_batch_embeddings(image_paths)
        if len(embeddings) < 2:
            return None
        matrix = self.compute_pairwise_matrix(embeddings)
        upper_indices = np.triu_indices(len(embeddings), k=1)
        similarities = matrix[upper_indices]
        pass_rate = float(np.mean(similarities >= self.similarity_threshold))
        processing_time = time.time() - start_time
        report = ConsistencyReport(
            technique=technique, num_images=len(image_paths), num_faces_detected=len(embeddings),
            mean_similarity=float(np.mean(similarities)), std_similarity=float(np.std(similarities)),
            min_similarity=float(np.min(similarities)), max_similarity=float(np.max(similarities)),
            median_similarity=float(np.median(similarities)), pairwise_matrix=matrix,
            threshold_pass_rate=pass_rate, processing_time_seconds=processing_time,
        )
        logger.info(f"Consistency: {technique} | mean={report.mean_similarity:.4f} | pass={report.threshold_pass_rate:.2%}")
        return report

    def compare_techniques(self, technique_batches: dict[str, list[str | Path]]) -> dict[str, ConsistencyReport]:
        results = {}
        for technique, paths in technique_batches.items():
            report = self.evaluate_consistency(paths, technique=technique)
            if report:
                results[technique] = report
        return results

    def generate_report_json(self, results: dict[str, ConsistencyReport], output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "metadata": {"evaluation_type": "facial_consistency", "model": self._model_name, "similarity_threshold": self.similarity_threshold, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
            "results": {name: r.to_dict() for name, r in results.items()},
            "ranking": sorted(results.keys(), key=lambda k: results[k].mean_similarity, reverse=True),
        }
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Report saved to: {output_path}")
        return output_path

    def generate_latex_table(self, results: dict[str, ConsistencyReport]) -> str:
        header = "\\begin{table}[h]\n\\centering\n\\caption{Facial Consistency Comparison Across Techniques}\n\\label{tab:facial_consistency}\n\\begin{tabular}{lcccccc}\n\\hline\n\\textbf{Technique} & \\textbf{Images} & \\textbf{Mean Sim.} & \\textbf{Std} & \\textbf{Min} & \\textbf{Max} & \\textbf{Pass Rate} \\\\\n\\hline\n"
        rows = []
        for name, r in sorted(results.items(), key=lambda x: x[1].mean_similarity, reverse=True):
            rows.append(f"{name} & {r.num_faces_detected} & {r.mean_similarity:.4f} & {r.std_similarity:.4f} & {r.min_similarity:.4f} & {r.max_similarity:.4f} & {r.threshold_pass_rate:.2%} \\\\")
        footer = "\n\\hline\n\\end{tabular}\n\\end{table}"
        return header + "\n".join(rows) + footer


def main():
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser(description="Evaluate facial consistency")
    parser.add_argument("--input-dir", "-i", required=True)
    parser.add_argument("--technique", "-t", default="unknown")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output", "-o", default="output/reports/consistency_report.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(str(Path(args.input_dir) / ext)))
    if not image_paths:
        logger.error(f"No images found in {args.input_dir}")
        return
    evaluator = FacialConsistencyEvaluator(similarity_threshold=args.threshold, device=args.device)
    report = evaluator.evaluate_consistency(image_paths, technique=args.technique)
    if report:
        evaluator.generate_report_json({args.technique: report}, args.output)
        print(f"\n{'='*50}\nFACIAL CONSISTENCY: {args.technique}\n{'='*50}")
        print(f"  Mean similarity: {report.mean_similarity:.4f}")
        print(f"  Pass rate (>{args.threshold}): {report.threshold_pass_rate:.2%}")


if __name__ == "__main__":
    main()
