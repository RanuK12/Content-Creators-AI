"""
Kohya_ss Automated LoRA Trainer Module
========================================
Academic Research: Automates the fine-tuning of Stable Diffusion models using
Low-Rank Adaptation (LoRA) via the kohya_ss training framework. This module
provides reproducible training configurations for evaluating identity-preserving
generation techniques across different hyperparameter regimes.

The LoRA approach enables parameter-efficient fine-tuning by decomposing weight
updates into low-rank matrices, dramatically reducing VRAM requirements while
maintaining generation quality. This is critical for character-specific model
customization in AI influencer pipelines.

References:
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
- Kohya, "sd-scripts: Training scripts for Stable Diffusion" (2023)
- Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models" (CVPR 2023)
- Kumari et al., "Multi-Concept Customization of Text-to-Image Diffusion" (CVPR 2023)
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
from loguru import logger


@dataclass
class TrainingConfig:
    """
    Complete hyperparameter configuration for LoRA training via kohya_ss.

    This dataclass encapsulates all training parameters required for reproducible
    LoRA fine-tuning experiments. Default values are optimized for character
    identity preservation on SDXL 1.0 with 24GB VRAM (RTX 4090).

    Attributes:
        pretrained_model: Path to the base Stable Diffusion checkpoint.
        network_dim: LoRA rank dimension. Higher values capture more detail but
            increase VRAM usage. 32 is optimal for character identity (Hu et al., 2022).
        network_alpha: LoRA alpha scaling factor. Typically set to rank/2 for
            stable training dynamics.
        learning_rate: Peak learning rate for the LoRA layers.
        unet_lr: Separate learning rate for UNet LoRA layers (if different from global).
        text_encoder_lr: Learning rate for text encoder LoRA layers.
        lr_scheduler: Learning rate schedule strategy.
        lr_warmup_steps: Number of warmup steps before reaching peak LR.
        lr_scheduler_num_cycles: Number of cosine annealing cycles (for cosine_with_restarts).
        max_train_steps: Maximum training iterations.
        train_batch_size: Batch size per GPU.
        resolution: Training image resolution (width, height).
        mixed_precision: Mixed precision mode for memory efficiency.
        gradient_accumulation_steps: Effective batch size multiplier.
        seed: Random seed for reproducibility.
        optimizer_type: Optimizer algorithm.
        network_module: LoRA network module type.
        dataset_dir: Path to the training dataset.
        output_dir: Path for saving checkpoints and logs.
        output_name: Name prefix for saved LoRA files.
        save_every_n_steps: Checkpoint save frequency.
        caption_extension: File extension for image captions.
        shuffle_caption: Whether to shuffle caption tokens during training.
        keep_tokens: Number of tokens to keep at the start when shuffling.
        clip_skip: CLIP skip layers (2 for anime-style, 1 for photorealistic).
        noise_offset: Noise offset for improved contrast (Birch, 2023).
        max_token_length: Maximum CLIP token length.
        xformers: Enable xformers memory-efficient attention.
        cache_latents: Cache VAE latents to disk for faster training.
        bucket_no_upscale: Disable upscaling in aspect ratio bucketing.
        min_snr_gamma: Minimum SNR gamma for loss weighting (Hang et al., 2023).
        early_stopping_patience: Number of steps without improvement before stopping.
        early_stopping_threshold: Minimum loss improvement to reset patience counter.
    """

    # Model configuration
    pretrained_model: str = "./models/sd_xl_base_1.0.safetensors"
    network_dim: int = 32
    network_alpha: int = 16
    network_module: str = "networks.lora"

    # Learning rate configuration
    learning_rate: float = 1e-4
    unet_lr: float = 1e-4
    text_encoder_lr: float = 5e-5
    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 100
    lr_scheduler_num_cycles: int = 3

    # Training duration
    max_train_steps: int = 2000
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    # Resolution and precision
    resolution: tuple[int, int] = (1024, 1024)
    mixed_precision: str = "bf16"

    # Reproducibility
    seed: int = 42

    # Optimizer
    optimizer_type: str = "AdamW8bit"

    # Dataset
    dataset_dir: str = "./datasets/character_v1"
    caption_extension: str = ".txt"
    shuffle_caption: bool = True
    keep_tokens: int = 1
    max_token_length: int = 225

    # Output
    output_dir: str = "./output/lora"
    output_name: str = "character_lora_v1"
    save_every_n_steps: int = 200

    # SDXL-specific
    clip_skip: int = 1

    # Training enhancements
    noise_offset: float = 0.0357
    xformers: bool = True
    cache_latents: bool = True
    bucket_no_upscale: bool = True
    min_snr_gamma: float = 5.0

    # Early stopping
    early_stopping_patience: int = 500
    early_stopping_threshold: float = 0.001

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> TrainingConfig:
        """
        Load training configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            A TrainingConfig instance populated from the YAML.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the YAML is malformed.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        # Extract training-specific section if nested
        training_data = raw.get("training", raw)

        # Handle resolution tuple
        if "resolution" in training_data:
            res = training_data["resolution"]
            if isinstance(res, list):
                training_data["resolution"] = tuple(res)
            elif isinstance(res, int):
                training_data["resolution"] = (res, res)

        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in training_data.items() if k in known_fields}

        logger.info(f"Loaded training config from: {config_path}")
        return cls(**filtered)

    def to_yaml(self, output_path: str | Path) -> Path:
        """Save configuration to a YAML file for reproducibility."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "training": {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in self.__dict__.items()
            }
        }

        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Config saved to: {output_path}")
        return output_path

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in self.__dict__.items()
        }


@dataclass
class TrainingMetrics:
    """
    Tracks training progress and performance metrics for analysis.

    Used for monitoring training dynamics, detecting overfitting,
    and generating loss curves for the research paper.

    Attributes:
        steps: List of step numbers where metrics were recorded.
        losses: Corresponding training loss values.
        learning_rates: Learning rate at each recorded step.
        checkpoints_saved: Paths to saved checkpoint files.
        best_loss: Lowest training loss observed.
        best_step: Step at which best loss was observed.
        early_stopped: Whether training was terminated early.
        early_stop_step: Step at which early stopping was triggered.
        total_training_time: Total wall-clock training time in seconds.
        gpu_memory_peak_gb: Peak GPU memory usage in GB.
    """

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    checkpoints_saved: list[str] = field(default_factory=list)
    best_loss: float = float("inf")
    best_step: int = 0
    early_stopped: bool = False
    early_stop_step: Optional[int] = None
    total_training_time: float = 0.0
    gpu_memory_peak_gb: float = 0.0

    def update(self, step: int, loss: float, lr: float) -> None:
        """Record metrics for a training step."""
        self.steps.append(step)
        self.losses.append(loss)
        self.learning_rates.append(lr)

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_step = step

    def get_moving_average(self, window: int = 50) -> list[float]:
        """Compute moving average of training loss."""
        if len(self.losses) < window:
            return self.losses[:]
        return [
            float(np.mean(self.losses[max(0, i - window):i + 1]))
            for i in range(len(self.losses))
        ]

    def to_dict(self) -> dict:
        """Serialize metrics to dictionary."""
        return {
            "total_steps": len(self.steps),
            "final_loss": self.losses[-1] if self.losses else None,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "early_stopped": self.early_stopped,
            "early_stop_step": self.early_stop_step,
            "total_training_time_seconds": round(self.total_training_time, 2),
            "total_training_time_minutes": round(self.total_training_time / 60, 2),
            "gpu_memory_peak_gb": round(self.gpu_memory_peak_gb, 2),
            "checkpoints_saved": self.checkpoints_saved,
            "loss_trajectory": {
                "steps": self.steps,
                "losses": self.losses,
                "learning_rates": self.learning_rates,
            },
        }


class KohyaTrainer:
    """
    Automated LoRA training orchestrator using kohya_ss sd-scripts.

    This class wraps the kohya_ss training pipeline, providing:
    - Automatic dataset configuration generation
    - Command-line argument construction for sd-scripts
    - Real-time training output parsing and metric tracking
    - Early stopping based on loss plateau detection
    - Comprehensive training report generation

    The trainer is designed for reproducible experiments comparing different
    LoRA configurations for AI character identity preservation.

    Example:
        >>> config = TrainingConfig.from_yaml("config.yaml")
        >>> trainer = KohyaTrainer(config)
        >>> metrics = trainer.train()
        >>> trainer.save_training_report(metrics, "output/reports/training_report.json")
    """

    def __init__(self, config: TrainingConfig, kohya_dir: str = "./kohya_ss"):
        """
        Initialize the Kohya trainer.

        Args:
            config: Training configuration dataclass.
            kohya_dir: Path to the kohya_ss sd-scripts installation.
        """
        self.config = config
        self.kohya_dir = Path(kohya_dir)
        self._validate_paths()
        logger.info(
            f"KohyaTrainer initialized | "
            f"dim={config.network_dim} | alpha={config.network_alpha} | "
            f"lr={config.learning_rate} | steps={config.max_train_steps}"
        )

    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        dataset_path = Path(self.config.dataset_dir)
        if not dataset_path.exists():
            logger.warning(f"Dataset directory not found: {dataset_path}")

        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    def generate_dataset_config(self, output_path: Optional[str | Path] = None) -> Path:
        """
        Generate a kohya_ss-compatible dataset configuration TOML file.

        The dataset config specifies image directories, repeat counts,
        and caption settings for the training pipeline.

        Args:
            output_path: Where to save the config. Defaults to output_dir/dataset_config.toml.

        Returns:
            Path to the generated configuration file.
        """
        if output_path is None:
            output_path = Path(self.config.output_dir) / "dataset_config.toml"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse dataset directory structure
        dataset_dir = Path(self.config.dataset_dir)
        res_w, res_h = self.config.resolution

        # Build TOML content
        toml_content = f"""[general]
shuffle_caption = {str(self.config.shuffle_caption).lower()}
caption_extension = "{self.config.caption_extension}"
keep_tokens = {self.config.keep_tokens}

[[datasets]]
resolution = [{res_w}, {res_h}]
batch_size = {self.config.train_batch_size}
enable_bucket = true
bucket_no_upscale = {str(self.config.bucket_no_upscale).lower()}

  [[datasets.subsets]]
  image_dir = "{dataset_dir}"
  num_repeats = 10
  class_tokens = "{self.config.output_name}"
"""

        with open(output_path, "w") as f:
            f.write(toml_content)

        logger.info(f"Dataset config generated: {output_path}")
        return output_path

    def build_training_command(self, dataset_config_path: str | Path) -> list[str]:
        """
        Construct the full command-line invocation for kohya_ss training.

        Builds a subprocess-compatible argument list for the sdxl_train_network.py
        script with all configured hyperparameters.

        Args:
            dataset_config_path: Path to the dataset TOML configuration.

        Returns:
            List of command-line arguments for subprocess execution.
        """
        cfg = self.config
        train_script = str(self.kohya_dir / "sdxl_train_network.py")

        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process", "1",
            train_script,
            "--pretrained_model_name_or_path", cfg.pretrained_model,
            "--dataset_config", str(dataset_config_path),
            "--output_dir", cfg.output_dir,
            "--output_name", cfg.output_name,
            "--network_module", cfg.network_module,
            "--network_dim", str(cfg.network_dim),
            "--network_alpha", str(cfg.network_alpha),
            "--learning_rate", str(cfg.learning_rate),
            "--unet_lr", str(cfg.unet_lr),
            "--text_encoder_lr", str(cfg.text_encoder_lr),
            "--lr_scheduler", cfg.lr_scheduler,
            "--lr_warmup_steps", str(cfg.lr_warmup_steps),
            "--lr_scheduler_num_cycles", str(cfg.lr_scheduler_num_cycles),
            "--max_train_steps", str(cfg.max_train_steps),
            "--train_batch_size", str(cfg.train_batch_size),
            "--gradient_accumulation_steps", str(cfg.gradient_accumulation_steps),
            "--mixed_precision", cfg.mixed_precision,
            "--save_every_n_steps", str(cfg.save_every_n_steps),
            "--seed", str(cfg.seed),
            "--optimizer_type", cfg.optimizer_type,
            "--max_token_length", str(cfg.max_token_length),
            "--clip_skip", str(cfg.clip_skip),
            "--noise_offset", str(cfg.noise_offset),
            "--min_snr_gamma", str(cfg.min_snr_gamma),
        ]

        if cfg.xformers:
            cmd.append("--xformers")
        if cfg.cache_latents:
            cmd.append("--cache_latents")
        if cfg.shuffle_caption:
            cmd.append("--shuffle_caption")

        logger.debug(f"Training command: {' '.join(cmd)}")
        return cmd

    def _parse_training_output(self, line: str, metrics: TrainingMetrics) -> None:
        """
        Parse a single line of training output to extract metrics.

        Kohya_ss outputs training progress in the format:
            steps: N/TOTAL loss: X.XXXX lr: Y.YYYY

        Args:
            line: A single line from the training process stdout/stderr.
            metrics: The TrainingMetrics instance to update.
        """
        # Pattern: steps: 100/2000 loss: 0.0834 lr: 9.5e-05
        step_match = re.search(r"steps?:\s*(\d+)/\d+", line)
        loss_match = re.search(r"loss:\s*([\d.]+(?:e[+-]?\d+)?)", line)
        lr_match = re.search(r"lr:\s*([\d.]+(?:e[+-]?\d+)?)", line)

        if step_match and loss_match:
            step = int(step_match.group(1))
            loss = float(loss_match.group(1))
            lr = float(lr_match.group(1)) if lr_match else 0.0
            metrics.update(step, loss, lr)

        # Detect checkpoint saves
        if "saving checkpoint" in line.lower() or "model saved" in line.lower():
            checkpoint_match = re.search(r"([\w/\\.-]+\.safetensors)", line)
            if checkpoint_match:
                metrics.checkpoints_saved.append(checkpoint_match.group(1))
                logger.info(f"Checkpoint saved: {checkpoint_match.group(1)}")

        # Detect GPU memory
        mem_match = re.search(r"memory:\s*([\d.]+)\s*GB", line, re.IGNORECASE)
        if mem_match:
            mem = float(mem_match.group(1))
            metrics.gpu_memory_peak_gb = max(metrics.gpu_memory_peak_gb, mem)

    def _check_early_stopping(self, metrics: TrainingMetrics) -> bool:
        """
        Check if training should be stopped early due to loss plateau.

        Early stopping is triggered when the training loss has not improved
        by at least `early_stopping_threshold` for `early_stopping_patience` steps.

        Args:
            metrics: Current training metrics.

        Returns:
            True if training should be stopped, False otherwise.
        """
        if len(metrics.steps) < 2:
            return False

        current_step = metrics.steps[-1]
        steps_since_improvement = current_step - metrics.best_step

        if steps_since_improvement >= self.config.early_stopping_patience:
            # Verify the plateau using moving average
            recent_window = min(100, len(metrics.losses))
            recent_avg = np.mean(metrics.losses[-recent_window:])
            best_window_start = max(0, metrics.best_step - 50)
            best_window_end = min(len(metrics.losses), metrics.best_step + 50)

            if best_window_end > best_window_start and best_window_end <= len(metrics.losses):
                best_avg = np.mean(metrics.losses[best_window_start:best_window_end])
                improvement = best_avg - recent_avg

                if improvement < self.config.early_stopping_threshold:
                    logger.warning(
                        f"Early stopping triggered at step {current_step} | "
                        f"No improvement for {steps_since_improvement} steps | "
                        f"Best loss: {metrics.best_loss:.6f} at step {metrics.best_step}"
                    )
                    metrics.early_stopped = True
                    metrics.early_stop_step = current_step
                    return True

        return False

    def train(self, dry_run: bool = False) -> TrainingMetrics:
        """
        Execute the full LoRA training pipeline.

        This method:
        1. Generates the dataset configuration
        2. Builds the training command
        3. Launches the kohya_ss training subprocess
        4. Monitors output in real-time for metrics
        5. Applies early stopping if configured
        6. Returns comprehensive training metrics

        Args:
            dry_run: If True, only generate configs and print the command
                without actually launching training.

        Returns:
            TrainingMetrics with full training history and statistics.
        """
        metrics = TrainingMetrics()
        start_time = time.time()

        # Generate dataset config
        dataset_config_path = self.generate_dataset_config()

        # Build command
        cmd = self.build_training_command(dataset_config_path)

        if dry_run:
            logger.info("DRY RUN - Training command:")
            logger.info(f"  {' '.join(cmd)}")
            return metrics

        # Save config for reproducibility
        config_save_path = Path(self.config.output_dir) / "training_config.yaml"
        self.config.to_yaml(config_save_path)

        logger.info(f"Starting LoRA training | Steps: {self.config.max_train_steps}")
        logger.info(f"  Network: dim={self.config.network_dim}, alpha={self.config.network_alpha}")
        logger.info(f"  LR: {self.config.learning_rate} ({self.config.lr_scheduler})")
        logger.info(f"  Output: {self.config.output_dir}/{self.config.output_name}")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.kohya_dir),
            )

            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if line:
                    self._parse_training_output(line, metrics)

                    # Check early stopping every 50 steps
                    if metrics.steps and metrics.steps[-1] % 50 == 0:
                        if self._check_early_stopping(metrics):
                            logger.info("Terminating training process due to early stopping...")
                            process.terminate()
                            process.wait(timeout=30)
                            break

            process.wait()

            if process.returncode != 0 and not metrics.early_stopped:
                logger.error(f"Training process exited with code {process.returncode}")

        except FileNotFoundError:
            logger.error(
                "Training command not found. Ensure kohya_ss and accelerate are installed."
            )
        except subprocess.TimeoutExpired:
            logger.error("Training process timed out during termination.")
            process.kill()
        except Exception as e:
            logger.error(f"Training failed with exception: {e}")

        metrics.total_training_time = time.time() - start_time
        logger.info(
            f"Training complete | "
            f"Steps: {len(metrics.steps)} | "
            f"Time: {metrics.total_training_time / 60:.1f} min | "
            f"Best loss: {metrics.best_loss:.6f} @ step {metrics.best_step}"
        )

        return metrics

    def save_training_report(
        self, metrics: TrainingMetrics, output_path: str | Path
    ) -> Path:
        """
        Save comprehensive training report to JSON.

        Includes full configuration, metrics history, and training statistics
        for research documentation and comparison across experiments.

        Args:
            metrics: Completed training metrics.
            output_path: Path to save the JSON report.

        Returns:
            Path to the saved report file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "metadata": {
                "report_type": "lora_training",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "framework": "kohya_ss",
                "method": "LoRA (Hu et al., 2022)",
            },
            "config": self.config.to_dict(),
            "metrics": metrics.to_dict(),
            "summary": {
                "converged": not metrics.early_stopped,
                "final_loss": metrics.losses[-1] if metrics.losses else None,
                "best_loss": metrics.best_loss,
                "total_steps_trained": len(metrics.steps),
                "effective_batch_size": (
                    self.config.train_batch_size * self.config.gradient_accumulation_steps
                ),
                "total_params_trained": f"~{self.config.network_dim * 2 * 1000:.0f}K (LoRA rank={self.config.network_dim})",
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Training report saved: {output_path}")
        return output_path


def main():
    """CLI entry point for automated LoRA training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated LoRA Training via kohya_ss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python kohya_trainer.py --config config.yaml

  # Dry run to verify command
  python kohya_trainer.py --config config.yaml --dry-run

  # Train with custom output
  python kohya_trainer.py --config config.yaml --output-dir ./output/experiment_01
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--kohya-dir",
        type=str,
        default="./kohya_ss",
        help="Path to kohya_ss installation directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate config and print command without training",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save training report JSON",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = TrainingConfig.from_yaml(args.config)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = TrainingConfig()

    # Apply overrides
    if args.output_dir:
        config.output_dir = args.output_dir

    # Initialize trainer
    trainer = KohyaTrainer(config, kohya_dir=args.kohya_dir)

    # Run training
    metrics = trainer.train(dry_run=args.dry_run)

    # Save report
    if not args.dry_run:
        report_path = args.report or str(
            Path(config.output_dir) / "training_report.json"
        )
        trainer.save_training_report(metrics, report_path)

        # Print summary
        print(f"\n{'='*60}")
        print(f"  LORA TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Steps trained:    {len(metrics.steps)}")
        print(f"  Best loss:        {metrics.best_loss:.6f} (step {metrics.best_step})")
        print(f"  Final loss:       {metrics.losses[-1] if metrics.losses else 'N/A'}")
        print(f"  Early stopped:    {metrics.early_stopped}")
        print(f"  Training time:    {metrics.total_training_time / 60:.1f} minutes")
        print(f"  Report saved:     {report_path}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
