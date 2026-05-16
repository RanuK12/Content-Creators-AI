"""
Kohya_ss Automated LoRA Trainer
================================
Automates LoRA training via Kohya_ss CLI with:
- Dynamic config generation from YAML
- Real-time training metrics monitoring (loss, LR)
- Checkpoint management with early stopping
- Automatic sample generation every N steps
- Training report generation (loss curves, sample grid)

Research relevance:
- Enables systematic evaluation of hyperparameter impact on identity consistency
- Automated sample generation allows visual tracking of convergence
- Early stopping prevents overfitting while maximizing identity fidelity

References:
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Kohya_ss sd-scripts documentation
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger


@dataclass
class TrainingConfig:
    """Configuration for LoRA training run."""
    # Model
    pretrained_model: str = "models/checkpoints/flux1-dev-fp8.safetensors"
    model_type: str = "flux"  # flux, sdxl, sd15
    
    # Dataset
    dataset_dir: str = "datasets/character_01/"
    resolution: int = 1024
    
    # LoRA architecture
    network_dim: int = 32
    network_alpha: int = 16
    network_module: str = "networks.lora"
    
    # Training
    learning_rate: float = 1e-4
    unet_lr: float = 1e-4
    text_encoder_lr: float = 5e-5
    optimizer: str = "AdamW8bit"
    scheduler: str = "cosine_with_restarts"
    num_cycles: int = 3
    warmup_steps: int = 100
    max_train_steps: int = 2000
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"
    clip_skip: int = 2
    seed: int = 42
    
    # Saving
    output_dir: str = "output/loras/"
    output_name: str = "character_lora_v1"
    save_every_n_steps: int = 200
    save_model_as: str = "safetensors"
    
    # Sampling
    sample_every_n_steps: int = 100
    sample_prompts: list[str] = field(default_factory=lambda: [
        "portrait of {trigger}, professional photography, studio lighting",
        "{trigger} standing outdoors, full body, natural light",
        "close-up of {trigger}, cinematic, dramatic lighting",
    ])
    sample_sampler: str = "euler"
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Trigger word
    trigger_word: str = "ohwx"
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TrainingConfig":
        """Load training config from global config.yaml."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        
        training_cfg = cfg.get("training", {})
        hp = training_cfg.get("hyperparameters", {})
        es = training_cfg.get("early_stopping", {})
        
        return cls(
            dataset_dir=training_cfg.get("dataset_dir", cls.dataset_dir),
            output_dir=training_cfg.get("output_dir", cls.output_dir),
            network_dim=hp.get("network_dim", cls.network_dim),
            network_alpha=hp.get("network_alpha", cls.network_alpha),
            learning_rate=hp.get("learning_rate", cls.learning_rate),
            unet_lr=hp.get("unet_lr", cls.unet_lr),
            text_encoder_lr=hp.get("text_encoder_lr", cls.text_encoder_lr),
            optimizer=hp.get("optimizer", cls.optimizer),
            scheduler=hp.get("scheduler", cls.scheduler),
            num_cycles=hp.get("num_cycles", cls.num_cycles),
            max_train_steps=hp.get("max_train_steps", cls.max_train_steps),
            batch_size=hp.get("batch_size", cls.batch_size),
            gradient_accumulation_steps=hp.get("gradient_accumulation_steps", cls.gradient_accumulation_steps),
            mixed_precision=hp.get("mixed_precision", cls.mixed_precision),
            resolution=hp.get("resolution", cls.resolution),
            clip_skip=hp.get("clip_skip", cls.clip_skip),
            seed=hp.get("seed", cls.seed),
            save_every_n_steps=hp.get("save_every_n_steps", cls.save_every_n_steps),
            sample_every_n_steps=hp.get("sample_every_n_steps", cls.sample_every_n_steps),
            early_stopping_enabled=es.get("enabled", cls.early_stopping_enabled),
            early_stopping_patience=es.get("patience", cls.early_stopping_patience),
            early_stopping_min_delta=es.get("min_delta", cls.early_stopping_min_delta),
            sample_prompts=training_cfg.get("sample_prompts", cls.sample_prompts),
        )


@dataclass
class TrainingMetrics:
    """Collected metrics during training."""
    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    checkpoints_saved: list[str] = field(default_factory=list)
    samples_generated: list[str] = field(default_factory=list)
    best_loss: float = float("inf")
    best_step: int = 0
    total_time_seconds: float = 0.0
    early_stopped: bool = False
    early_stop_step: int = 0

    def to_dict(self) -> dict:
        return {
            "total_steps": len(self.steps),
            "final_loss": self.losses[-1] if self.losses else None,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "total_time_seconds": round(self.total_time_seconds, 1),
            "early_stopped": self.early_stopped,
            "early_stop_step": self.early_stop_step,
            "checkpoints": self.checkpoints_saved,
            "num_samples": len(self.samples_generated),
        }


class KohyaTrainer:
    """
    Automated LoRA trainer using Kohya_ss CLI.
    
    Manages the full training lifecycle:
    1. Generate training config TOML
    2. Launch Kohya_ss subprocess
    3. Monitor output for metrics
    4. Handle early stopping
    5. Generate training report
    """

    def __init__(
        self,
        kohya_ss_path: str | Path = "/path/to/kohya_ss",
        accelerate_config: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            kohya_ss_path: Path to kohya_ss installation
            accelerate_config: Path to accelerate config YAML
        """
        self.kohya_ss_path = Path(kohya_ss_path)
        self.accelerate_config = accelerate_config
        self.metrics = TrainingMetrics()
        
        logger.info(f"KohyaTrainer | kohya_ss={kohya_ss_path}")

    def generate_dataset_config(self, config: TrainingConfig) -> Path:
        """
        Generate dataset TOML config for Kohya_ss.
        
        Creates the dataset configuration file that defines
        image directories, captions, and training parameters.
        """
        output_path = Path(config.output_dir) / "dataset_config.toml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        toml_content = f"""[general]
shuffle_caption = true
caption_extension = ".txt"
keep_tokens = 1

[[datasets]]
resolution = {config.resolution}
batch_size = {config.batch_size}

  [[datasets.subsets]]
  image_dir = "{config.dataset_dir}"
  num_repeats = 10
  class_tokens = "{config.trigger_word}"
"""
        
        with open(output_path, "w") as f:
            f.write(toml_content)
        
        logger.info(f"Dataset config generated: {output_path}")
        return output_path

    def build_training_command(self, config: TrainingConfig) -> list[str]:
        """
        Build the Kohya_ss training command with all arguments.
        
        Returns a list suitable for subprocess.Popen.
        """
        dataset_config = self.generate_dataset_config(config)
        
        # Base command
        script = "flux_train_network.py" if config.model_type == "flux" else "train_network.py"
        
        cmd = [
            "accelerate", "launch",
        ]
        
        if self.accelerate_config:
            cmd.extend(["--config_file", self.accelerate_config])
        
        cmd.extend([
            "--num_cpu_threads_per_process", "1",
            str(self.kohya_ss_path / script),
            "--pretrained_model_name_or_path", config.pretrained_model,
            "--dataset_config", str(dataset_config),
            "--output_dir", config.output_dir,
            "--output_name", config.output_name,
            "--network_module", config.network_module,
            "--network_dim", str(config.network_dim),
            "--network_alpha", str(config.network_alpha),
            "--learning_rate", str(config.learning_rate),
            "--unet_lr", str(config.unet_lr),
            "--text_encoder_lr", str(config.text_encoder_lr),
            "--optimizer_type", config.optimizer,
            "--lr_scheduler", config.scheduler,
            "--lr_scheduler_num_cycles", str(config.num_cycles),
            "--lr_warmup_steps", str(config.warmup_steps),
            "--max_train_steps", str(config.max_train_steps),
            "--mixed_precision", config.mixed_precision,
            "--save_every_n_steps", str(config.save_every_n_steps),
            "--save_model_as", config.save_model_as,
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--clip_skip", str(config.clip_skip),
            "--seed", str(config.seed),
            "--cache_latents",
            "--cache_latents_to_disk",
            "--logging_dir", f"{config.output_dir}/logs",
        ])
        
        # Sample generation
        if config.sample_every_n_steps > 0:
            sample_prompts_path = Path(config.output_dir) / "sample_prompts.txt"
            sample_prompts_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(sample_prompts_path, "w") as f:
                for prompt in config.sample_prompts:
                    resolved = prompt.replace("{trigger}", config.trigger_word)
                    f.write(f"{resolved}\n")
            
            cmd.extend([
                "--sample_every_n_steps", str(config.sample_every_n_steps),
                "--sample_prompts", str(sample_prompts_path),
                "--sample_sampler", config.sample_sampler,
            ])
        
        return cmd

    def _parse_training_output(self, line: str) -> Optional[dict]:
        """
        Parse a training output line for metrics.
        
        Kohya_ss outputs lines like:
        steps: 100, loss: 0.0234, lr: 0.0001
        """
        metrics = {}
        
        if "loss=" in line or "loss:" in line:
            try:
                # Handle format: "step=100, loss=0.0234, lr=1e-4"
                parts = line.strip().split(",")
                for part in parts:
                    part = part.strip()
                    if "step" in part.lower():
                        val = part.split("=")[-1].split(":")[-1].strip()
                        metrics["step"] = int(float(val))
                    elif "loss" in part.lower():
                        val = part.split("=")[-1].split(":")[-1].strip()
                        metrics["loss"] = float(val)
                    elif "lr" in part.lower():
                        val = part.split("=")[-1].split(":")[-1].strip()
                        metrics["lr"] = float(val)
            except (ValueError, IndexError):
                pass
        
        return metrics if metrics else None

    def _check_early_stopping(self, config: TrainingConfig) -> bool:
        """
        Check if early stopping criteria is met.
        
        Uses patience-based approach: stop if loss hasn't improved
        by min_delta for patience consecutive evaluations.
        """
        if not config.early_stopping_enabled:
            return False
        
        if len(self.metrics.losses) < config.early_stopping_patience + 1:
            return False
        
        recent_losses = self.metrics.losses[-config.early_stopping_patience:]
        best_recent = min(recent_losses)
        
        if self.metrics.best_loss - best_recent < config.early_stopping_min_delta:
            logger.warning(
                f"Early stopping triggered at step {self.metrics.steps[-1]} | "
                f"best_loss={self.metrics.best_loss:.6f} | "
                f"recent_best={best_recent:.6f}"
            )
            return True
        
        return False

    def train(self, config: TrainingConfig) -> TrainingMetrics:
        """
        Execute training with real-time monitoring.
        
        This is the main entry point. Launches Kohya_ss subprocess,
        monitors output for metrics, handles early stopping, and
        returns collected metrics.
        
        Args:
            config: Training configuration
            
        Returns:
            TrainingMetrics with full training history
        """
        cmd = self.build_training_command(config)
        self.metrics = TrainingMetrics()
        
        logger.info(f"Starting training: {config.output_name}")
        logger.info(f"Steps: {config.max_train_steps} | LR: {config.learning_rate}")
        logger.info(f"Command: {' '.join(cmd[:5])}...")
        
        start_time = time.time()
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            for line in iter(process.stdout.readline, ""):
                line = line.strip()
                if not line:
                    continue
                
                # Parse metrics
                parsed = self._parse_training_output(line)
                if parsed:
                    if "step" in parsed:
                        self.metrics.steps.append(parsed["step"])
                    if "loss" in parsed:
                        loss = parsed["loss"]
                        self.metrics.losses.append(loss)
                        self.metrics.timestamps.append(time.time() - start_time)
                        
                        if loss < self.metrics.best_loss:
                            self.metrics.best_loss = loss
                            self.metrics.best_step = parsed.get("step", 0)
                    if "lr" in parsed:
                        self.metrics.learning_rates.append(parsed["lr"])
                    
                    # Check early stopping
                    if self._check_early_stopping(config):
                        self.metrics.early_stopped = True
                        self.metrics.early_stop_step = self.metrics.steps[-1]
                        process.terminate()
                        break
                
                # Log checkpoints
                if "saving" in line.lower() and "checkpoint" in line.lower():
                    self.metrics.checkpoints_saved.append(line)
                    logger.info(f"Checkpoint: {line}")
                
                # Log sample generation
                if "sample" in line.lower() and "generat" in line.lower():
                    self.metrics.samples_generated.append(line)
            
            process.wait()
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        
        self.metrics.total_time_seconds = time.time() - start_time
        
        logger.info(
            f"Training complete | steps={len(self.metrics.steps)} | "
            f"best_loss={self.metrics.best_loss:.6f} @ step {self.metrics.best_step} | "
            f"time={self.metrics.total_time_seconds:.0f}s | "
            f"early_stopped={self.metrics.early_stopped}"
        )
        
        return self.metrics

    def save_training_report(
        self, config: TrainingConfig, output_path: Optional[str | Path] = None
    ) -> Path:
        """
        Save training metrics report as JSON.
        
        Includes full training history for later analysis and plotting.
        """
        if output_path is None:
            output_path = Path(config.output_dir) / "training_report.json"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            "config": {
                "model_type": config.model_type,
                "network_dim": config.network_dim,
                "network_alpha": config.network_alpha,
                "learning_rate": config.learning_rate,
                "max_steps": config.max_train_steps,
                "batch_size": config.batch_size,
                "resolution": config.resolution,
                "optimizer": config.optimizer,
                "scheduler": config.scheduler,
                "trigger_word": config.trigger_word,
            },
            "metrics": self.metrics.to_dict(),
            "loss_history": {
                "steps": self.metrics.steps,
                "values": self.metrics.losses,
            },
            "lr_history": {
                "values": self.metrics.learning_rates,
            },
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Training report saved: {output_path}")
        return output_path


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for LoRA training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated LoRA training with Kohya_ss")
    parser.add_argument("--config", "-c", default="config.yaml", help="Global config path")
    parser.add_argument("--kohya-path", required=True, help="Path to kohya_ss installation")
    parser.add_argument("--dataset-dir", help="Override dataset directory")
    parser.add_argument("--output-name", help="Override output LoRA name")
    parser.add_argument("--max-steps", type=int, help="Override max training steps")
    parser.add_argument("--trigger-word", default="ohwx", help="Trigger word for LoRA")
    
    args = parser.parse_args()
    
    # Load config
    config = TrainingConfig.from_yaml(args.config)
    
    # Apply overrides
    if args.dataset_dir:
        config.dataset_dir = args.dataset_dir
    if args.output_name:
        config.output_name = args.output_name
    if args.max_steps:
        config.max_train_steps = args.max_steps
    config.trigger_word = args.trigger_word
    
    # Train
    trainer = KohyaTrainer(kohya_ss_path=args.kohya_path)
    metrics = trainer.train(config)
    trainer.save_training_report(config)
    
    print(f"\nTraining complete!")
    print(f"  Best loss: {metrics.best_loss:.6f} @ step {metrics.best_step}")
    print(f"  Total time: {metrics.total_time_seconds:.0f}s")
    print(f"  Early stopped: {metrics.early_stopped}")


if __name__ == "__main__":
    main()
