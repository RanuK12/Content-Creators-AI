# Content-Creators-AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-compatible-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![RTX 4090](https://img.shields.io/badge/GPU-RTX%204090-76B900.svg)](https://www.nvidia.com/)

> **Automated AI character content generation pipeline** — from LoRA training to multi-platform publishing with facial consistency evaluation.

---

## Overview

Content-Creators-AI is a research-grade pipeline for generating consistent AI character content at scale. It combines fine-tuned LoRA models, ComfyUI batch processing, facial consistency evaluation, video generation, and automated multi-platform publishing into a single orchestrated system.

**Key capabilities:**
- Train character-specific LoRA models with Kohya ss
- Generate batches of consistent character images via ComfyUI
- Evaluate facial consistency with InsightFace embeddings
- Animate images to video with Kling API (image-to-video)
- Post-produce with watermarking, resizing, and platform-specific formatting
- Schedule and publish across Instagram, Twitter, and TikTok

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE ORCHESTRATOR                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐  │
│  │  Dataset  │──▶│   Training   │──▶│ Generation │──▶│Evaluation│  │
│  │   Prep   │   │  (Kohya ss)  │   │ (ComfyUI)  │   │  (Face)  │  │
│  └──────────┘   └──────────────┘   └────────────┘   └──────────┘  │
│                                                           │         │
│                                                           ▼         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌──────────┐  │
│  │Publishing│◀──│  Scheduling  │◀──│    Post     │◀──│  Video   │  │
│  │(Social)  │   │  (Calendar)  │   │ Production │   │  (Kling) │  │
│  └──────────┘   └──────────────┘   └────────────┘   └──────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    METRICS & BENCHMARKING                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Content-Creators-AI/
├── config.yaml                    # Master configuration
├── requirements.txt               # Python dependencies
├── tasks_example.json             # Example task definitions
├── .env.example                   # Environment variables template
│
├── src/
│   ├── orchestrator.py            # Pipeline orchestration (entry point)
│   ├── video_pipeline.py          # Video generation pipeline
│   ├── api_wrappers/
│   │   └── kling_api.py           # Kling API client (I2V)
│   ├── batch_processing/
│   │   └── comfyui_batch.py       # ComfyUI batch processor
│   ├── dataset_prep/
│   │   └── kohya_prep.py          # Dataset preparation for training
│   ├── evaluation/
│   │   └── facial_consistency.py  # Face embedding consistency scorer
│   ├── metrics/
│   │   └── benchmark.py           # Performance benchmarking
│   ├── post_production/
│   │   └── post_prod.py           # Watermark, resize, format
│   ├── publishing/
│   │   └── social_publisher.py    # Multi-platform publisher
│   ├── scheduling/
│   │   └── content_scheduler.py   # Content calendar & scheduling
│   └── training/
│       └── kohya_trainer.py       # LoRA training wrapper
│
├── workflows/
│   ├── workflow_a_character_consistency/  # Base character generation
│   ├── workflow_b_outfit_variation/      # Outfit/style variation
│   ├── workflow_c_motion_reel/           # Motion/video keyframes
│   ├── workflow_d_multi_character/       # Multi-character scenes
│   ├── workflow_e_style_transfer/        # Artistic style transfer
│   └── workflow_f_inpainting_advanced/   # Advanced inpainting
│
├── docs/
│   ├── diagrams/
│   │   └── architecture.md        # Mermaid architecture diagrams
│   ├── paper/
│   │   └── paper_structure.md     # Academic paper outline
│   └── influencer_action_plan.md  # Growth strategy document
│
├── tests/
│   └── __init__.py
│
├── scripts/
│   └── setup/
│       └── setup.sh               # Environment setup script
│
└── output/                        # Generated content (gitignored)
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/Content-Creators-AI.git
cd Content-Creators-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys (KLING_API_KEY, social media tokens)
```

### 2. Dataset Preparation

```bash
# Prepare training dataset (20-50 curated images per character)
python -m src.dataset_prep.kohya_prep \
  --input-dir data/raw/character_name \
  --output-dir data/datasets/character_name \
  --resolution 1024 \
  --caption-method blip2
```

### 3. LoRA Training

```bash
# Train character LoRA (uses config.yaml hyperparameters)
python -m src.training.kohya_trainer \
  --dataset data/datasets/character_name \
  --output models/loras/character_name_v1.safetensors \
  --config config.yaml
```

### 4. Run Pipeline

```bash
# Execute full pipeline from task file
python -m src.orchestrator \
  --task-file tasks_example.json \
  --config config.yaml

# Dry run to validate tasks
python -m src.orchestrator \
  --task-file tasks_example.json \
  --dry-run

# Verbose mode with custom output
python -m src.orchestrator \
  --task-file tasks_example.json \
  --verbose \
  --output-dir output/batch_001
```

### 5. Evaluation

```bash
# Run facial consistency benchmark
python -m src.metrics.benchmark \
  --character character_name \
  --reference-dir data/references/character_name \
  --generated-dir output/character_name/raw
```

---

## Hardware Requirements

| Component | Minimum | Recommended | Used In |
|-----------|---------|-------------|---------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) | Training, Generation |
| VRAM | 10 GB | 24 GB | Model loading, batching |
| RAM | 32 GB | 64 GB | Dataset processing |
| Storage | 100 GB SSD | 500 GB NVMe | Models, outputs |
| CPU | 8 cores | 16+ cores | Data prep, FFmpeg |

**VRAM Budget (RTX 4090):**
| Stage | VRAM Usage |
|-------|-----------|
| Flux.1 Dev (fp8) | ~12 GB |
| LoRA Training (bs=2) | ~18 GB |
| ComfyUI + ControlNet | ~16 GB |
| InsightFace Evaluation | ~4 GB |
| Video Post-Production | ~2 GB |

---

## Workflows

| Workflow | Purpose | Key Nodes |
|----------|---------|-----------|
| A - Character Consistency | Base character generation | Flux + LoRA + PuLID |
| B - Outfit Variation | Style/clothing changes | LoRA + IP-Adapter + ControlNet |
| C - Motion Reel | Video keyframe generation | AnimateDiff + Temporal |
| D - Multi-Character | Two-character scenes | Regional Prompting + 2× LoRA |
| E - Style Transfer | Artistic style application | Style LoRA + IP-Adapter FaceID |
| F - Advanced Inpainting | Targeted region editing | SAM + GroundingDino + Inpaint |

---

## Metrics

The pipeline tracks the following quality and performance metrics:

- **Facial Consistency Score** — cosine similarity of face embeddings (target: ≥0.85)
- **Filter Pass Rate** — percentage of images passing quality threshold
- **CLIP Score** — text-image alignment measurement
- **Aesthetic Score** — learned aesthetic quality prediction
- **Temporal Consistency** — frame-to-frame stability in video
- **Generation Throughput** — images/minute at current batch size
- **Platform Delivery Rate** — successful posts per scheduling window
- **Engagement Correlation** — quality score vs. audience engagement

---

## Ethical Disclosure

This project generates AI-created content. All outputs are:

- **Disclosed** — Published content includes AI-generation disclosure
- **Watermarked** — Configurable watermark on all generated images
- **Non-deceptive** — Not designed to impersonate real individuals
- **Research-oriented** — Built for studying synthetic media pipelines
- **Consent-aware** — Training data sourced from consented/licensed material

Users of this pipeline are responsible for:
1. Complying with platform terms of service
2. Disclosing AI-generated nature of content
3. Not using outputs for deception or fraud
4. Respecting intellectual property rights

---

## Research Context

This project explores the intersection of:
- **Generative AI consistency** — maintaining character identity across generations
- **Automated content pipelines** — end-to-end production without manual intervention
- **Quality evaluation** — automated scoring of synthetic media fidelity
- **Multi-modal generation** — image-to-video animation pipelines

See `docs/paper/paper_structure.md` for the academic paper outline.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

Models, weights, and third-party APIs are subject to their own licenses.
Flux.1 Dev is subject to the FLUX.1 [dev] Non-Commercial License.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

*Built with ComfyUI, Kohya ss, InsightFace, Kling API, and FFmpeg.*

## Licencia

MIT — © 2026 Ranuk IT Solutions | ranuk.dev
