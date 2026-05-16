# AI Art Pipeline - Academic Research

> **Automated Pipeline for Consistent Character Generation Using Diffusion Models**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AI Generated](https://img.shields.io/badge/Content-AI_Generated-red.svg)]()

## Overview

End-to-end automated pipeline for generating visually consistent digital characters using AI diffusion models. Built for academic research on identity preservation techniques across multiple generation paradigms.

**All output content is explicitly marked as AI-generated. Characters are fictional concept art.**

## Architecture

```
Input (Config + Prompts)
    → LoRA Training (Kohya_ss) 
    → Image Generation (ComfyUI + Flux.1)
    → Quality Evaluation (InsightFace embeddings)
    → Consistency Filtering (cosine similarity > 0.85)
    → Video Generation (Kling AI I2V)
    → Post-Production (FFmpeg + watermark)
    → Multi-Platform Publishing (Instagram + Twitter)
    → Metrics & Reports (LaTeX + matplotlib)
```

## Project Structure

```
├── config.yaml                     # Global configuration
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
│
├── src/
│   ├── orchestrator.py             # Main pipeline coordinator
│   ├── video_pipeline.py           # End-to-end video generation
│   ├── api_wrappers/
│   │   └── kling_api.py            # Kling AI async API wrapper
│   ├── batch_processing/
│   │   └── comfyui_batch.py        # ComfyUI WebSocket batch generator
│   ├── dataset_prep/
│   │   └── kohya_prep.py           # Dataset augmentation & tagging
│   ├── evaluation/
│   │   └── facial_consistency.py   # ArcFace embeddings + similarity
│   ├── metrics/
│   │   └── benchmark.py            # Full metrics dashboard + KPIs
│   ├── post_production/
│   │   └── post_prod.py            # Watermark, resize, FFmpeg
│   ├── publishing/
│   │   └── social_publisher.py     # Instagram & Twitter automation
│   ├── scheduling/
│   │   └── content_scheduler.py    # Editorial calendar + rotation
│   └── training/
│       └── kohya_trainer.py        # Automated LoRA training
│
├── workflows/
│   ├── workflow_a_character_consistency/   # LoRA + PuLID + IP-Adapter + ControlNet
│   ├── workflow_b_outfit_variation/       # Inpainting + Depth ControlNet
│   ├── workflow_c_motion_reel/            # 9:16 frames for I2V
│   ├── workflow_d_multi_character/        # Regional prompting + multi-LoRA
│   ├── workflow_e_style_transfer/         # Style LoRA + IP-Adapter
│   └── workflow_f_inpainting_advanced/    # SAM segmentation + inpaint
│
├── docs/
│   ├── diagrams/architecture.md    # Mermaid architecture diagram
│   └── paper/paper_structure.md    # Academic paper outline
│
└── output/                         # Generated content (gitignored)
    ├── portfolio/{character}/{date}/
    ├── reports/metrics/
    └── calendar/
```

## Quick Start

### 1. Environment Setup

```bash
# Clone
git clone https://github.com/RanuK12/Content-Creators-AI.git
cd Content-Creators-AI

# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys
```

### 2. Dataset Preparation

```bash
python -m src.dataset_prep.kohya_prep \
    --input datasets/raw_images/ \
    --output datasets/prepared/ \
    --trigger "ohwx" \
    --resolution 1024 \
    --crop-faces
```

### 3. LoRA Training

```bash
python -m src.training.kohya_trainer \
    --config config.yaml \
    --kohya-path /path/to/kohya_ss \
    --dataset-dir datasets/prepared/ \
    --output-name character_v1 \
    --trigger-word "ohwx"
```

### 4. Run Pipeline

```bash
# Full orchestration
python -m src.orchestrator --config config.yaml --tasks tasks.json

# Generate editorial calendar
python -m src.scheduling.content_scheduler --period week --posts-per-day 2

# Run metrics dashboard
python -m src.metrics.benchmark --reports-dir output/reports/ --charts --latex
```

### 5. Evaluate Consistency

```bash
python -m src.evaluation.facial_consistency \
    --input-dir output/portfolio/character_01/ \
    --technique "LoRA+PuLID" \
    --threshold 0.85
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 16GB | RTX 4090 24GB |
| RAM | 32GB | 64GB |
| Storage | 50GB SSD | 200GB NVMe |
| CPU | 8 cores | i9-13900K |

## Workflows

| Workflow | Purpose | Key Technique |
|----------|---------|---------------|
| A | Character consistency | LoRA + PuLID + IP-Adapter + ControlNet |
| B | Outfit variation | Inpainting + Depth ControlNet |
| C | Motion reels (9:16) | PuLID + centered composition |
| D | Multi-character scenes | Regional prompting + multi-LoRA |
| E | Style transfer | Style LoRA + IP-Adapter |
| F | Background swap | SAM segmentation + inpainting |

## Metrics & Evaluation

- **Facial Consistency**: ArcFace cosine similarity (target >0.85)
- **Image Quality**: LPIPS, SSIM, FID
- **Video Quality**: Temporal SSIM, flicker score
- **Pipeline KPIs**: Success rate, generation time, cost/video

## Ethical Disclosure

- All generated content is watermarked as "AI Generated Concept Art"
- EXIF metadata includes provenance information
- Characters are fictional digital creations
- No real person's likeness is reproduced
- Pipeline includes mandatory AI disclosure in all published captions

## Research Context

This pipeline supports academic research on visual consistency in diffusion models. Key research questions:

1. How do LoRA weight values affect identity preservation?
2. Which face preservation technique (PuLID vs InstantID vs IP-Adapter) provides optimal consistency-diversity trade-off?
3. Can automated pipelines achieve publication-quality output with minimal human intervention?

## License

MIT License - See [LICENSE](LICENSE) for details.
