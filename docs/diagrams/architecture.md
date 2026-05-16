# Pipeline Architecture Diagram

```mermaid
graph TB
    subgraph "INPUT LAYER"
        A[config.yaml] --> ORCH
        B[Task JSON / Calendar] --> ORCH
        C[Character LoRAs] --> GEN
        D[Reference Images] --> GEN
    end

    subgraph "TRAINING LAYER"
        DS[Dataset Prep<br/>kohya_prep.py] --> TR[Kohya Trainer<br/>kohya_trainer.py]
        TR --> LORA[Trained LoRA<br/>.safetensors]
        TR --> REPORT_T[Training Report<br/>loss curves]
    end

    subgraph "GENERATION LAYER"
        ORCH[Orchestrator<br/>orchestrator.py] --> GEN[ComfyUI Batch<br/>comfyui_batch.py]
        GEN --> WF_A[Workflow A<br/>Character Consistency]
        GEN --> WF_B[Workflow B<br/>Outfit Variation]
        GEN --> WF_C[Workflow C<br/>Motion Reel]
        GEN --> WF_D[Workflow D<br/>Multi-Character]
        GEN --> WF_E[Workflow E<br/>Style Transfer]
        GEN --> WF_F[Workflow F<br/>Inpainting + SAM]
        GEN --> RAW[Raw Generated Images]
    end

    subgraph "EVALUATION LAYER"
        RAW --> EVAL[Facial Consistency<br/>facial_consistency.py]
        EVAL --> FILTER{Threshold<br/>> 0.85?}
        FILTER -->|Pass| GOOD[Filtered Images]
        FILTER -->|Fail| REJECT[Rejected]
    end

    subgraph "VIDEO LAYER"
        GOOD --> VP[Video Pipeline<br/>video_pipeline.py]
        VP --> KLING[Kling AI API<br/>Image-to-Video]
        KLING --> RAW_V[Raw Videos]
    end

    subgraph "POST-PRODUCTION"
        GOOD --> PP[Post Production<br/>post_prod.py]
        RAW_V --> PP
        PP --> WM[Watermarked Content]
        PP --> RESIZE[Multi-Platform Resize]
        PP --> META[Metadata Injection]
    end

    subgraph "PUBLISHING LAYER"
        SCH[Content Scheduler<br/>content_scheduler.py] --> PUB[Social Publisher<br/>social_publisher.py]
        WM --> PUB
        PUB --> IG[Instagram<br/>Feed / Stories / Reels]
        PUB --> TW[Twitter/X<br/>Tweets / Threads]
    end

    subgraph "METRICS LAYER"
        EVAL --> DASH[Metrics Dashboard<br/>benchmark.py]
        VP --> DASH
        PUB --> DASH
        REPORT_T --> DASH
        DASH --> CHARTS[Matplotlib Charts]
        DASH --> LATEX[LaTeX Tables]
        DASH --> KPI[Influencer KPIs]
        DASH --> JSON_R[JSON Reports]
    end

    style ORCH fill:#e1f5fe
    style EVAL fill:#fff3e0
    style PP fill:#e8f5e9
    style DASH fill:#fce4ec
    style KLING fill:#f3e5f5
```

## Data Flow Summary

| Stage | Input | Output | Module |
|-------|-------|--------|--------|
| Training | Raw images + config | LoRA .safetensors | `kohya_trainer.py` |
| Generation | LoRA + prompts + workflow | Raw images | `comfyui_batch.py` |
| Evaluation | Raw images | Filtered images + scores | `facial_consistency.py` |
| Video | Best images | MP4 videos | `video_pipeline.py` + `kling_api.py` |
| Post-prod | Filtered content | Watermarked + resized | `post_prod.py` |
| Scheduling | Config + characters | Editorial calendar | `content_scheduler.py` |
| Publishing | Final content + calendar | Published posts | `social_publisher.py` |
| Metrics | All stage outputs | Reports + charts + KPIs | `benchmark.py` |

## VRAM Usage Estimates (RTX 4090 24GB)

| Operation | VRAM (GB) | Duration |
|-----------|-----------|----------|
| Flux.1 dev fp8 generation | ~12.5 | 15-30s/image |
| SDXL fallback | ~6.5 | 8-15s/image |
| LoRA training (bf16, batch=1) | ~18-22 | 30-90min total |
| InsightFace evaluation | ~2 | <1s/image |
| ControlNet (single) | +1.5 | +5s/image |
| PuLID + IP-Adapter | +3-4 | +10s/image |
