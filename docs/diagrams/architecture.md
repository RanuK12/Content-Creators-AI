# Architecture Diagrams

## Full Pipeline Flow

```mermaid
flowchart TD
    subgraph Input["Input Layer"]
        A1[Raw Images 20-50 per character]
        A2[Captions & Tags]
        A3[Reference Embeddings]
    end

    subgraph Training["Training Layer"]
        B1[Dataset Preparation<br/>Kohya Prep]
        B2[LoRA Training<br/>Kohya ss / AdamW8bit]
        B3[Model Validation<br/>Sample Generation]
    end

    subgraph Generation["Generation Layer"]
        C1[ComfyUI Server<br/>WebSocket API]
        C2[Flux.1 Dev fp8<br/>Primary Model]
        C3[LoRA Injection<br/>Character Identity]
        C4[PuLID / IP-Adapter<br/>Face Preservation]
        C5[ControlNet<br/>Pose & Composition]
        C6[Batch Processing<br/>Multi-seed Generation]
    end

    subgraph Evaluation["Evaluation Layer"]
        D1[InsightFace<br/>Face Embedding Extraction]
        D2[Cosine Similarity<br/>Reference Comparison]
        D3[Quality Scoring<br/>Sharpness, Aesthetics, CLIP]
        D4[Threshold Filtering<br/>≥0.85 Consistency]
    end

    subgraph Video["Video Layer"]
        E1[Keyframe Generation<br/>ComfyUI Interpolation]
        E2[Kling API<br/>Image-to-Video]
        E3[FFmpeg Processing<br/>Encoding & Compilation]
    end

    subgraph PostProd["Post-Production Layer"]
        F1[Watermarking<br/>Configurable Overlay]
        F2[Platform Resizing<br/>IG/Twitter/Portfolio]
        F3[Format Conversion<br/>JPG/PNG/MP4]
        F4[Metadata Injection<br/>EXIF & Tags]
    end

    subgraph Publishing["Publishing Layer"]
        G1[Content Scheduler<br/>Optimal Timing]
        G2[Instagram API<br/>Feed, Stories, Reels]
        G3[Twitter API<br/>Posts & Threads]
        G4[TikTok API<br/>Short-form Video]
    end

    subgraph Metrics["Metrics Layer"]
        H1[Generation Metrics<br/>Throughput, Success Rate]
        H2[Quality Metrics<br/>Consistency Scores]
        H3[Engagement Metrics<br/>Platform Analytics]
        H4[Benchmarking<br/>Historical Comparison]
    end

    %% Flow connections
    A1 --> B1
    A2 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> C3

    A3 --> D1
    C1 --> C6
    C2 --> C1
    C3 --> C1
    C4 --> C1
    C5 --> C1
    C6 --> D1

    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 -->|Passed| E1
    D4 -->|Passed| F1

    E1 --> E2
    E2 --> E3
    E3 --> F2

    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> G1

    G1 --> G2
    G1 --> G3
    G1 --> G4

    G2 --> H3
    G3 --> H3
    G4 --> H3
    D4 --> H2
    C6 --> H1
    H1 --> H4
    H2 --> H4
    H3 --> H4
```

## Pipeline Orchestrator Sequence

```mermaid
sequenceDiagram
    participant CLI as CLI / Task File
    participant Orch as Orchestrator
    participant CUI as ComfyUI
    participant Eval as Evaluator
    participant Kling as Kling API
    participant Post as Post-Production
    participant Pub as Publisher

    CLI->>Orch: execute_task(PipelineTask)
    
    Note over Orch: Stage 1: Generation
    Orch->>CUI: generate_batch(workflow, prompts, seeds)
    CUI-->>Orch: generated_images[]

    Note over Orch: Stage 2: Evaluation
    Orch->>Eval: evaluate(images, references)
    Eval-->>Orch: consistency_scores{}

    Note over Orch: Stage 3: Filtering
    Orch->>Orch: filter(scores >= 0.85)

    Note over Orch: Stage 4: Video (optional)
    Orch->>Kling: image_to_video(filtered_images)
    Kling-->>Orch: video_paths[]

    Note over Orch: Stage 5: Post-Production
    Orch->>Post: batch_watermark(filtered)
    Post-->>Orch: watermarked[]
    Orch->>Post: batch_resize(watermarked, platforms)
    Post-->>Orch: platform_outputs{}

    Note over Orch: Stage 6: Output
    Orch-->>CLI: PipelineResult
```

## Data Flow Table

| Stage | Input | Process | Output | VRAM |
|-------|-------|---------|--------|------|
| Dataset Prep | Raw photos (20-50) | Crop, caption, tag | Training-ready dataset | 0 GB |
| Training | Dataset + base model | Kohya LoRA training | .safetensors LoRA file | 18 GB |
| Generation | Workflow + LoRA + prompts | ComfyUI batch render | Raw images (N×seeds) | 12-16 GB |
| Face Eval | Generated + reference images | InsightFace embedding | Cosine similarity scores | 4 GB |
| Filtering | Scores + threshold | Threshold comparison | Filtered image subset | 0 GB |
| Video Gen | Filtered images + prompt | Kling I2V API | Raw video clips | 0 GB (API) |
| Post-Prod | Images/videos | Watermark + resize | Platform-ready assets | 2 GB |
| Publishing | Assets + schedule | API upload | Published content | 0 GB |
| Metrics | All stage outputs | Aggregate + analyze | Dashboard + reports | 0 GB |

## VRAM Usage Estimates (RTX 4090 - 24GB)

```mermaid
pie title VRAM Allocation During Generation
    "Flux.1 Dev (fp8)" : 12
    "LoRA Weights" : 1
    "PuLID Face" : 3
    "ControlNet" : 2
    "KSampler Working" : 4
    "Available Buffer" : 2
```

### Peak VRAM by Pipeline Stage

| Stage | Peak VRAM | Duration | Can Overlap |
|-------|-----------|----------|-------------|
| Training | 18 GB | 2-4 hours | No |
| Generation (Flux) | 16 GB | 30-60s/image | No |
| Generation (SDXL fallback) | 12 GB | 15-30s/image | No |
| Face Evaluation | 4 GB | 2-5s/image | Yes* |
| Video Post-Processing | 2 GB | 5-10s/video | Yes |

*Face evaluation can overlap with generation on 24GB GPU if using SDXL fallback model.

## Component Dependencies

```mermaid
graph LR
    subgraph External
        ComfyUI[ComfyUI Server]
        Kling[Kling API]
        IG[Instagram API]
        TW[Twitter API]
        TK[TikTok API]
    end

    subgraph Models
        Flux[Flux.1 Dev]
        RVXL[RealVisXL]
        IF[InsightFace]
        PuLID[PuLID]
        IPA[IP-Adapter]
    end

    subgraph Pipeline
        Orch[Orchestrator]
        Batch[Batch Processor]
        Eval[Evaluator]
        VP[Video Pipeline]
        PP[Post-Production]
        Sched[Scheduler]
        Pub[Publisher]
    end

    Orch --> Batch
    Orch --> Eval
    Orch --> VP
    Orch --> PP
    Orch --> Sched
    Sched --> Pub

    Batch --> ComfyUI
    VP --> Kling
    Pub --> IG
    Pub --> TW
    Pub --> TK

    ComfyUI --> Flux
    ComfyUI --> RVXL
    ComfyUI --> PuLID
    ComfyUI --> IPA
    Eval --> IF
```
