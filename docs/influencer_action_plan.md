# Influencer Mode - Action Plan & KPI Framework

## Executive Summary

This document defines the operational strategy for running the AI art pipeline in "influencer mode" — maximizing content output, engagement, and measurable growth metrics while maintaining quality and ethical transparency.

---

## Phase 1: Foundation (Week 1-2)

### Actions
| # | Task | Tool/Script | KPI |
|---|------|-------------|-----|
| 1 | Train 2-3 character LoRAs | `kohya_trainer.py` | Loss < 0.02, consistency > 0.85 |
| 2 | Generate 50+ reference images per character | `comfyui_batch.py` | Pass rate > 80% |
| 3 | Evaluate consistency baselines | `facial_consistency.py` | Mean sim > 0.88 |
| 4 | Create first editorial calendar (2 weeks) | `content_scheduler.py` | 28+ posts scheduled |
| 5 | Set up platform accounts | Manual | Accounts verified |
| 6 | Configure API access | `.env` setup | All APIs responding |

### Success Criteria
- [ ] 3 trained LoRAs with documented trigger words
- [ ] Consistency score > 0.85 for all characters
- [ ] 2-week calendar generated and exported
- [ ] Test post published on each platform

---

## Phase 2: Content Velocity (Week 3-4)

### Daily Operations
```
Morning (automated):
  1. content_scheduler.py → get_todays_posts()
  2. orchestrator.py → execute_batch(today's tasks)
  3. facial_consistency.py → filter quality
  4. post_prod.py → watermark + resize
  5. social_publisher.py → publish at scheduled times

Evening (review):
  6. benchmark.py → daily metrics
  7. Review engagement, adjust next day's prompts
```

### Content Mix Strategy
| Platform | Posts/Day | Content Types | Best Hours |
|----------|-----------|---------------|------------|
| Instagram Feed | 2 | 40% single, 30% carousel, 30% reel | 9am, 12pm, 5pm, 8pm |
| Instagram Stories | 3-5 | Quick behind-the-scenes, polls | 8am-10pm spread |
| Twitter/X | 4 | 50% image, 30% thread, 20% video | 8am, 12pm, 3pm, 7pm, 10pm |

### Character Rotation Rules
- Max 3 consecutive posts of same character
- Weekly character distribution: aim for equal split ±20%
- Introduce "surprise" elements every 7th post (new outfit, style transfer)

---

## Phase 3: Growth & Optimization (Week 5-8)

### A/B Testing Matrix

| Variable | Option A | Option B | Metric |
|----------|----------|----------|--------|
| LoRA weight | 0.7 | 0.85 | Engagement rate |
| Posting time | Morning (9am) | Evening (8pm) | Reach |
| Content type | Single image | Carousel | Saves + Shares |
| Style | Photorealistic | Anime style transfer | Comments |
| Caption length | Short (< 100 chars) | Long (story format) | Time on post |
| Video length | 5s | 10s | Completion rate |

### Engagement Optimization
1. **Caption Strategy**: Include call-to-action, use relevant hashtags
2. **Carousel Design**: Hook on first slide, value on remaining
3. **Reel Strategy**: 3-5s hook, show process/variation, end with result
4. **Thread Strategy**: First tweet is standalone value, thread adds depth

---

## Phase 4: Monetization Research (Week 9-12)

### Revenue Streams (Research)
| Stream | Platform | Expected Range | Notes |
|--------|----------|----------------|-------|
| LoRA sales | CivitAI, Gumroad | $5-50/LoRA | Documented training configs |
| Workflow packs | Gumroad | $10-30/pack | ComfyUI JSON + documentation |
| Prompt engineering guides | Gumroad | $15-40 | Structured prompt templates |
| Commissioned concept art | Direct/Fiverr | $50-200/project | Client knows it's AI-generated |
| Technical tutorials | YouTube/Blog | Ad revenue | Behind-the-scenes pipeline |

### Asset Packaging
Each sellable asset includes:
- LoRA file (.safetensors) + trigger word documentation
- Training config (hyperparameters, dataset requirements)
- Sample outputs (10+ images showing range)
- Recommended workflow JSON
- Usage guide (weights, combinations, compatible models)

---

## KPI Dashboard Metrics

### Content Production KPIs
| Metric | Target | Measurement | Frequency |
|--------|--------|-------------|-----------|
| Posts/week | 14-28 | `content_scheduler.py` | Weekly |
| Generation success rate | > 90% | `orchestrator.py` report | Per batch |
| Avg consistency score | > 0.87 | `facial_consistency.py` | Per batch |
| Pipeline uptime | > 95% | Error rate tracking | Daily |
| Content variety score | > 0.7 | Unique prompts / total | Weekly |

### Quality KPIs
| Metric | Target | Measurement | Frequency |
|--------|--------|-------------|-----------|
| Facial consistency (mean) | > 0.87 | ArcFace cosine sim | Per generation |
| Threshold pass rate | > 85% | % above 0.85 threshold | Per batch |
| Video temporal consistency | > 0.90 | Frame SSIM | Per video |
| Flicker score | < 0.05 | Std of frame diffs | Per video |

### Platform KPIs (to track manually or via API)
| Metric | Instagram Target | Twitter Target | Frequency |
|--------|-----------------|----------------|-----------|
| Follower growth rate | 5-10%/week | 3-7%/week | Weekly |
| Engagement rate | > 5% | > 3% | Per post |
| Reach per post | Growing trend | Growing trend | Weekly |
| Save rate | > 3% | N/A | Per post |
| Share rate | > 1% | > 2% (retweets) | Per post |
| Profile visits | Growing trend | Growing trend | Weekly |
| DMs / Inquiries | Track volume | Track volume | Daily |

### Pipeline Efficiency KPIs
| Metric | Target | Measurement | Frequency |
|--------|--------|-------------|-----------|
| Avg generation time/image | < 30s | Timer in orchestrator | Per batch |
| Avg video generation time | < 5min | Kling polling time | Per video |
| Cost per video (API) | < $0.15 | API billing tracking | Monthly |
| Human intervention time | < 30 min/day | Self-tracked | Daily |
| Storage usage | < 50GB/month | Disk monitoring | Weekly |

---

## Weekly Review Checklist

```markdown
## Week {N} Review - {Date}

### Production
- [ ] Total posts published: ___
- [ ] Success rate: ___%
- [ ] Avg consistency: ___
- [ ] New content types tried: ___

### Growth
- [ ] Instagram followers: ___ (Δ: +___)
- [ ] Twitter followers: ___ (Δ: +___)
- [ ] Best performing post: ___
- [ ] Worst performing post: ___

### Quality
- [ ] Mean facial consistency: ___
- [ ] Videos generated: ___
- [ ] Temporal quality issues: ___

### Learnings
- What worked: ___
- What didn't: ___
- Adjustments for next week: ___

### Pipeline Health
- [ ] Errors encountered: ___
- [ ] Fixes applied: ___
- [ ] New features needed: ___
```

---

## Automation Schedule

### Cron-like Execution Plan

```
# Daily automated tasks
06:00 - Generate today's content batch (orchestrator.py)
07:00 - Evaluate and filter results (facial_consistency.py)
07:30 - Post-produce approved content (post_prod.py)
09:00 - Publish first post (social_publisher.py)
12:00 - Publish second post
17:00 - Publish third post (if applicable)
20:00 - Publish fourth post (if applicable)
23:00 - Generate metrics report (benchmark.py)
23:30 - Backup outputs, rotate logs

# Weekly tasks (Monday)
- Generate new weekly calendar
- Run full benchmark comparison
- Generate LaTeX tables for paper
- Review and adjust character rotation

# Monthly tasks
- Full LoRA retraining evaluation
- Platform analytics deep-dive
- Content strategy adjustment
- Asset packaging for sales
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| API rate limits | Missed posts | Queue + retry with backoff |
| Kling AI downtime | No videos | Skip video, post images only |
| ComfyUI crash | No generation | Health check + auto-restart |
| Low engagement | Growth stall | A/B test, vary content types |
| Account suspension | Total loss | Multi-account, follow TOS |
| VRAM OOM | Pipeline halt | Batch size reduction, queue |
| Quality degradation | Audience loss | Threshold filtering + monitoring |

---

## Ethical Guardrails

1. **Every post** includes #AIGenerated or equivalent disclosure
2. **Every image** is watermarked "AI Generated Concept Art"
3. **No impersonation** of real individuals
4. **No deceptive** framing (never claim human-made)
5. **Metadata** includes AI provenance information
6. **Prompt data** kept private (not exposed in metadata)
7. **Platform TOS** compliance maintained
