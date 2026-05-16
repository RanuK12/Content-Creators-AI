"""
Content Scheduler
==================
Editorial calendar management with prompt template rotation.

Features:
- JSON/CSV-based editorial calendar
- Character rotation with variety enforcement
- Prompt template system with variable substitution
- Optimal posting time calculation per platform
- Weekly/monthly content planning automation

Research relevance:
- Systematic content scheduling enables controlled A/B testing
- Character rotation prevents audience fatigue and measures engagement per character
- Time-based analytics reveal optimal posting windows
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class PromptTemplate:
    """Template for generating varied prompts."""
    id: str
    category: str  # portrait, full_body, scene, close_up, action
    template: str  # Uses {character}, {outfit}, {setting}, {mood}, {style}
    platforms: list[str] = field(default_factory=lambda: ["instagram_feed", "twitter"])
    weight: float = 1.0  # Higher = more likely to be selected
    tags: list[str] = field(default_factory=list)


@dataclass
class Character:
    """Character definition for rotation."""
    name: str
    trigger_word: str
    lora_name: str
    lora_weight: float = 0.8
    description: str = ""
    default_outfits: list[str] = field(default_factory=list)
    default_settings: list[str] = field(default_factory=list)
    mood_pool: list[str] = field(default_factory=list)


@dataclass
class ScheduledPost:
    """A single scheduled content post."""
    id: str
    datetime_utc: str  # ISO format
    platform: str
    character: str
    prompt: str
    workflow: str
    content_type: str  # image, carousel, reel
    status: str = "pending"  # pending, generating, generated, published, failed
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "datetime_utc": self.datetime_utc,
            "platform": self.platform,
            "character": self.character,
            "prompt": self.prompt,
            "workflow": self.workflow,
            "content_type": self.content_type,
            "status": self.status,
            "metadata": self.metadata,
        }


# =============================================================================
# DEFAULT TEMPLATES
# =============================================================================

DEFAULT_TEMPLATES = [
    PromptTemplate(
        id="portrait_studio",
        category="portrait",
        template="{character}, professional studio portrait, {mood} expression, {style} lighting, sharp focus, 8k",
        platforms=["instagram_feed", "twitter"],
        weight=1.5,
        tags=["portrait", "studio", "professional"],
    ),
    PromptTemplate(
        id="portrait_cinematic",
        category="portrait",
        template="{character}, cinematic portrait, {mood}, dramatic lighting, film grain, bokeh background",
        platforms=["instagram_feed"],
        weight=1.2,
        tags=["cinematic", "dramatic"],
    ),
    PromptTemplate(
        id="full_body_outdoor",
        category="full_body",
        template="{character} standing in {setting}, full body shot, {outfit}, natural lighting, golden hour",
        platforms=["instagram_feed", "twitter"],
        weight=1.0,
        tags=["full_body", "outdoor"],
    ),
    PromptTemplate(
        id="close_up_detail",
        category="close_up",
        template="extreme close-up of {character}, {mood} expression, detailed skin texture, {style} color grading",
        platforms=["instagram_feed"],
        weight=0.8,
        tags=["close_up", "detail"],
    ),
    PromptTemplate(
        id="action_dynamic",
        category="action",
        template="{character} in dynamic pose, {setting}, {outfit}, motion blur, energetic, {style}",
        platforms=["instagram_feed", "twitter"],
        weight=0.7,
        tags=["action", "dynamic"],
    ),
    PromptTemplate(
        id="reel_motion",
        category="reel",
        template="{character}, centered portrait, looking at camera, {mood}, clean background, upper body, 9:16",
        platforms=["instagram_story"],
        weight=1.3,
        tags=["reel", "video", "vertical"],
    ),
    PromptTemplate(
        id="carousel_outfit",
        category="carousel",
        template="{character} wearing {outfit}, fashion photography, {setting}, {style} aesthetic",
        platforms=["instagram_feed"],
        weight=1.0,
        tags=["carousel", "fashion", "outfit"],
    ),
    PromptTemplate(
        id="scene_narrative",
        category="scene",
        template="{character} in {setting}, {mood} atmosphere, {outfit}, storytelling composition, cinematic",
        platforms=["instagram_feed", "twitter"],
        weight=0.9,
        tags=["scene", "narrative"],
    ),
]

# Variable pools for template substitution
VARIABLE_POOLS = {
    "outfit": [
        "casual streetwear", "elegant evening dress", "business formal suit",
        "athletic wear", "bohemian style", "minimalist outfit",
        "leather jacket and jeans", "summer floral dress", "cyberpunk attire",
        "vintage retro clothing", "high fashion couture", "cozy sweater",
    ],
    "setting": [
        "a modern city rooftop at sunset", "a misty forest path",
        "a neon-lit Tokyo street at night", "a sunlit Mediterranean cafe",
        "a cozy library with warm lighting", "an industrial loft space",
        "a beach at golden hour", "a rainy Paris street",
        "a futuristic cyberpunk cityscape", "a flower garden in spring",
        "a snowy mountain landscape", "an art gallery with white walls",
    ],
    "mood": [
        "confident", "serene", "playful", "mysterious", "passionate",
        "contemplative", "joyful", "fierce", "elegant", "dreamy",
        "determined", "gentle",
    ],
    "style": [
        "warm tones", "cool blue", "high contrast", "soft pastel",
        "moody dark", "vibrant saturated", "vintage film",
        "neon glow", "natural", "editorial fashion",
    ],
}


class ContentScheduler:
    """
    Manages editorial calendar and content generation scheduling.
    
    Coordinates character rotation, prompt variation, and optimal
    posting times across platforms.
    """

    def __init__(
        self,
        characters: list[Character],
        templates: Optional[list[PromptTemplate]] = None,
        variable_pools: Optional[dict[str, list[str]]] = None,
        max_same_character_consecutive: int = 3,
        calendar_path: str | Path = "output/calendar/",
    ):
        """
        Initialize content scheduler.
        
        Args:
            characters: List of Character definitions
            templates: Prompt templates (uses defaults if None)
            variable_pools: Variable substitution pools
            max_same_character_consecutive: Max consecutive posts with same character
            calendar_path: Directory for calendar files
        """
        self.characters = {c.name: c for c in characters}
        self.templates = templates or DEFAULT_TEMPLATES
        self.variable_pools = variable_pools or VARIABLE_POOLS
        self.max_consecutive = max_same_character_consecutive
        self.calendar_path = Path(calendar_path)
        self.calendar_path.mkdir(parents=True, exist_ok=True)
        
        self._schedule: list[ScheduledPost] = []
        self._last_characters: list[str] = []
        
        logger.info(
            f"ContentScheduler | {len(self.characters)} characters | "
            f"{len(self.templates)} templates | max_consecutive={max_same_character_consecutive}"
        )

    def _select_character(self) -> Character:
        """
        Select next character with variety enforcement.
        
        Prevents more than max_consecutive posts of the same character.
        Uses weighted random selection favoring less recently used characters.
        """
        available = list(self.characters.values())
        
        # Check consecutive constraint
        if len(self._last_characters) >= self.max_consecutive:
            recent = self._last_characters[-self.max_consecutive:]
            if len(set(recent)) == 1:
                # All recent were same character, exclude it
                exclude = recent[0]
                available = [c for c in available if c.name != exclude]
        
        if not available:
            available = list(self.characters.values())
        
        selected = random.choice(available)
        self._last_characters.append(selected.name)
        
        # Keep history manageable
        if len(self._last_characters) > 20:
            self._last_characters = self._last_characters[-10:]
        
        return selected

    def _select_template(self, platform: str, content_type: str = "image") -> PromptTemplate:
        """Select weighted random template appropriate for platform."""
        eligible = [
            t for t in self.templates
            if platform in t.platforms or any(p.startswith(platform.split("_")[0]) for p in t.platforms)
        ]
        
        if content_type == "reel":
            eligible = [t for t in eligible if "reel" in t.tags or "video" in t.tags] or eligible
        elif content_type == "carousel":
            eligible = [t for t in eligible if "carousel" in t.tags] or eligible
        
        if not eligible:
            eligible = self.templates
        
        weights = [t.weight for t in eligible]
        return random.choices(eligible, weights=weights, k=1)[0]

    def _fill_template(self, template: PromptTemplate, character: Character) -> str:
        """
        Fill template with character info and random variables.
        
        Substitutes: {character}, {outfit}, {setting}, {mood}, {style}
        """
        prompt = template.template
        
        # Character trigger word
        prompt = prompt.replace("{character}", character.trigger_word)
        
        # Fill from character-specific pools first, then global
        for var_name, pool in self.variable_pools.items():
            placeholder = f"{{{var_name}}}"
            if placeholder in prompt:
                # Check character-specific options
                char_pool = getattr(character, f"default_{var_name}s", [])
                source = char_pool if char_pool else pool
                prompt = prompt.replace(placeholder, random.choice(source))
        
        return prompt

    def _get_posting_times(
        self,
        platform: str,
        date: datetime,
        posts_per_day: int = 2,
    ) -> list[datetime]:
        """Get optimal posting times for a platform on a given date."""
        # Platform-specific best hours
        best_hours = {
            "instagram_feed": [9, 12, 17, 20],
            "instagram_story": [8, 11, 14, 18, 21],
            "twitter": [8, 12, 15, 19, 22],
        }
        
        hours = best_hours.get(platform, [9, 12, 17, 20])
        selected_hours = random.sample(hours, min(posts_per_day, len(hours)))
        selected_hours.sort()
        
        times = []
        for hour in selected_hours:
            minute = random.randint(0, 45)
            post_time = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            times.append(post_time)
        
        return times

    def _determine_content_type(self, platform: str) -> str:
        """Randomly select content type based on platform ratios."""
        ratios = {
            "instagram_feed": {"image": 0.4, "carousel": 0.3, "reel": 0.3},
            "instagram_story": {"reel": 0.7, "image": 0.3},
            "twitter": {"image": 0.5, "thread": 0.3, "video": 0.2},
        }
        
        platform_ratios = ratios.get(platform, {"image": 1.0})
        types = list(platform_ratios.keys())
        weights = list(platform_ratios.values())
        
        return random.choices(types, weights=weights, k=1)[0]

    def _determine_workflow(self, content_type: str) -> str:
        """Map content type to appropriate workflow."""
        workflow_map = {
            "image": "workflows/workflow_a_character_consistency/workflow_api.json",
            "carousel": "workflows/workflow_b_outfit_variation/workflow_api.json",
            "reel": "workflows/workflow_c_motion_reel/workflow_api.json",
            "video": "workflows/workflow_c_motion_reel/workflow_api.json",
            "thread": "workflows/workflow_a_character_consistency/workflow_api.json",
        }
        return workflow_map.get(content_type, workflow_map["image"])

    def generate_weekly_schedule(
        self,
        start_date: Optional[datetime] = None,
        platforms: Optional[list[str]] = None,
        posts_per_day_per_platform: int = 2,
    ) -> list[ScheduledPost]:
        """
        Generate a full week of scheduled content.
        
        Creates a balanced mix of characters, content types, and platforms
        with optimal posting times.
        
        Args:
            start_date: Schedule start (defaults to tomorrow)
            platforms: Target platforms
            posts_per_day_per_platform: Posts per day per platform
            
        Returns:
            List of ScheduledPost objects
        """
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
        
        if platforms is None:
            platforms = ["instagram_feed", "twitter"]
        
        schedule = []
        post_counter = 0
        
        for day_offset in range(7):
            current_date = start_date + timedelta(days=day_offset)
            
            for platform in platforms:
                posting_times = self._get_posting_times(
                    platform, current_date, posts_per_day_per_platform
                )
                
                for post_time in posting_times:
                    character = self._select_character()
                    content_type = self._determine_content_type(platform)
                    template = self._select_template(platform, content_type)
                    prompt = self._fill_template(template, character)
                    workflow = self._determine_workflow(content_type)
                    
                    post = ScheduledPost(
                        id=f"post_{post_counter:04d}",
                        datetime_utc=post_time.isoformat(),
                        platform=platform,
                        character=character.name,
                        prompt=prompt,
                        workflow=workflow,
                        content_type=content_type,
                        metadata={
                            "template_id": template.id,
                            "template_category": template.category,
                            "lora_name": character.lora_name,
                            "lora_weight": character.lora_weight,
                            "tags": template.tags,
                        },
                    )
                    
                    schedule.append(post)
                    post_counter += 1
        
        self._schedule = schedule
        logger.info(
            f"Weekly schedule generated: {len(schedule)} posts | "
            f"{len(platforms)} platforms | {posts_per_day_per_platform}/day/platform"
        )
        
        return schedule

    def generate_monthly_schedule(
        self,
        start_date: Optional[datetime] = None,
        platforms: Optional[list[str]] = None,
        posts_per_day_per_platform: int = 2,
    ) -> list[ScheduledPost]:
        """Generate 30 days of content schedule."""
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
        
        all_posts = []
        for week in range(4):
            week_start = start_date + timedelta(weeks=week)
            week_posts = self.generate_weekly_schedule(
                start_date=week_start,
                platforms=platforms,
                posts_per_day_per_platform=posts_per_day_per_platform,
            )
            all_posts.extend(week_posts)
        
        # Re-index
        for i, post in enumerate(all_posts):
            post.id = f"post_{i:04d}"
        
        self._schedule = all_posts
        logger.info(f"Monthly schedule: {len(all_posts)} total posts")
        return all_posts

    def export_calendar_json(self, output_path: Optional[str | Path] = None) -> Path:
        """Export schedule as JSON."""
        if output_path is None:
            output_path = self.calendar_path / "editorial_calendar.json"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_posts": len(self._schedule),
                "characters": list(self.characters.keys()),
                "platforms": list(set(p.platform for p in self._schedule)),
            },
            "schedule": [post.to_dict() for post in self._schedule],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Calendar exported: {output_path}")
        return output_path

    def export_calendar_csv(self, output_path: Optional[str | Path] = None) -> Path:
        """Export schedule as CSV for spreadsheet compatibility."""
        if output_path is None:
            output_path = self.calendar_path / "editorial_calendar.csv"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "datetime_utc", "platform", "character",
                "content_type", "workflow", "prompt", "status",
                "template_id", "lora_name", "tags",
            ])
            
            for post in self._schedule:
                writer.writerow([
                    post.id,
                    post.datetime_utc,
                    post.platform,
                    post.character,
                    post.content_type,
                    post.workflow,
                    post.prompt,
                    post.status,
                    post.metadata.get("template_id", ""),
                    post.metadata.get("lora_name", ""),
                    "|".join(post.metadata.get("tags", [])),
                ])
        
        logger.info(f"CSV calendar exported: {output_path}")
        return output_path

    def get_pending_posts(self, platform: Optional[str] = None) -> list[ScheduledPost]:
        """Get pending posts, optionally filtered by platform."""
        posts = [p for p in self._schedule if p.status == "pending"]
        if platform:
            posts = [p for p in posts if p.platform == platform]
        return posts

    def get_todays_posts(self) -> list[ScheduledPost]:
        """Get all posts scheduled for today."""
        today = datetime.now().date().isoformat()
        return [
            p for p in self._schedule
            if p.datetime_utc.startswith(today)
        ]

    def mark_post_status(self, post_id: str, status: str) -> None:
        """Update post status (generating, generated, published, failed)."""
        for post in self._schedule:
            if post.id == post_id:
                post.status = status
                logger.debug(f"Post {post_id} → {status}")
                return
        logger.warning(f"Post {post_id} not found")

    def get_analytics_summary(self) -> dict:
        """Generate summary analytics of the schedule."""
        if not self._schedule:
            return {}
        
        # Character distribution
        char_counts = {}
        for post in self._schedule:
            char_counts[post.character] = char_counts.get(post.character, 0) + 1
        
        # Platform distribution
        platform_counts = {}
        for post in self._schedule:
            platform_counts[post.platform] = platform_counts.get(post.platform, 0) + 1
        
        # Content type distribution
        type_counts = {}
        for post in self._schedule:
            type_counts[post.content_type] = type_counts.get(post.content_type, 0) + 1
        
        # Status summary
        status_counts = {}
        for post in self._schedule:
            status_counts[post.status] = status_counts.get(post.status, 0) + 1
        
        return {
            "total_posts": len(self._schedule),
            "character_distribution": char_counts,
            "platform_distribution": platform_counts,
            "content_type_distribution": type_counts,
            "status_summary": status_counts,
            "variety_score": len(set(p.character for p in self._schedule)) / len(self.characters) if self.characters else 0,
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point for content scheduling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate content editorial calendar")
    parser.add_argument("--period", choices=["week", "month"], default="week", help="Schedule period")
    parser.add_argument("--output", "-o", default="output/calendar/", help="Output directory")
    parser.add_argument("--posts-per-day", type=int, default=2, help="Posts per day per platform")
    parser.add_argument("--platforms", nargs="+", default=["instagram_feed", "twitter"])
    
    args = parser.parse_args()
    
    # Example characters
    characters = [
        Character(
            name="character_01",
            trigger_word="ohwx",
            lora_name="character_01.safetensors",
            lora_weight=0.8,
            description="Main character - versatile concept art subject",
            default_outfits=["casual modern outfit", "elegant evening wear"],
            mood_pool=["confident", "serene", "mysterious"],
        ),
        Character(
            name="character_02",
            trigger_word="sks_char",
            lora_name="character_02.safetensors",
            lora_weight=0.75,
            description="Secondary character - action-oriented",
            default_outfits=["sporty attire", "streetwear"],
            mood_pool=["fierce", "determined", "playful"],
        ),
    ]
    
    scheduler = ContentScheduler(
        characters=characters,
        calendar_path=args.output,
    )
    
    if args.period == "week":
        scheduler.generate_weekly_schedule(
            platforms=args.platforms,
            posts_per_day_per_platform=args.posts_per_day,
        )
    else:
        scheduler.generate_monthly_schedule(
            platforms=args.platforms,
            posts_per_day_per_platform=args.posts_per_day,
        )
    
    # Export both formats
    scheduler.export_calendar_json()
    scheduler.export_calendar_csv()
    
    # Print summary
    summary = scheduler.get_analytics_summary()
    print(f"\nSchedule Generated:")
    print(f"  Total posts: {summary['total_posts']}")
    print(f"  Characters: {summary['character_distribution']}")
    print(f"  Platforms: {summary['platform_distribution']}")
    print(f"  Content types: {summary['content_type_distribution']}")
    print(f"  Variety score: {summary['variety_score']:.2f}")


if __name__ == "__main__":
    main()
