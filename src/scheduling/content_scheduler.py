"""
Content Scheduling Module
==========================
Manages the automated scheduling and generation of AI influencer content
across multiple social media platforms. Implements variety-enforced character
rotation, template-based prompt generation, and calendar export functionality.

This module provides the bridge between AI content generation pipelines and
social media publishing, ensuring consistent posting cadence and content diversity.

Features:
- Template-based prompt generation with variable substitution
- Character rotation with variety enforcement (no consecutive repeats)
- Platform-specific posting time optimization
- Weekly and monthly schedule generation
- Calendar export (JSON and CSV formats)
- Analytics summary for content planning
"""

from __future__ import annotations

import csv
import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class PromptTemplate:
    """
    A reusable prompt template for content generation.

    Templates use variable placeholders that are filled at scheduling time
    to create diverse prompts while maintaining consistent quality.

    Attributes:
        id: Unique template identifier.
        category: Content category (portrait, full_body, close_up, action, reel, carousel, scene).
        template: Prompt string with {character}, {outfit}, {setting}, {mood}, {style} variables.
        platforms: List of platforms this template is suitable for.
        weight: Selection probability weight (higher = more likely to be chosen).
        tags: Descriptive tags for filtering and analytics.
    """

    id: str
    category: str
    template: str
    platforms: list[str]
    weight: float = 1.0
    tags: list[str] = field(default_factory=list)


@dataclass
class Character:
    """
    An AI influencer character definition.

    Attributes:
        name: Display name of the character.
        trigger_word: LoRA trigger word for generation.
        lora_name: Filename of the LoRA model.
        lora_weight: LoRA strength during inference (0.0-1.0).
        default_outfits: List of outfit descriptions this character commonly wears.
        settings: List of typical environments/backgrounds.
        mood_pool: List of moods/expressions suitable for this character.
    """

    name: str
    trigger_word: str
    lora_name: str
    lora_weight: float = 0.85
    default_outfits: list[str] = field(default_factory=list)
    settings: list[str] = field(default_factory=list)
    mood_pool: list[str] = field(default_factory=list)


@dataclass
class ScheduledPost:
    """
    A fully resolved scheduled content post.

    Attributes:
        id: Unique post identifier (UUID).
        datetime_utc: Scheduled publication time in UTC.
        platform: Target platform (instagram, twitter, tiktok).
        character: Character assigned to this post.
        prompt: Fully resolved generation prompt.
        workflow: ComfyUI workflow to use for generation.
        content_type: Type of content (image, carousel, reel, story).
        status: Current status (scheduled, generating, generated, published, failed).
        metadata: Additional metadata for tracking and analytics.
    """

    id: str
    datetime_utc: datetime
    platform: str
    character: Character
    prompt: str
    workflow: str
    content_type: str
    status: str = "scheduled"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        return {
            "id": self.id,
            "datetime_utc": self.datetime_utc.isoformat(),
            "platform": self.platform,
            "character": self.character.name,
            "trigger_word": self.character.trigger_word,
            "lora_name": self.character.lora_name,
            "prompt": self.prompt,
            "workflow": self.workflow,
            "content_type": self.content_type,
            "status": self.status,
            "metadata": self.metadata,
        }


# ==============================================================================
# Default Templates
# ==============================================================================

DEFAULT_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        id="portrait_01",
        category="portrait",
        template="{character}, {outfit}, portrait shot, {setting}, {mood} expression, {style}, 8k uhd, sharp focus",
        platforms=["instagram", "twitter"],
        weight=1.5,
        tags=["portrait", "headshot", "identity"],
    ),
    PromptTemplate(
        id="full_body_01",
        category="full_body",
        template="{character}, {outfit}, full body shot, standing pose, {setting}, {mood}, {style}, professional photography",
        platforms=["instagram", "twitter"],
        weight=1.2,
        tags=["full_body", "fashion", "pose"],
    ),
    PromptTemplate(
        id="close_up_01",
        category="close_up",
        template="{character}, extreme close-up, {mood} expression, {style}, shallow depth of field, {setting} background bokeh",
        platforms=["instagram", "twitter", "tiktok"],
        weight=1.0,
        tags=["close_up", "detail", "expression"],
    ),
    PromptTemplate(
        id="action_01",
        category="action",
        template="{character}, {outfit}, dynamic action pose, {setting}, {mood}, {style}, motion blur accents, cinematic",
        platforms=["instagram", "tiktok"],
        weight=0.8,
        tags=["action", "dynamic", "motion"],
    ),
    PromptTemplate(
        id="reel_01",
        category="reel",
        template="{character}, {outfit}, video portrait, {setting}, {mood}, {style}, smooth motion, 24fps cinematic",
        platforms=["instagram", "tiktok"],
        weight=1.0,
        tags=["video", "reel", "motion"],
    ),
    PromptTemplate(
        id="carousel_01",
        category="carousel",
        template="{character}, {outfit}, editorial photoshoot, multiple angles, {setting}, {mood}, {style}, magazine quality",
        platforms=["instagram"],
        weight=0.9,
        tags=["carousel", "editorial", "multi-image"],
    ),
    PromptTemplate(
        id="scene_01",
        category="scene",
        template="{character}, {outfit}, environmental portrait, {setting}, {mood} atmosphere, {style}, wide angle, storytelling",
        platforms=["instagram", "twitter"],
        weight=1.1,
        tags=["scene", "environment", "storytelling"],
    ),
    PromptTemplate(
        id="lifestyle_01",
        category="scene",
        template="{character}, {outfit}, lifestyle photography, {setting}, {mood}, {style}, natural lighting, candid moment",
        platforms=["instagram", "twitter", "tiktok"],
        weight=1.3,
        tags=["lifestyle", "candid", "natural"],
    ),
]


# ==============================================================================
# Variable Pools
# ==============================================================================

VARIABLE_POOLS: dict[str, list[str]] = {
    "outfit": [
        "elegant black dress",
        "casual white t-shirt and jeans",
        "business blazer and pencil skirt",
        "bohemian floral maxi dress",
        "athletic wear with sneakers",
        "leather jacket and boots",
        "summer sundress",
        "cozy oversized sweater",
        "cocktail dress with heels",
        "streetwear hoodie and cargo pants",
        "vintage denim jacket ensemble",
        "silk blouse and tailored pants",
    ],
    "setting": [
        "modern city rooftop at golden hour",
        "cozy coffee shop interior",
        "lush botanical garden",
        "minimalist white studio",
        "neon-lit urban alley at night",
        "sun-drenched beach at sunset",
        "elegant art gallery",
        "rustic countryside meadow",
        "luxury penthouse with city view",
        "vibrant street market",
        "serene Japanese garden",
        "industrial loft with exposed brick",
    ],
    "mood": [
        "confident and radiant",
        "serene and contemplative",
        "playful and joyful",
        "mysterious and alluring",
        "warm and inviting",
        "fierce and determined",
        "dreamy and ethereal",
        "sophisticated and poised",
        "energetic and vibrant",
        "gentle and tender",
        "bold and edgy",
        "relaxed and carefree",
    ],
    "style": [
        "cinematic color grading",
        "soft natural lighting",
        "dramatic chiaroscuro",
        "pastel aesthetic",
        "high fashion editorial",
        "warm golden tones",
        "cool blue hour tones",
        "film grain analog",
        "clean modern minimal",
        "rich moody shadows",
        "bright and airy",
        "vintage film look",
    ],
}


class ContentScheduler:
    """
    Manages content scheduling for AI influencer accounts.

    Implements variety-enforced scheduling to ensure diverse content
    across characters, templates, and visual styles while maintaining
    optimal posting cadence for each platform.

    Attributes:
        characters: List of available AI characters.
        templates: List of prompt templates.
        platforms: Active platforms with posting configurations.
        posts_per_day: Target number of posts per day per platform.
    """

    def __init__(
        self,
        characters: list[Character],
        templates: Optional[list[PromptTemplate]] = None,
        platforms: Optional[list[str]] = None,
        posts_per_day: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Initialize the content scheduler.

        Args:
            characters: List of AI characters to schedule content for.
            templates: Prompt templates (defaults to DEFAULT_TEMPLATES).
            platforms: Active platforms (defaults to instagram, twitter, tiktok).
            posts_per_day: Number of posts per day per platform.
            seed: Random seed for reproducible schedules.
        """
        self.characters = characters
        self.templates = templates or DEFAULT_TEMPLATES
        self.platforms = platforms or ["instagram", "twitter", "tiktok"]
        self.posts_per_day = posts_per_day
        self._rng = random.Random(seed)
        self._last_character_idx: dict[str, int] = {}
        self._character_usage_count: dict[str, int] = {c.name: 0 for c in characters}
        self._template_history: list[str] = []

        logger.info(
            f"ContentScheduler initialized | "
            f"characters={len(characters)} | templates={len(self.templates)} | "
            f"platforms={self.platforms} | posts_per_day={posts_per_day}"
        )

    def _select_character(self, platform: str) -> Character:
        """
        Select a character with variety enforcement.

        Ensures no consecutive posts feature the same character on a given
        platform, and balances overall character usage across the schedule.

        Args:
            platform: The target platform for this post.

        Returns:
            Selected Character instance.
        """
        if len(self.characters) == 1:
            return self.characters[0]

        last_idx = self._last_character_idx.get(platform, -1)

        # Filter out last used character for variety
        available = [
            (i, c) for i, c in enumerate(self.characters) if i != last_idx
        ]

        # Weight by inverse usage count for balance
        min_usage = min(self._character_usage_count.values()) if self._character_usage_count else 0
        weights = [
            max(1, 3 - (self._character_usage_count.get(c.name, 0) - min_usage))
            for _, c in available
        ]

        # Weighted random selection
        selected_idx, selected_char = self._rng.choices(available, weights=weights, k=1)[0]

        self._last_character_idx[platform] = selected_idx
        self._character_usage_count[selected_char.name] = (
            self._character_usage_count.get(selected_char.name, 0) + 1
        )

        return selected_char

    def _select_template(self, platform: str) -> PromptTemplate:
        """
        Select a prompt template suitable for the platform.

        Applies weighted random selection with recency bias to avoid
        repeating the same template consecutively.

        Args:
            platform: Target platform.

        Returns:
            Selected PromptTemplate.
        """
        eligible = [t for t in self.templates if platform in t.platforms]
        if not eligible:
            eligible = self.templates

        # Reduce weight of recently used templates
        weights = []
        for t in eligible:
            w = t.weight
            if t.id in self._template_history[-3:]:
                w *= 0.3
            weights.append(w)

        selected = self._rng.choices(eligible, weights=weights, k=1)[0]
        self._template_history.append(selected.id)

        # Keep history bounded
        if len(self._template_history) > 20:
            self._template_history = self._template_history[-20:]

        return selected

    def _fill_template(self, template: PromptTemplate, character: Character) -> str:
        """
        Fill a prompt template with character-specific and random variables.

        Uses character-specific pools when available, falling back to
        global VARIABLE_POOLS for variety.

        Args:
            template: The prompt template to fill.
            character: The character for this post.

        Returns:
            Fully resolved prompt string.
        """
        # Select variables
        outfit = self._rng.choice(
            character.default_outfits if character.default_outfits else VARIABLE_POOLS["outfit"]
        )
        setting = self._rng.choice(
            character.settings if character.settings else VARIABLE_POOLS["setting"]
        )
        mood = self._rng.choice(
            character.mood_pool if character.mood_pool else VARIABLE_POOLS["mood"]
        )
        style = self._rng.choice(VARIABLE_POOLS["style"])

        # Fill template
        prompt = template.template.format(
            character=character.trigger_word,
            outfit=outfit,
            setting=setting,
            mood=mood,
            style=style,
        )

        return prompt

    def _get_posting_times(self, date: datetime, platform: str) -> list[datetime]:
        """
        Generate optimal posting times for a given date and platform.

        Based on general social media engagement research for optimal
        posting windows.

        Args:
            date: The date to generate posting times for.
            platform: Target platform.

        Returns:
            List of datetime objects for optimal posting times.
        """
        # Optimal posting hours by platform (UTC)
        optimal_hours: dict[str, list[int]] = {
            "instagram": [11, 14, 17, 19, 21],
            "twitter": [9, 12, 15, 18, 20],
            "tiktok": [10, 13, 16, 19, 21],
        }

        hours = optimal_hours.get(platform, [10, 14, 18, 20])

        # Select posting times based on posts_per_day
        selected_hours = self._rng.sample(
            hours, min(self.posts_per_day, len(hours))
        )
        selected_hours.sort()

        times = []
        for hour in selected_hours:
            minute = self._rng.randint(0, 45)
            post_time = date.replace(
                hour=hour, minute=minute, second=0, microsecond=0,
                tzinfo=timezone.utc,
            )
            times.append(post_time)

        return times

    def _determine_workflow(self, template: PromptTemplate) -> str:
        """Determine which ComfyUI workflow to use based on content type."""
        workflow_map = {
            "portrait": "workflow_a_character_consistency",
            "full_body": "workflow_b_outfit_variation",
            "close_up": "workflow_a_character_consistency",
            "action": "workflow_b_outfit_variation",
            "reel": "workflow_c_motion_reel",
            "carousel": "workflow_b_outfit_variation",
            "scene": "workflow_b_outfit_variation",
        }
        return workflow_map.get(template.category, "workflow_a_character_consistency")

    def _determine_content_type(self, template: PromptTemplate) -> str:
        """Determine content type from template category."""
        type_map = {
            "portrait": "image",
            "full_body": "image",
            "close_up": "image",
            "action": "image",
            "reel": "reel",
            "carousel": "carousel",
            "scene": "image",
        }
        return type_map.get(template.category, "image")

    def generate_weekly_schedule(
        self, start_date: Optional[datetime] = None
    ) -> list[ScheduledPost]:
        """
        Generate a complete weekly content schedule.

        Creates posts for each platform for 7 days starting from the
        given date, with variety enforcement across characters and templates.

        Args:
            start_date: Starting date for the week. Defaults to next Monday.

        Returns:
            List of ScheduledPost objects for the entire week.
        """
        if start_date is None:
            today = datetime.now(timezone.utc)
            days_until_monday = (7 - today.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            start_date = today + timedelta(days=days_until_monday)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        schedule: list[ScheduledPost] = []

        for day_offset in range(7):
            current_date = start_date + timedelta(days=day_offset)

            for platform in self.platforms:
                posting_times = self._get_posting_times(current_date, platform)

                for post_time in posting_times:
                    character = self._select_character(platform)
                    template = self._select_template(platform)
                    prompt = self._fill_template(template, character)
                    workflow = self._determine_workflow(template)
                    content_type = self._determine_content_type(template)

                    post = ScheduledPost(
                        id=str(uuid.uuid4()),
                        datetime_utc=post_time,
                        platform=platform,
                        character=character,
                        prompt=prompt,
                        workflow=workflow,
                        content_type=content_type,
                        metadata={
                            "template_id": template.id,
                            "template_category": template.category,
                            "day_of_week": current_date.strftime("%A"),
                        },
                    )
                    schedule.append(post)

        schedule.sort(key=lambda p: p.datetime_utc)
        logger.info(f"Generated weekly schedule: {len(schedule)} posts across {len(self.platforms)} platforms")
        return schedule

    def generate_monthly_schedule(
        self, start_date: Optional[datetime] = None, weeks: int = 4
    ) -> list[ScheduledPost]:
        """
        Generate a monthly content schedule (4 weeks by default).

        Args:
            start_date: Starting date. Defaults to next Monday.
            weeks: Number of weeks to schedule.

        Returns:
            List of ScheduledPost objects for the entire month.
        """
        if start_date is None:
            today = datetime.now(timezone.utc)
            days_until_monday = (7 - today.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            start_date = today + timedelta(days=days_until_monday)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        all_posts: list[ScheduledPost] = []

        for week in range(weeks):
            week_start = start_date + timedelta(weeks=week)
            weekly_posts = self.generate_weekly_schedule(week_start)
            all_posts.extend(weekly_posts)

        logger.info(f"Generated monthly schedule: {len(all_posts)} posts over {weeks} weeks")
        return all_posts

    def export_calendar_json(
        self, schedule: list[ScheduledPost], output_path: str | Path
    ) -> Path:
        """
        Export schedule to a JSON calendar file.

        Args:
            schedule: List of scheduled posts.
            output_path: Output file path.

        Returns:
            Path to the saved JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        calendar_data = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "total_posts": len(schedule),
                "platforms": list(set(p.platform for p in schedule)),
                "characters": list(set(p.character.name for p in schedule)),
                "date_range": {
                    "start": schedule[0].datetime_utc.isoformat() if schedule else None,
                    "end": schedule[-1].datetime_utc.isoformat() if schedule else None,
                },
            },
            "schedule": [post.to_dict() for post in schedule],
        }

        with open(output_path, "w") as f:
            json.dump(calendar_data, f, indent=2)

        logger.info(f"Calendar JSON exported: {output_path} ({len(schedule)} posts)")
        return output_path

    def export_calendar_csv(
        self, schedule: list[ScheduledPost], output_path: str | Path
    ) -> Path:
        """
        Export schedule to a CSV file for spreadsheet compatibility.

        Args:
            schedule: List of scheduled posts.
            output_path: Output file path.

        Returns:
            Path to the saved CSV file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "id", "datetime_utc", "platform", "character", "content_type",
            "workflow", "template_category", "prompt", "status",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for post in schedule:
                writer.writerow({
                    "id": post.id,
                    "datetime_utc": post.datetime_utc.isoformat(),
                    "platform": post.platform,
                    "character": post.character.name,
                    "content_type": post.content_type,
                    "workflow": post.workflow,
                    "template_category": post.metadata.get("template_category", ""),
                    "prompt": post.prompt,
                    "status": post.status,
                })

        logger.info(f"Calendar CSV exported: {output_path} ({len(schedule)} posts)")
        return output_path

    def get_analytics_summary(self, schedule: list[ScheduledPost]) -> dict:
        """
        Generate analytics summary for a content schedule.

        Provides insights into content distribution across platforms,
        characters, content types, and posting patterns.

        Args:
            schedule: List of scheduled posts to analyze.

        Returns:
            Dictionary with analytics breakdown.
        """
        if not schedule:
            return {"error": "Empty schedule"}

        # Platform distribution
        platform_counts: dict[str, int] = {}
        for post in schedule:
            platform_counts[post.platform] = platform_counts.get(post.platform, 0) + 1

        # Character distribution
        character_counts: dict[str, int] = {}
        for post in schedule:
            name = post.character.name
            character_counts[name] = character_counts.get(name, 0) + 1

        # Content type distribution
        type_counts: dict[str, int] = {}
        for post in schedule:
            type_counts[post.content_type] = type_counts.get(post.content_type, 0) + 1

        # Workflow distribution
        workflow_counts: dict[str, int] = {}
        for post in schedule:
            workflow_counts[post.workflow] = workflow_counts.get(post.workflow, 0) + 1

        # Day of week distribution
        day_counts: dict[str, int] = {}
        for post in schedule:
            day = post.datetime_utc.strftime("%A")
            day_counts[day] = day_counts.get(day, 0) + 1

        # Time range
        dates = [p.datetime_utc for p in schedule]
        date_range_days = (max(dates) - min(dates)).days + 1

        summary = {
            "total_posts": len(schedule),
            "date_range_days": date_range_days,
            "posts_per_day": round(len(schedule) / max(date_range_days, 1), 1),
            "platform_distribution": platform_counts,
            "character_distribution": character_counts,
            "content_type_distribution": type_counts,
            "workflow_distribution": workflow_counts,
            "day_of_week_distribution": day_counts,
            "unique_characters": len(character_counts),
            "unique_platforms": len(platform_counts),
        }

        return summary


def main():
    """CLI entry point for content scheduling."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Influencer Content Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--period",
        choices=["week", "month"],
        default="week",
        help="Schedule period to generate",
    )
    parser.add_argument(
        "--posts-per-day",
        type=int,
        default=2,
        help="Number of posts per platform per day",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/schedules/schedule.json",
        help="Output path for JSON calendar",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="output/schedules/schedule.csv",
        help="Output path for CSV calendar",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible schedules",
    )
    parser.add_argument(
        "--analytics",
        action="store_true",
        help="Print analytics summary",
    )

    args = parser.parse_args()

    # Create example characters
    characters = [
        Character(
            name="Luna",
            trigger_word="luna_character",
            lora_name="luna_v3.safetensors",
            lora_weight=0.85,
            default_outfits=["elegant black dress", "casual denim", "white blouse"],
            settings=["modern city", "coffee shop", "rooftop garden"],
            mood_pool=["confident", "warm", "playful"],
        ),
        Character(
            name="Aria",
            trigger_word="aria_character",
            lora_name="aria_v2.safetensors",
            lora_weight=0.80,
            default_outfits=["bohemian dress", "athletic wear", "summer outfit"],
            settings=["beach", "botanical garden", "art studio"],
            mood_pool=["serene", "joyful", "dreamy"],
        ),
    ]

    # Initialize scheduler
    scheduler = ContentScheduler(
        characters=characters,
        posts_per_day=args.posts_per_day,
        seed=args.seed,
    )

    # Generate schedule
    if args.period == "week":
        schedule = scheduler.generate_weekly_schedule()
    else:
        schedule = scheduler.generate_monthly_schedule()

    # Export
    scheduler.export_calendar_json(schedule, args.output_json)
    scheduler.export_calendar_csv(schedule, args.output_csv)

    # Analytics
    if args.analytics:
        summary = scheduler.get_analytics_summary(schedule)
        print(f"\n{'='*50}")
        print(f"  CONTENT SCHEDULE ANALYTICS")
        print(f"{'='*50}")
        print(f"  Total posts:     {summary['total_posts']}")
        print(f"  Date range:      {summary['date_range_days']} days")
        print(f"  Posts/day:       {summary['posts_per_day']}")
        print(f"\n  Platform distribution:")
        for platform, count in summary["platform_distribution"].items():
            print(f"    {platform}: {count}")
        print(f"\n  Character distribution:")
        for char, count in summary["character_distribution"].items():
            print(f"    {char}: {count}")
        print(f"\n  Content types:")
        for ctype, count in summary["content_type_distribution"].items():
            print(f"    {ctype}: {count}")
        print(f"{'='*50}\n")

    print(f"Schedule generated: {len(schedule)} posts")
    print(f"  JSON: {args.output_json}")
    print(f"  CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
