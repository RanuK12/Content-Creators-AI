"""
Social Media Publishing Module
===============================
Handles automated publishing of AI-generated content to social media platforms.
Implements platform-specific APIs for Instagram (Graph API v18.0) and
Twitter/X (API v2) with full async support.

All published content includes mandatory AI disclosure (#AIGenerated) to comply
with platform transparency policies and emerging AI content labeling regulations.

Features:
- Instagram: Single image posts, carousel posts, Reels via Graph API v18.0
- Twitter/X: Single tweets with media, threads via API v2
- Async implementation with aiohttp for non-blocking I/O
- Automatic AI disclosure enforcement in all captions
- Rate limiting and retry logic
- Publishing orchestration across multiple platforms

References:
- Meta Graph API v18.0 documentation
- Twitter API v2 documentation
- EU AI Act transparency requirements (2024)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger


# ==============================================================================
# Constants
# ==============================================================================

AI_DISCLOSURE_TAG = "#AIGenerated"
AI_DISCLOSURE_TEXT = "\n\n🤖 This content was created with AI. #AIGenerated #AIArt"

INSTAGRAM_GRAPH_API_BASE = "https://graph.facebook.com/v18.0"
TWITTER_API_BASE = "https://api.twitter.com/2"
TWITTER_UPLOAD_BASE = "https://upload.twitter.com/1.1"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5.0


# ==============================================================================
# Data Classes
# ==============================================================================


@dataclass
class PublishResult:
    """
    Result of a publishing operation.

    Attributes:
        success: Whether the post was published successfully.
        platform: Target platform name.
        post_id: Platform-assigned post ID (if successful).
        url: Direct URL to the published post (if available).
        error: Error message (if failed).
        timestamp: UTC timestamp of the publish attempt.
        metadata: Additional platform-specific response data.
        retries: Number of retry attempts made.
    """

    success: bool
    platform: str
    post_id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    retries: int = 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "platform": self.platform,
            "post_id": self.post_id,
            "url": self.url,
            "error": self.error,
            "timestamp": self.timestamp,
            "retries": self.retries,
            "metadata": self.metadata,
        }


def _enforce_ai_disclosure(caption: str) -> str:
    """
    Ensure AI disclosure is present in the caption.

    If the caption does not already contain #AIGenerated,
    appends the standard disclosure text.

    Args:
        caption: Original caption text.

    Returns:
        Caption with AI disclosure guaranteed.
    """
    if AI_DISCLOSURE_TAG.lower() in caption.lower():
        return caption
    return caption + AI_DISCLOSURE_TEXT



# ==============================================================================
# Instagram Publisher
# ==============================================================================


class InstagramPublisher:
    """
    Publishes content to Instagram via the Meta Graph API v18.0.

    Supports single image posts, carousel posts (up to 10 images),
    and Reels (video content). Requires a valid Instagram Business
    account access token and page ID.

    Attributes:
        access_token: Meta Graph API long-lived access token.
        instagram_account_id: Instagram Business account ID.
        session: Shared aiohttp session for connection pooling.
    """

    def __init__(self, access_token: str, instagram_account_id: str):
        """
        Initialize Instagram publisher.

        Args:
            access_token: Meta Graph API access token with publish permissions.
            instagram_account_id: Instagram Business/Creator account ID.
        """
        self.access_token = access_token
        self.instagram_account_id = instagram_account_id
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info(f"InstagramPublisher initialized | account_id={instagram_account_id[:8]}...")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Content-Type": "application/json"}
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _create_media_container(
        self, image_url: str, caption: str, is_carousel_item: bool = False
    ) -> Optional[str]:
        """
        Create a media container for an image.

        Args:
            image_url: Public URL of the image to post.
            caption: Post caption (ignored for carousel items).
            is_carousel_item: Whether this is part of a carousel.

        Returns:
            Container ID if successful, None otherwise.
        """
        session = await self._get_session()
        url = f"{INSTAGRAM_GRAPH_API_BASE}/{self.instagram_account_id}/media"

        params = {
            "image_url": image_url,
            "access_token": self.access_token,
        }

        if is_carousel_item:
            params["is_carousel_item"] = "true"
        else:
            params["caption"] = _enforce_ai_disclosure(caption)

        async with session.post(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("id")
            else:
                error_data = await response.text()
                logger.error(f"Instagram container creation failed: {error_data}")
                return None

    async def _publish_container(self, container_id: str) -> Optional[str]:
        """
        Publish a media container.

        Args:
            container_id: The media container ID to publish.

        Returns:
            Published media ID if successful, None otherwise.
        """
        session = await self._get_session()
        url = f"{INSTAGRAM_GRAPH_API_BASE}/{self.instagram_account_id}/media_publish"

        params = {
            "creation_id": container_id,
            "access_token": self.access_token,
        }

        async with session.post(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("id")
            else:
                error_data = await response.text()
                logger.error(f"Instagram publish failed: {error_data}")
                return None

    async def publish_single_image(
        self, image_url: str, caption: str
    ) -> PublishResult:
        """
        Publish a single image post to Instagram.

        Args:
            image_url: Public URL of the image.
            caption: Post caption (AI disclosure will be added).

        Returns:
            PublishResult with success status and post details.
        """
        caption = _enforce_ai_disclosure(caption)
        logger.info(f"Publishing single image to Instagram...")

        for attempt in range(MAX_RETRIES):
            try:
                container_id = await self._create_media_container(image_url, caption)
                if not container_id:
                    continue

                # Wait for container to be ready
                await asyncio.sleep(2)

                post_id = await self._publish_container(container_id)
                if post_id:
                    return PublishResult(
                        success=True,
                        platform="instagram",
                        post_id=post_id,
                        url=f"https://www.instagram.com/p/{post_id}/",
                        retries=attempt,
                    )

            except aiohttp.ClientError as e:
                logger.warning(f"Instagram attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

        return PublishResult(
            success=False,
            platform="instagram",
            error="Max retries exceeded",
            retries=MAX_RETRIES,
        )

    async def publish_carousel(
        self, image_urls: list[str], caption: str
    ) -> PublishResult:
        """
        Publish a carousel post (multiple images) to Instagram.

        Args:
            image_urls: List of public image URLs (2-10 images).
            caption: Carousel caption (AI disclosure will be added).

        Returns:
            PublishResult with success status.
        """
        if len(image_urls) < 2 or len(image_urls) > 10:
            return PublishResult(
                success=False,
                platform="instagram",
                error=f"Carousel requires 2-10 images, got {len(image_urls)}",
            )

        caption = _enforce_ai_disclosure(caption)
        logger.info(f"Publishing carousel ({len(image_urls)} images) to Instagram...")

        try:
            # Create individual media containers
            container_ids = []
            for img_url in image_urls:
                container_id = await self._create_media_container(
                    img_url, "", is_carousel_item=True
                )
                if container_id:
                    container_ids.append(container_id)
                await asyncio.sleep(1)

            if len(container_ids) < 2:
                return PublishResult(
                    success=False,
                    platform="instagram",
                    error="Failed to create enough media containers",
                )

            # Create carousel container
            session = await self._get_session()
            url = f"{INSTAGRAM_GRAPH_API_BASE}/{self.instagram_account_id}/media"
            params = {
                "media_type": "CAROUSEL",
                "children": ",".join(container_ids),
                "caption": caption,
                "access_token": self.access_token,
            }

            async with session.post(url, params=params) as response:
                if response.status != 200:
                    error_data = await response.text()
                    return PublishResult(
                        success=False,
                        platform="instagram",
                        error=f"Carousel container failed: {error_data}",
                    )
                data = await response.json()
                carousel_id = data.get("id")

            await asyncio.sleep(3)

            # Publish carousel
            post_id = await self._publish_container(carousel_id)
            if post_id:
                return PublishResult(
                    success=True,
                    platform="instagram",
                    post_id=post_id,
                    url=f"https://www.instagram.com/p/{post_id}/",
                    metadata={"num_images": len(container_ids)},
                )

        except Exception as e:
            logger.error(f"Carousel publish error: {e}")

        return PublishResult(
            success=False, platform="instagram", error="Carousel publish failed"
        )

    async def publish_reel(
        self, video_url: str, caption: str, cover_url: Optional[str] = None
    ) -> PublishResult:
        """
        Publish a Reel (short video) to Instagram.

        Args:
            video_url: Public URL of the video file.
            caption: Reel caption (AI disclosure will be added).
            cover_url: Optional cover image URL.

        Returns:
            PublishResult with success status.
        """
        caption = _enforce_ai_disclosure(caption)
        logger.info("Publishing Reel to Instagram...")

        try:
            session = await self._get_session()
            url = f"{INSTAGRAM_GRAPH_API_BASE}/{self.instagram_account_id}/media"

            params = {
                "media_type": "REELS",
                "video_url": video_url,
                "caption": caption,
                "access_token": self.access_token,
            }
            if cover_url:
                params["cover_url"] = cover_url

            async with session.post(url, params=params) as response:
                if response.status != 200:
                    error_data = await response.text()
                    return PublishResult(
                        success=False,
                        platform="instagram",
                        error=f"Reel container failed: {error_data}",
                    )
                data = await response.json()
                container_id = data.get("id")

            # Reels need more processing time
            await asyncio.sleep(10)

            # Check status and publish
            post_id = await self._publish_container(container_id)
            if post_id:
                return PublishResult(
                    success=True,
                    platform="instagram",
                    post_id=post_id,
                    url=f"https://www.instagram.com/reel/{post_id}/",
                    metadata={"content_type": "reel"},
                )

        except Exception as e:
            logger.error(f"Reel publish error: {e}")

        return PublishResult(
            success=False, platform="instagram", error="Reel publish failed"
        )



# ==============================================================================
# Twitter/X Publisher
# ==============================================================================


class TwitterPublisher:
    """
    Publishes content to Twitter/X via the API v2.

    Supports single tweets with media attachments and multi-tweet threads.
    Uses OAuth 2.0 Bearer Token for authentication.

    Attributes:
        bearer_token: Twitter API v2 Bearer token.
        api_key: Twitter API key (for media upload via v1.1).
        api_secret: Twitter API secret.
        access_token: OAuth 1.0a access token (for media upload).
        access_token_secret: OAuth 1.0a access token secret.
    """

    def __init__(
        self,
        bearer_token: str,
        api_key: str,
        api_secret: str,
        access_token: str,
        access_token_secret: str,
    ):
        """
        Initialize Twitter publisher.

        Args:
            bearer_token: API v2 Bearer token for tweet creation.
            api_key: Consumer API key.
            api_secret: Consumer API secret.
            access_token: OAuth 1.0a user access token.
            access_token_secret: OAuth 1.0a user access token secret.
        """
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self._session: Optional[aiohttp.ClientSession] = None
        logger.info("TwitterPublisher initialized")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.bearer_token}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _upload_media(self, media_path: str | Path) -> Optional[str]:
        """
        Upload media to Twitter via the v1.1 media upload endpoint.

        Args:
            media_path: Local path to the media file.

        Returns:
            Media ID string if successful, None otherwise.
        """
        import hashlib
        import hmac
        import urllib.parse

        media_path = Path(media_path)
        if not media_path.exists():
            logger.error(f"Media file not found: {media_path}")
            return None

        url = f"{TWITTER_UPLOAD_BASE}/media/upload.json"

        # Build OAuth 1.0a signature for media upload
        timestamp = str(int(time.time()))
        nonce = hashlib.md5(timestamp.encode()).hexdigest()

        oauth_params = {
            "oauth_consumer_key": self.api_key,
            "oauth_nonce": nonce,
            "oauth_signature_method": "HMAC-SHA1",
            "oauth_timestamp": timestamp,
            "oauth_token": self.access_token,
            "oauth_version": "1.0",
        }

        # Create signature base string
        param_string = "&".join(
            f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}"
            for k, v in sorted(oauth_params.items())
        )
        base_string = f"POST&{urllib.parse.quote(url, safe='')}&{urllib.parse.quote(param_string, safe='')}"

        # Sign with HMAC-SHA1
        signing_key = f"{urllib.parse.quote(self.api_secret, safe='')}&{urllib.parse.quote(self.access_token_secret, safe='')}"
        import base64
        signature = base64.b64encode(
            hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
        ).decode()

        oauth_params["oauth_signature"] = signature
        auth_header = "OAuth " + ", ".join(
            f'{k}="{urllib.parse.quote(v, safe="")}"'
            for k, v in oauth_params.items()
        )

        try:
            async with aiohttp.ClientSession() as session:
                with open(media_path, "rb") as f:
                    data = aiohttp.FormData()
                    data.add_field(
                        "media",
                        f,
                        filename=media_path.name,
                        content_type="image/png",
                    )

                    async with session.post(
                        url, data=data, headers={"Authorization": auth_header}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            media_id = result.get("media_id_string")
                            logger.info(f"Media uploaded: {media_id}")
                            return media_id
                        else:
                            error = await response.text()
                            logger.error(f"Media upload failed: {error}")
                            return None

        except Exception as e:
            logger.error(f"Media upload error: {e}")
            return None

    async def publish_tweet(
        self, text: str, media_path: Optional[str | Path] = None
    ) -> PublishResult:
        """
        Publish a single tweet, optionally with a media attachment.

        Args:
            text: Tweet text (AI disclosure will be added if not present).
            media_path: Optional path to media file to attach.

        Returns:
            PublishResult with success status.
        """
        text = _enforce_ai_disclosure(text)
        logger.info("Publishing tweet to Twitter/X...")

        # Truncate to Twitter limit (280 chars)
        if len(text) > 280:
            text = text[:277] + "..."

        payload: dict = {"text": text}

        # Upload media if provided
        if media_path:
            media_id = await self._upload_media(media_path)
            if media_id:
                payload["media"] = {"media_ids": [media_id]}

        session = await self._get_session()

        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    f"{TWITTER_API_BASE}/tweets", json=payload
                ) as response:
                    if response.status in (200, 201):
                        data = await response.json()
                        tweet_id = data.get("data", {}).get("id")
                        return PublishResult(
                            success=True,
                            platform="twitter",
                            post_id=tweet_id,
                            url=f"https://twitter.com/i/status/{tweet_id}",
                            retries=attempt,
                        )
                    elif response.status == 429:
                        # Rate limited
                        retry_after = int(
                            response.headers.get("retry-after", RETRY_DELAY_SECONDS * 10)
                        )
                        logger.warning(f"Twitter rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                    else:
                        error_data = await response.text()
                        logger.error(f"Tweet failed (attempt {attempt + 1}): {error_data}")
                        await asyncio.sleep(RETRY_DELAY_SECONDS)

            except aiohttp.ClientError as e:
                logger.warning(f"Twitter attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

        return PublishResult(
            success=False,
            platform="twitter",
            error="Max retries exceeded",
            retries=MAX_RETRIES,
        )

    async def publish_thread(
        self,
        tweets: list[str],
        media_paths: Optional[list[Optional[str | Path]]] = None,
    ) -> list[PublishResult]:
        """
        Publish a thread of connected tweets.

        Args:
            tweets: List of tweet texts (first tweet gets AI disclosure).
            media_paths: Optional list of media paths (one per tweet, None for no media).

        Returns:
            List of PublishResult for each tweet in the thread.
        """
        if not tweets:
            return []

        # Ensure AI disclosure on first tweet
        tweets[0] = _enforce_ai_disclosure(tweets[0])

        results: list[PublishResult] = []
        previous_tweet_id: Optional[str] = None
        session = await self._get_session()

        for i, tweet_text in enumerate(tweets):
            if len(tweet_text) > 280:
                tweet_text = tweet_text[:277] + "..."

            payload: dict = {"text": tweet_text}

            if previous_tweet_id:
                payload["reply"] = {"in_reply_to_tweet_id": previous_tweet_id}

            # Attach media if available
            if media_paths and i < len(media_paths) and media_paths[i]:
                media_id = await self._upload_media(media_paths[i])
                if media_id:
                    payload["media"] = {"media_ids": [media_id]}

            try:
                async with session.post(
                    f"{TWITTER_API_BASE}/tweets", json=payload
                ) as response:
                    if response.status in (200, 201):
                        data = await response.json()
                        tweet_id = data.get("data", {}).get("id")
                        previous_tweet_id = tweet_id
                        results.append(
                            PublishResult(
                                success=True,
                                platform="twitter",
                                post_id=tweet_id,
                                url=f"https://twitter.com/i/status/{tweet_id}",
                                metadata={"thread_position": i + 1},
                            )
                        )
                    else:
                        error_data = await response.text()
                        results.append(
                            PublishResult(
                                success=False,
                                platform="twitter",
                                error=f"Thread tweet {i + 1} failed: {error_data}",
                                metadata={"thread_position": i + 1},
                            )
                        )
                        break  # Stop thread on failure

            except Exception as e:
                results.append(
                    PublishResult(
                        success=False,
                        platform="twitter",
                        error=str(e),
                        metadata={"thread_position": i + 1},
                    )
                )
                break

            # Rate limit between tweets
            await asyncio.sleep(2)

        logger.info(f"Thread published: {sum(1 for r in results if r.success)}/{len(tweets)} tweets")
        return results



# ==============================================================================
# Publishing Orchestrator
# ==============================================================================


class PublishingOrchestrator:
    """
    Routes content to the appropriate platform publisher.

    Provides a unified interface for publishing content across all supported
    platforms, handling platform selection, content type routing, and
    aggregate result tracking.

    Attributes:
        instagram: Instagram publisher instance (if configured).
        twitter: Twitter publisher instance (if configured).
        results_history: History of all publish attempts.
    """

    def __init__(
        self,
        instagram: Optional[InstagramPublisher] = None,
        twitter: Optional[TwitterPublisher] = None,
    ):
        """
        Initialize the publishing orchestrator.

        Args:
            instagram: Configured Instagram publisher (or None to skip).
            twitter: Configured Twitter publisher (or None to skip).
        """
        self.instagram = instagram
        self.twitter = twitter
        self.results_history: list[PublishResult] = []

        platforms = []
        if instagram:
            platforms.append("instagram")
        if twitter:
            platforms.append("twitter")

        logger.info(f"PublishingOrchestrator initialized | platforms={platforms}")

    async def publish(
        self,
        platform: str,
        content_type: str,
        caption: str,
        media_urls: Optional[list[str]] = None,
        media_paths: Optional[list[str | Path]] = None,
        video_url: Optional[str] = None,
    ) -> PublishResult:
        """
        Publish content to a specific platform.

        Routes to the appropriate publisher based on platform and content type.
        AI disclosure is enforced automatically.

        Args:
            platform: Target platform ("instagram" or "twitter").
            content_type: Type of content ("image", "carousel", "reel", "tweet", "thread").
            caption: Caption/text for the post.
            media_urls: Public URLs for media (Instagram).
            media_paths: Local file paths for media (Twitter).
            video_url: Video URL for reels.

        Returns:
            PublishResult with operation outcome.
        """
        result: PublishResult

        if platform == "instagram":
            result = await self._publish_instagram(content_type, caption, media_urls, video_url)
        elif platform == "twitter":
            result = await self._publish_twitter(content_type, caption, media_paths)
        else:
            result = PublishResult(
                success=False,
                platform=platform,
                error=f"Unsupported platform: {platform}",
            )

        self.results_history.append(result)
        return result

    async def _publish_instagram(
        self,
        content_type: str,
        caption: str,
        media_urls: Optional[list[str]],
        video_url: Optional[str],
    ) -> PublishResult:
        """Route Instagram publishing based on content type."""
        if not self.instagram:
            return PublishResult(
                success=False,
                platform="instagram",
                error="Instagram publisher not configured",
            )

        if content_type == "image" and media_urls:
            return await self.instagram.publish_single_image(media_urls[0], caption)
        elif content_type == "carousel" and media_urls:
            return await self.instagram.publish_carousel(media_urls, caption)
        elif content_type == "reel" and video_url:
            return await self.instagram.publish_reel(video_url, caption)
        else:
            return PublishResult(
                success=False,
                platform="instagram",
                error=f"Invalid content_type/media combination: {content_type}",
            )

    async def _publish_twitter(
        self,
        content_type: str,
        caption: str,
        media_paths: Optional[list[str | Path]],
    ) -> PublishResult:
        """Route Twitter publishing based on content type."""
        if not self.twitter:
            return PublishResult(
                success=False,
                platform="twitter",
                error="Twitter publisher not configured",
            )

        if content_type in ("image", "tweet"):
            media_path = media_paths[0] if media_paths else None
            return await self.twitter.publish_tweet(caption, media_path)
        elif content_type == "thread":
            # Split caption into thread tweets at double newlines
            tweets = [t.strip() for t in caption.split("\n\n") if t.strip()]
            if not tweets:
                tweets = [caption]
            results = await self.twitter.publish_thread(tweets, media_paths)
            # Return the first result as the primary result
            if results:
                return results[0]
            return PublishResult(
                success=False, platform="twitter", error="Empty thread"
            )
        else:
            # Default to single tweet with optional media
            media_path = media_paths[0] if media_paths else None
            return await self.twitter.publish_tweet(caption, media_path)

    async def publish_cross_platform(
        self,
        caption: str,
        media_urls: Optional[list[str]] = None,
        media_paths: Optional[list[str | Path]] = None,
        platforms: Optional[list[str]] = None,
    ) -> dict[str, PublishResult]:
        """
        Publish the same content across multiple platforms simultaneously.

        Args:
            caption: Shared caption (adapted per platform).
            media_urls: Public URLs for Instagram.
            media_paths: Local paths for Twitter.
            platforms: Platforms to publish to (defaults to all configured).

        Returns:
            Dictionary mapping platform names to their PublishResults.
        """
        if platforms is None:
            platforms = []
            if self.instagram:
                platforms.append("instagram")
            if self.twitter:
                platforms.append("twitter")

        tasks = []
        for platform in platforms:
            content_type = "image"
            if media_urls and len(media_urls) > 1:
                content_type = "carousel"

            tasks.append(
                self.publish(
                    platform=platform,
                    content_type=content_type,
                    caption=caption,
                    media_urls=media_urls,
                    media_paths=media_paths,
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_map: dict[str, PublishResult] = {}
        for platform, result in zip(platforms, results):
            if isinstance(result, Exception):
                result_map[platform] = PublishResult(
                    success=False, platform=platform, error=str(result)
                )
            else:
                result_map[platform] = result

        return result_map

    def get_success_rate(self) -> float:
        """Calculate overall publishing success rate."""
        if not self.results_history:
            return 0.0
        return sum(1 for r in self.results_history if r.success) / len(self.results_history)

    def get_summary(self) -> dict:
        """Get publishing statistics summary."""
        total = len(self.results_history)
        successes = sum(1 for r in self.results_history if r.success)

        platform_stats: dict[str, dict] = {}
        for result in self.results_history:
            if result.platform not in platform_stats:
                platform_stats[result.platform] = {"total": 0, "success": 0, "failed": 0}
            platform_stats[result.platform]["total"] += 1
            if result.success:
                platform_stats[result.platform]["success"] += 1
            else:
                platform_stats[result.platform]["failed"] += 1

        return {
            "total_attempts": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": round(successes / max(total, 1), 4),
            "platform_stats": platform_stats,
        }

    async def close(self) -> None:
        """Close all publisher sessions."""
        if self.instagram:
            await self.instagram.close()
        if self.twitter:
            await self.twitter.close()
        logger.info("All publisher sessions closed")


# ==============================================================================
# CLI Interface
# ==============================================================================


async def _async_main():
    """Async main for testing publishing."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Social Media Publisher")
    parser.add_argument("--platform", choices=["instagram", "twitter", "all"], default="all")
    parser.add_argument("--content-type", choices=["image", "carousel", "reel", "tweet"], default="image")
    parser.add_argument("--caption", type=str, required=True)
    parser.add_argument("--media-url", type=str, nargs="*", default=None)
    parser.add_argument("--media-path", type=str, nargs="*", default=None)
    parser.add_argument("--video-url", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print config without publishing")

    args = parser.parse_args()

    if args.dry_run:
        caption = _enforce_ai_disclosure(args.caption)
        print(f"\n{'='*50}")
        print(f"  PUBLISHING DRY RUN")
        print(f"{'='*50}")
        print(f"  Platform:     {args.platform}")
        print(f"  Content type: {args.content_type}")
        print(f"  Caption:      {caption[:100]}...")
        print(f"  Media URLs:   {args.media_url}")
        print(f"  Media paths:  {args.media_path}")
        print(f"  AI Disclosure: {'Present' if AI_DISCLOSURE_TAG.lower() in caption.lower() else 'MISSING'}")
        print(f"{'='*50}\n")
        return

    # Initialize publishers from environment
    instagram = None
    twitter = None

    ig_token = os.environ.get("INSTAGRAM_ACCESS_TOKEN")
    ig_account = os.environ.get("INSTAGRAM_ACCOUNT_ID")
    if ig_token and ig_account:
        instagram = InstagramPublisher(ig_token, ig_account)

    tw_bearer = os.environ.get("TWITTER_BEARER_TOKEN")
    tw_key = os.environ.get("TWITTER_API_KEY")
    tw_secret = os.environ.get("TWITTER_API_SECRET")
    tw_access = os.environ.get("TWITTER_ACCESS_TOKEN")
    tw_access_secret = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")
    if all([tw_bearer, tw_key, tw_secret, tw_access, tw_access_secret]):
        twitter = TwitterPublisher(tw_bearer, tw_key, tw_secret, tw_access, tw_access_secret)

    orchestrator = PublishingOrchestrator(instagram=instagram, twitter=twitter)

    try:
        result = await orchestrator.publish(
            platform=args.platform,
            content_type=args.content_type,
            caption=args.caption,
            media_urls=args.media_url,
            media_paths=[Path(p) for p in args.media_path] if args.media_path else None,
            video_url=args.video_url,
        )
        print(f"\nPublish result: {result.to_dict()}")
    finally:
        await orchestrator.close()


def main():
    """CLI entry point."""
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
