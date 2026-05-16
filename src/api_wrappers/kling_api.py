"""
Kling AI API Wrapper
====================
Async wrapper for Kling AI v2.0 Image-to-Video generation.

Implements:
- JWT authentication (access_key + secret_key)
- Async task submission with configurable parameters
- Polling-based status checking with exponential backoff
- Automatic download and retry logic
- Camera movement control (dolly, pan, zoom)

Reference: Kling AI API Documentation v2.0
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from loguru import logger


class TaskStatus(str, Enum):
    """Kling AI task statuses."""
    SUBMITTED = "submitted"
    PROCESSING = "processing"
    SUCCEED = "succeed"
    FAILED = "failed"


class CameraMovement(str, Enum):
    """Available camera movements for I2V generation."""
    NONE = "none"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    PAN_UP = "pan_up"
    PAN_DOWN = "pan_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


@dataclass
class VideoTask:
    """Container for a video generation task."""
    task_id: str
    status: TaskStatus
    video_url: Optional[str] = None
    duration: float = 5.0
    created_at: float = 0.0
    completed_at: float = 0.0
    error_message: Optional[str] = None
    local_path: Optional[str] = None


class KlingAPIClient:
    """
    Async client for Kling AI Image-to-Video API v2.0.
    
    Handles authentication, task lifecycle, and file downloads.
    Designed for batch processing with rate limiting awareness.
    """

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        base_url: str = "https://api.klingai.com/v2",
        polling_interval: int = 10,
        max_polling_attempts: int = 60,
        timeout: int = 600,
    ):
        """
        Initialize Kling API client.
        
        Args:
            access_key: API access key
            secret_key: API secret key
            base_url: API base URL
            polling_interval: Seconds between status checks
            max_polling_attempts: Maximum number of polling attempts
            timeout: Request timeout in seconds
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.polling_interval = polling_interval
        self.max_polling_attempts = max_polling_attempts
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(
            f"KlingAPIClient initialized | base_url={base_url} | "
            f"polling={polling_interval}s | max_attempts={max_polling_attempts}"
        )

    def _generate_jwt_token(self) -> str:
        """
        Generate JWT token for API authentication.
        
        Kling API uses HS256 JWT with access_key as issuer.
        Token expires after 30 minutes.
        """
        now = int(time.time())
        
        header = {
            "alg": "HS256",
            "typ": "JWT"
        }
        payload = {
            "iss": self.access_key,
            "exp": now + 1800,  # 30 min expiry
            "nbf": now - 5,
            "iat": now,
        }
        
        # Encode header and payload
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).rstrip(b"=").decode()
        
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).rstrip(b"=").decode()
        
        # Sign
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
        
        return f"{header_b64}.{payload_b64}.{signature_b64}"

    @property
    def _headers(self) -> dict[str, str]:
        """Generate authenticated headers."""
        return {
            "Authorization": f"Bearer {self._generate_jwt_token()}",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def submit_image_to_video(
        self,
        image_path: str | Path,
        prompt: str = "",
        duration: float = 5.0,
        aspect_ratio: str = "9:16",
        motion_strength: float = 0.7,
        camera_movement: CameraMovement = CameraMovement.NONE,
        camera_strength: float = 0.5,
        negative_prompt: str = "",
        seed: Optional[int] = None,
    ) -> VideoTask:
        """
        Submit an image-to-video generation task.
        
        Args:
            image_path: Path to source image
            prompt: Motion/action prompt
            duration: Video duration in seconds (5 or 10)
            aspect_ratio: Output aspect ratio (9:16, 16:9, 1:1)
            motion_strength: Motion intensity [0.0 - 1.0]
            camera_movement: Type of camera movement
            camera_strength: Camera movement intensity [0.0 - 1.0]
            negative_prompt: Negative prompt for quality control
            seed: Random seed for reproducibility
            
        Returns:
            VideoTask with task_id and initial status
        """
        session = await self._get_session()
        image_path = Path(image_path)
        
        # Read and encode image
        async with aiofiles.open(image_path, "rb") as f:
            image_data = await f.read()
        image_b64 = base64.b64encode(image_data).decode()
        
        # Build request payload
        payload = {
            "image": image_b64,
            "prompt": prompt,
            "duration": str(duration),
            "aspect_ratio": aspect_ratio,
            "cfg_scale": motion_strength,
        }
        
        if camera_movement != CameraMovement.NONE:
            payload["camera_control"] = {
                "type": camera_movement.value,
                "config": {"strength": camera_strength},
            }
        
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
            
        if seed is not None:
            payload["seed"] = seed

        # Submit task
        url = f"{self.base_url}/images/generations/videos"
        
        try:
            async with session.post(url, headers=self._headers, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Kling API error {resp.status}: {error_text}")
                    return VideoTask(
                        task_id="",
                        status=TaskStatus.FAILED,
                        error_message=f"HTTP {resp.status}: {error_text}",
                    )
                
                data = await resp.json()
                task_id = data.get("data", {}).get("task_id", "")
                
                logger.info(f"Task submitted: {task_id} | image={image_path.name}")
                
                return VideoTask(
                    task_id=task_id,
                    status=TaskStatus.SUBMITTED,
                    duration=duration,
                    created_at=time.time(),
                )
                
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            return VideoTask(
                task_id="",
                status=TaskStatus.FAILED,
                error_message=str(e),
            )

    async def check_task_status(self, task_id: str) -> VideoTask:
        """
        Check the status of a video generation task.
        
        Args:
            task_id: The task ID to check
            
        Returns:
            Updated VideoTask with current status
        """
        session = await self._get_session()
        url = f"{self.base_url}/images/generations/videos/{task_id}"
        
        try:
            async with session.get(url, headers=self._headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return VideoTask(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        error_message=f"HTTP {resp.status}: {error_text}",
                    )
                
                data = await resp.json()
                task_data = data.get("data", {})
                status_str = task_data.get("task_status", "processing")
                video_url = None
                
                if status_str == "succeed":
                    videos = task_data.get("task_result", {}).get("videos", [])
                    if videos:
                        video_url = videos[0].get("url")
                
                return VideoTask(
                    task_id=task_id,
                    status=TaskStatus(status_str),
                    video_url=video_url,
                )
                
        except Exception as e:
            logger.error(f"Status check failed for {task_id}: {e}")
            return VideoTask(
                task_id=task_id,
                status=TaskStatus.PROCESSING,
                error_message=str(e),
            )

    async def wait_for_completion(
        self, task_id: str, callback=None
    ) -> VideoTask:
        """
        Poll until task completes or fails.
        
        Args:
            task_id: Task to monitor
            callback: Optional async callback(VideoTask) on each poll
            
        Returns:
            Final VideoTask with result
        """
        for attempt in range(self.max_polling_attempts):
            await asyncio.sleep(self.polling_interval)
            
            task = await self.check_task_status(task_id)
            
            if callback:
                await callback(task)
            
            if task.status == TaskStatus.SUCCEED:
                task.completed_at = time.time()
                logger.info(f"Task {task_id} completed successfully")
                return task
            elif task.status == TaskStatus.FAILED:
                logger.error(f"Task {task_id} failed: {task.error_message}")
                return task
            
            logger.debug(
                f"Task {task_id} polling [{attempt + 1}/{self.max_polling_attempts}]"
            )
        
        # Timeout
        logger.warning(f"Task {task_id} timed out after {self.max_polling_attempts} attempts")
        return VideoTask(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error_message="Polling timeout exceeded",
        )

    async def download_video(
        self, video_url: str, output_path: str | Path
    ) -> Path:
        """
        Download completed video to local filesystem.
        
        Args:
            video_url: URL of the generated video
            output_path: Local path to save the video
            
        Returns:
            Path to downloaded file
        """
        session = await self._get_session()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with session.get(video_url) as resp:
            if resp.status == 200:
                async with aiofiles.open(output_path, "wb") as f:
                    await f.write(await resp.read())
                logger.info(f"Video downloaded: {output_path}")
            else:
                logger.error(f"Download failed: HTTP {resp.status}")
                raise RuntimeError(f"Download failed: {resp.status}")
        
        return output_path

    async def generate_video(
        self,
        image_path: str | Path,
        output_path: str | Path,
        prompt: str = "",
        duration: float = 5.0,
        aspect_ratio: str = "9:16",
        motion_strength: float = 0.7,
        camera_movement: CameraMovement = CameraMovement.NONE,
        camera_strength: float = 0.5,
        max_retries: int = 3,
    ) -> Optional[VideoTask]:
        """
        Full pipeline: submit → poll → download.
        
        Convenience method that handles the complete lifecycle with retries.
        
        Args:
            image_path: Source image
            output_path: Where to save the video
            prompt: Motion prompt
            duration: Video duration
            aspect_ratio: Output ratio
            motion_strength: Motion intensity
            camera_movement: Camera type
            camera_strength: Camera intensity
            max_retries: Number of retry attempts on failure
            
        Returns:
            VideoTask with local_path set, or None on failure
        """
        for attempt in range(max_retries):
            logger.info(
                f"Video generation attempt {attempt + 1}/{max_retries} | "
                f"image={Path(image_path).name}"
            )
            
            # Submit
            task = await self.submit_image_to_video(
                image_path=image_path,
                prompt=prompt,
                duration=duration,
                aspect_ratio=aspect_ratio,
                motion_strength=motion_strength,
                camera_movement=camera_movement,
                camera_strength=camera_strength,
            )
            
            if task.status == TaskStatus.FAILED:
                logger.warning(f"Submit failed, retry {attempt + 1}/{max_retries}")
                await asyncio.sleep(5 * (attempt + 1))
                continue
            
            # Wait
            task = await self.wait_for_completion(task.task_id)
            
            if task.status == TaskStatus.SUCCEED and task.video_url:
                # Download
                path = await self.download_video(task.video_url, output_path)
                task.local_path = str(path)
                return task
            
            logger.warning(f"Generation failed, retry {attempt + 1}/{max_retries}")
            await asyncio.sleep(10 * (attempt + 1))
        
        logger.error(f"All {max_retries} attempts failed for {image_path}")
        return None

    async def batch_generate(
        self,
        tasks: list[dict],
        output_dir: str | Path,
        concurrency: int = 3,
    ) -> list[VideoTask]:
        """
        Generate multiple videos with controlled concurrency.
        
        Args:
            tasks: List of dicts with keys: image_path, prompt, duration, etc.
            output_dir: Base directory for output videos
            concurrency: Max simultaneous tasks (API rate limit aware)
            
        Returns:
            List of completed VideoTask objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        semaphore = asyncio.Semaphore(concurrency)
        results = []
        
        async def process_task(idx: int, task_config: dict):
            async with semaphore:
                image_path = task_config["image_path"]
                output_path = output_dir / f"video_{idx:04d}.mp4"
                
                result = await self.generate_video(
                    image_path=image_path,
                    output_path=output_path,
                    prompt=task_config.get("prompt", ""),
                    duration=task_config.get("duration", 5.0),
                    motion_strength=task_config.get("motion_strength", 0.7),
                    camera_movement=CameraMovement(
                        task_config.get("camera_movement", "none")
                    ),
                )
                return result
        
        # Launch all tasks
        coros = [process_task(i, t) for i, t in enumerate(tasks)]
        completed = await asyncio.gather(*coros, return_exceptions=True)
        
        for result in completed:
            if isinstance(result, VideoTask):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Task exception: {result}")
        
        logger.info(
            f"Batch complete: {len(results)}/{len(tasks)} videos generated"
        )
        return results

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
