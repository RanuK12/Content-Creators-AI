"""
Kling AI API Wrapper - Python
Async wrapper con polling, reintentos y descarga automática.
"""

import os
import time
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import json
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

@dataclass
class KlingConfig:
    access_key: str
    secret_key: str
    base_url: str = "https://api.klingai.com"
    timeout: int = 300
    max_retries: int = 3
    output_dir: str = "./data/output"

class KlingAPIError(Exception):
    pass

class KlingAPI:
    def __init__(self, config: Optional[KlingConfig] = None):
        if config is None:
            config = KlingConfig(
                access_key=os.getenv("KLING_ACCESS_KEY"),
                secret_key=os.getenv("KLING_SECRET_KEY"),
                output_dir=os.getenv("OUTPUT_BASE_DIR", "./data/output")
            )
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.access_key}:{self.config.secret_key}",
                "Content-Type": "application/json"
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.config.base_url}{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited. Esperando {retry_after}s...")
                        await asyncio.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    return await response.json()

            except aiohttp.ClientError as e:
                wait_time = (2 ** attempt) + (attempt * 0.1)
                logger.error(f"Error en intento {attempt + 1}: {e}. Reintentando en {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise KlingAPIError(f"Max retries ({self.config.max_retries}) excedidos para {endpoint}")

    async def text_to_video(
        self,
        prompt: str,
        model: str = "kling-v2-master",
        negative_prompt: str = "",
        duration: str = "5",
        aspect_ratio: str = "9:16",
        mode: str = "std",
        **kwargs
    ) -> Dict[str, Any]:
        """Genera video desde texto."""
        payload = {
            "prompt": prompt,
            "model_name": model,
            "negative_prompt": negative_prompt,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
            **kwargs
        }

        logger.info(f"Iniciando text-to-video: {prompt[:50]}...")
        result = await self._request("POST", "/v2/videos/text2video", json=payload)
        return await self._poll_task(result["data"]["task_id"], "video")

    async def image_to_video(
        self,
        image_path: str,
        prompt: str = "",
        model: str = "kling-v2-master",
        duration: str = "5",
        mode: str = "std",
        **kwargs
    ) -> Dict[str, Any]:
        """Genera video desde imagen."""
        # Convertir imagen a base64
        import base64
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "image": image_b64,
            "prompt": prompt,
            "model_name": model,
            "duration": duration,
            "mode": mode,
            **kwargs
        }

        logger.info(f"Iniciando image-to-video desde: {image_path}")
        result = await self._request("POST", "/v2/videos/image2video", json=payload)
        return await self._poll_task(result["data"]["task_id"], "video")

    async def _poll_task(self, task_id: str, task_type: str, poll_interval: int = 10) -> Dict[str, Any]:
        """Polling de estado de tarea con descarga automática."""
        endpoint = f"/v2/videos/{task_id}" if task_type == "video" else f"/v2/images/{task_id}"

        while True:
            result = await self._request("GET", endpoint)
            status = result["data"]["task_status"]

            if status == "succeed":
                logger.success(f"Tarea {task_id} completada!")
                return await self._download_result(result["data"], task_type)
            elif status == "failed":
                raise KlingAPIError(f"Tarea {task_id} falló: {result['data'].get('task_status_msg', 'Unknown error')}")
            else:
                logger.info(f"Tarea {task_id} status: {status}. Esperando {poll_interval}s...")
                await asyncio.sleep(poll_interval)

    async def _download_result(self, task_data: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Descarga el resultado y lo guarda localmente."""
        output_dir = Path(self.config.output_dir) / task_type
        output_dir.mkdir(parents=True, exist_ok=True)

        if task_type == "video":
            video_url = task_data["task_result"]["videos"][0]["url"]
            filename = f"kling_{task_data['task_id']}.mp4"
        else:
            video_url = task_data["task_result"]["images"][0]["url"]
            filename = f"kling_{task_data['task_id']}.png"

        filepath = output_dir / filename

        async with self.session.get(video_url) as response:
            content = await response.read()
            with open(filepath, "wb") as f:
                f.write(content)

        logger.success(f"Descargado: {filepath}")
        return {
            "task_id": task_data["task_id"],
            "filepath": str(filepath),
            "url": video_url,
            "type": task_type
        }

# Ejemplo de uso
async def main():
    async with KlingAPI() as api:
        # Text to Video
        result = await api.text_to_video(
            prompt="A beautiful young woman walking in a sunny park, cinematic lighting, slow motion",
            model="kling-v2-master",
            duration="5",
            aspect_ratio="9:16",
            mode="pro"
        )
        print(f"Video generado: {result['filepath']}")

        # Image to Video
        # result = await api.image_to_video(
        #     image_path="./data/output/character_base.png",
        #     prompt="Gentle smile, subtle head movement",
        #     duration="5"
        # )

if __name__ == "__main__":
    asyncio.run(main())
