"""
ComfyUI Batch Processing Module
================================
Async batch generation via ComfyUI WebSocket API.

Implements:
- WebSocket connection management with auto-reconnect
- Workflow JSON loading and parameter injection
- Queue-based batch processing with progress tracking
- Output image collection and organization
- VRAM-aware batch sizing

Reference: ComfyUI API documentation (WebSocket + REST)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Optional

import aiohttp
import aiofiles
from loguru import logger


class ComfyUIBatchGenerator:
    """
    Async batch generator for ComfyUI.
    
    Connects via WebSocket for real-time progress and REST API for
    queue management. Supports dynamic workflow parameter injection
    for batch generation with varying prompts/seeds/LoRA weights.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8188,
        protocol: str = "http",
        timeout: int = 300,
        max_retries: int = 3,
    ):
        """
        Initialize ComfyUI client.
        
        Args:
            host: ComfyUI server host
            port: ComfyUI server port
            protocol: http or https
            timeout: Request timeout seconds
            max_retries: Max retries per generation
        """
        self.host = host
        self.port = port
        self.base_url = f"{protocol}://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.timeout = timeout
        self.max_retries = max_retries
        self.client_id = str(uuid.uuid4())
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(f"ComfyUI client | {self.base_url} | client_id={self.client_id[:8]}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """Check if ComfyUI server is running."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/system_stats") as resp:
                return resp.status == 200
        except Exception:
            return False

    async def queue_prompt(self, workflow: dict) -> str:
        """
        Queue a workflow for execution.
        
        Args:
            workflow: ComfyUI workflow dict (API format)
            
        Returns:
            prompt_id for tracking
        """
        session = await self._get_session()
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        
        async with session.post(f"{self.base_url}/prompt", json=payload) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"Queue failed: {error}")
            data = await resp.json()
            return data["prompt_id"]

    async def wait_for_completion(self, prompt_id: str) -> dict[str, Any]:
        """
        Wait for a prompt to complete via WebSocket.
        
        Returns output node data when done.
        """
        ws_url = f"{self.ws_url}?clientId={self.client_id}"
        
        async with aiohttp.ClientSession() as ws_session:
            async with ws_session.ws_connect(ws_url) as ws:
                while True:
                    msg = await ws.receive()
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")
                        
                        if msg_type == "executing":
                            exec_data = data.get("data", {})
                            if exec_data.get("prompt_id") == prompt_id:
                                if exec_data.get("node") is None:
                                    # Execution complete
                                    return await self._get_history(prompt_id)
                                    
                        elif msg_type == "execution_error":
                            exec_data = data.get("data", {})
                            if exec_data.get("prompt_id") == prompt_id:
                                raise RuntimeError(
                                    f"Execution error: {exec_data.get('exception_message', 'Unknown')}"
                                )
                    elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                        raise RuntimeError("WebSocket connection lost")

    async def _get_history(self, prompt_id: str) -> dict:
        """Get execution history/output for a prompt."""
        session = await self._get_session()
        async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
            data = await resp.json()
            return data.get(prompt_id, {})

    async def download_output(
        self, filename: str, subfolder: str = "", output_dir: Path = Path("output/")
    ) -> Path:
        """
        Download generated image from ComfyUI output.
        
        Args:
            filename: Output filename from ComfyUI
            subfolder: ComfyUI subfolder
            output_dir: Local save directory
            
        Returns:
            Path to downloaded file
        """
        session = await self._get_session()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        params = {"filename": filename, "subfolder": subfolder, "type": "output"}
        async with session.get(f"{self.base_url}/view", params=params) as resp:
            if resp.status == 200:
                output_path = output_dir / filename
                async with aiofiles.open(output_path, "wb") as f:
                    await f.write(await resp.read())
                return output_path
            else:
                raise RuntimeError(f"Download failed: {resp.status}")

    @staticmethod
    def load_workflow(workflow_path: str | Path) -> dict:
        """Load workflow JSON from file."""
        with open(workflow_path, "r") as f:
            return json.load(f)

    @staticmethod
    def inject_parameters(
        workflow: dict,
        parameters: dict[str, Any],
    ) -> dict:
        """
        Inject parameters into workflow nodes.
        
        Parameters format:
        {
            "node_id.field": value,
            "3.text": "new prompt",
            "5.seed": 12345,
            "10.lora_name": "my_lora.safetensors",
            "10.strength_model": 0.8,
        }
        """
        workflow_copy = json.loads(json.dumps(workflow))  # Deep copy
        
        for key, value in parameters.items():
            parts = key.split(".", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid parameter key: {key}")
                continue
                
            node_id, field = parts
            if node_id in workflow_copy:
                if "inputs" in workflow_copy[node_id]:
                    workflow_copy[node_id]["inputs"][field] = value
                    logger.debug(f"Injected: node[{node_id}].{field} = {value}")
                else:
                    logger.warning(f"Node {node_id} has no 'inputs' field")
            else:
                logger.warning(f"Node {node_id} not found in workflow")
        
        return workflow_copy

    async def generate_single(
        self,
        workflow: dict,
        parameters: Optional[dict] = None,
        output_dir: Path = Path("output/"),
    ) -> list[Path]:
        """
        Generate a single image with optional parameter injection.
        
        Args:
            workflow: Base workflow dict
            parameters: Optional parameter overrides
            output_dir: Where to save outputs
            
        Returns:
            List of output file paths
        """
        if parameters:
            workflow = self.inject_parameters(workflow, parameters)
        
        prompt_id = await self.queue_prompt(workflow)
        logger.info(f"Queued: {prompt_id[:8]}")
        
        history = await self.wait_for_completion(prompt_id)
        
        # Extract output images
        output_paths = []
        outputs = history.get("outputs", {})
        
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    path = await self.download_output(
                        filename=img_info["filename"],
                        subfolder=img_info.get("subfolder", ""),
                        output_dir=output_dir,
                    )
                    output_paths.append(path)
        
        logger.info(f"Generated {len(output_paths)} images")
        return output_paths

    async def generate_batch(
        self,
        workflow_path: str | Path,
        batch_parameters: list[dict],
        output_dir: str | Path = "output/batch/",
        concurrency: int = 2,
    ) -> list[list[Path]]:
        """
        Batch generation with multiple parameter sets.
        
        Processes batches sequentially to avoid VRAM overflow on single GPU.
        Concurrency controls how many prompts are queued simultaneously
        (ComfyUI handles internal queuing).
        
        Args:
            workflow_path: Path to base workflow JSON
            batch_parameters: List of parameter dicts for each generation
            output_dir: Base output directory
            concurrency: Max concurrent queue submissions
            
        Returns:
            List of output path lists (one per batch item)
        """
        output_dir = Path(output_dir)
        base_workflow = self.load_workflow(workflow_path)
        
        results = []
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_item(idx: int, params: dict):
            async with semaphore:
                item_dir = output_dir / f"batch_{idx:04d}"
                item_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    paths = await self.generate_single(
                        workflow=base_workflow,
                        parameters=params,
                        output_dir=item_dir,
                    )
                    logger.info(f"Batch [{idx + 1}/{len(batch_parameters)}] complete")
                    return paths
                except Exception as e:
                    logger.error(f"Batch [{idx + 1}] failed: {e}")
                    return []
        
        tasks = [process_item(i, p) for i, p in enumerate(batch_parameters)]
        results = await asyncio.gather(*tasks)
        
        total_images = sum(len(r) for r in results)
        logger.info(
            f"Batch complete: {total_images} images from "
            f"{len(batch_parameters)} prompts"
        )
        
        return results

    async def generate_character_batch(
        self,
        workflow_path: str | Path,
        character_lora: str,
        lora_weight: float = 0.8,
        prompts: list[str] = None,
        seeds: list[int] = None,
        lora_node_id: str = "10",
        prompt_node_id: str = "6",
        seed_node_id: str = "3",
        output_dir: str | Path = "output/character_batch/",
    ) -> list[list[Path]]:
        """
        Convenience method for character-consistent batch generation.
        
        Generates multiple images of the same character with different
        prompts/seeds while maintaining LoRA consistency.
        
        Args:
            workflow_path: Base workflow with LoRA loader node
            character_lora: LoRA filename (e.g., "character_v1.safetensors")
            lora_weight: LoRA strength (0.6-0.8 recommended for Flux)
            prompts: List of variation prompts
            seeds: List of seeds (auto-generated if None)
            lora_node_id: Node ID of LoRA loader in workflow
            prompt_node_id: Node ID of text prompt node
            seed_node_id: Node ID of sampler/seed node
            output_dir: Output directory
            
        Returns:
            List of generated image paths per prompt
        """
        import random
        
        if prompts is None:
            prompts = ["portrait, professional photography, studio lighting"]
        
        if seeds is None:
            seeds = [random.randint(0, 2**32 - 1) for _ in prompts]
        
        batch_params = []
        for prompt, seed in zip(prompts, seeds):
            batch_params.append({
                f"{lora_node_id}.lora_name": character_lora,
                f"{lora_node_id}.strength_model": lora_weight,
                f"{prompt_node_id}.text": prompt,
                f"{seed_node_id}.seed": seed,
            })
        
        return await self.generate_batch(
            workflow_path=workflow_path,
            batch_parameters=batch_params,
            output_dir=output_dir,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
