"""
ComfyUI Batch Generator
Script para generación batch vía API de ComfyUI.
"""

import json
import uuid
import websocket
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger
import requests

@dataclass
class ComfyUIConfig:
    host: str = "127.0.0.1"
    port: int = 8188
    output_dir: str = "./data/output"

    @property
    def api_url(self):
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self):
        return f"ws://{self.host}:{self.port}/ws"

class ComfyUIBatchGenerator:
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.config = config or ComfyUIConfig()
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Encola un workflow en ComfyUI."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.config.api_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Obtiene el historial de una generación."""
        with urllib.request.urlopen(
            f"{self.config.api_url}/history/{prompt_id}"
        ) as response:
            return json.loads(response.read())

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Descarga una imagen generada."""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url = f"{self.config.api_url}/view?{urllib.parse.urlencode(data)}"

        with urllib.request.urlopen(url) as response:
            return response.read()

    def load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """Carga un workflow JSON exportado de ComfyUI."""
        with open(workflow_path, "r") as f:
            return json.load(f)

    def update_workflow_params(
        self,
        workflow: Dict[str, Any],
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Actualiza parámetros dinámicos en un workflow."""
        workflow = workflow.copy()

        for node_id, node in workflow.items():
            if node["class_type"] == "KSampler":
                if seed is not None:
                    node["inputs"]["seed"] = seed
                if batch_size is not None:
                    node["inputs"]["batch_size"] = batch_size

            elif node["class_type"] == "CLIPTextEncode":
                if prompt is not None and node["inputs"].get("text", "").startswith("positive"):
                    node["inputs"]["text"] = prompt
                if negative_prompt is not None and node["inputs"].get("text", "").startswith("negative"):
                    node["inputs"]["text"] = negative_prompt

            elif node["class_type"] == "EmptyLatentImage":
                if width is not None:
                    node["inputs"]["width"] = width
                if height is not None:
                    node["inputs"]["height"] = height

        return workflow

    def generate_batch(
        self,
        workflow_path: str,
        prompts: List[str],
        seeds: Optional[List[int]] = None,
        output_subfolder: str = "batch"
    ) -> List[str]:
        """Genera un batch de imágenes con diferentes prompts."""
        workflow = self.load_workflow(workflow_path)
        output_files = []

        for i, prompt in enumerate(prompts):
            seed = seeds[i] if seeds and i < len(seeds) else -1

            updated_workflow = self.update_workflow_params(
                workflow,
                prompt=prompt,
                seed=seed
            )

            logger.info(f"Generando {i+1}/{len(prompts)}: {prompt[:50]}...")
            result = self.queue_prompt(updated_workflow)
            prompt_id = result["prompt_id"]

            # Esperar resultado (polling simple)
            import time
            while True:
                history = self.get_history(prompt_id)
                if prompt_id in history:
                    outputs = history[prompt_id]["outputs"]
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            for image in node_output["images"]:
                                filename = image["filename"]
                                image_data = self.get_image(filename, image.get("subfolder", ""))

                                # Guardar localmente
                                output_path = Path(self.config.output_dir) / output_subfolder
                                output_path.mkdir(parents=True, exist_ok=True)
                                filepath = output_path / filename

                                with open(filepath, "wb") as f:
                                    f.write(image_data)

                                output_files.append(str(filepath))
                                logger.success(f"Guardado: {filepath}")
                    break
                time.sleep(1)

        return output_files

# Ejemplo de uso
def main():
    generator = ComfyUIBatchGenerator()

    prompts = [
        "beautiful young woman, portrait, soft lighting, 8k",
        "beautiful young woman, casual outfit, park background, golden hour",
        "beautiful young woman, professional headshot, studio lighting",
        "beautiful young woman, outdoor cafe, natural light, candid",
    ]

    files = generator.generate_batch(
        workflow_path="./workflows/workflow_a_character_consistency/workflow_api.json",
        prompts=prompts,
        output_subfolder="nova_batch_001"
    )

    print(f"Generadas {len(files)} imágenes")

if __name__ == "__main__":
    main()
