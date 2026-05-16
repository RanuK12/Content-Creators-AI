#!/bin/bash
# Setup script para Virtual Content Pipeline

echo "🚀 Iniciando setup del pipeline..."

# Crear directorios
mkdir -p data/{raw_images,processed_datasets,training_data,output}
mkdir -p logs
mkdir -p workflows/{workflow_a_character_consistency,workflow_b_outfit_variation,workflow_c_motion_reel}

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install -r requirements.txt

# Descargar modelos base (placeholder - manual o script separado)
echo "⚠️  Descarga manual requerida:"
echo "   - Flux.1 [dev] → ./config/models/flux1-dev.safetensors"
echo "   - RealVisXL V5.0 → ./config/models/realvisxlV50.safetensors"
echo "   - PuLID model → ./config/models/pulid_flux.safetensors"

# Copiar .env
cp .env.example .env
echo "📝 Copiado .env.example → .env (edita con tus API keys)"

# Verificar ComfyUI
echo "🔍 Verificando ComfyUI..."
if [ -d "../ComfyUI" ] || [ -d "./ComfyUI" ]; then
    echo "✅ ComfyUI detectado"
else
    echo "⚠️  ComfyUI no detectado. Clona en ../ComfyUI o ajusta config.yaml"
fi

echo "✅ Setup completo. Edita .env y config.yaml antes de ejecutar."
