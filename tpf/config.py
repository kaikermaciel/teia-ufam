import os

# Configurações do modelo
MODEL_CHOICES = [
    "gpt2", 
    "bert-base-uncased", 
    "google/gemma-2-9B-it",  # Modelo adicional do Google
    "huggingface/llama-7B",   # Modelo Llama 7B
    "deepseek/deepseek-model",  # Modelo DeepSeek
    "facebook/opt-2.7b",   # Outro modelo do Facebook (OPT)
    "EleutherAI/gpt-neo-2.7B",  # GPT-Neo 2.7B
    "bigscience/bloom-560m",   # Bloom 560M
    "meta/llama-13B"   # Modelo Llama 13B
]

# Configurações de hardware
DEFAULT_RAM_LIMIT = 8  # GB de RAM
DEFAULT_CPU_CORES = 2  # Núcleos de CPU
DEFAULT_GPU_LAYERS = 2  # Camadas da GPU

# Configuração do diretório de arquivos
UPLOAD_DIR = "uploaded_files"
