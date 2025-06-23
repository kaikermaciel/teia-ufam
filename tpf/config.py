import os

# Configurações do modelo
MODEL_CHOICES = [
    "facebook/rag-token-nq",  # Modelo leve de RAG baseado em token
    "facebook/rag-sequence-nq",  # Modelo leve de RAG baseado em sequência
    "bigscience/bloom-560m",  # Modelo Bloom de tamanho médio com suporte RAG
    "facebook/opt-1.3b",  # Modelo OPT 1.3B
    "facebook/opt-2.7b",  # Modelo OPT 2.7B
    "google/gemma-2-9B-it",  # Modelo Google Gemma 2.9B
    "huggingface/llama-7B",  # Llama 7B
    "huggingface/llama-13B",  # Llama 13B (mais pesado)
    "deepseek-ai/DeepSeek-R1-0528",
    "meta-llama/Llama-3.2-1B",
    "deepseek-r1:1.5b",
    "gemma3:4b", 
    "qwen3:4b"
]

# Configurações de hardware
DEFAULT_RAM_LIMIT = 8  # GB de RAM
DEFAULT_CPU_CORES = 2  # Núcleos de CPU
DEFAULT_GPU_LAYERS = 2  # Camadas da GPU

# Configuração do diretório de arquivos
UPLOAD_DIR = "uploaded_files"
