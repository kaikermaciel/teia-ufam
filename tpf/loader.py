from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch
import psutil
import os

# Função para verificar recursos do sistema
def check_system_resources(ram_limit, cpu_cores, gpu_layers):
    # Verificar a quantidade de RAM disponível
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # GB
    if available_ram < ram_limit:
        raise MemoryError(f"RAM insuficiente: {available_ram:.2f} GB disponíveis.")
    
    # Verificar a quantidade de núcleos da CPU
    available_cpu = psutil.cpu_count(logical=False)
    if available_cpu < cpu_cores:
        raise ValueError(f"Núcleos de CPU insuficientes: {available_cpu} disponíveis.")
    
    # GPU (simulação para limitar camadas; seria mais complexo com um gerenciador de GPU)
    if torch.cuda.is_available():
        available_gpu = torch.cuda.device_count()
        if available_gpu < gpu_layers:
            raise ValueError(f"Camadas da GPU insuficientes: {available_gpu} disponíveis.")
    else:
        raise EnvironmentError("Nenhuma GPU disponível.")

# Função para carregar o modelo RAG e o retriever
def load_rag_model(model_name):
    # Carregar o modelo RAG
    model = RagSequenceForGeneration.from_pretrained(model_name)
    tokenizer = RagTokenizer.from_pretrained(model_name)
    
    # Usar um retriever da HuggingFace
    retriever = RagRetriever.from_pretrained(model_name)
    
    return model, tokenizer, retriever

# Função para carregar documentos a partir de arquivos
def load_documents_from_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents
