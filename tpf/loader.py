from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, RagTokenizer, RagRetriever, RagSequenceForGeneration
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


def load_llama_instruct_model(model_name):
    """
    Função para carregar o modelo Llama-3.2-3B-Instruct.
    """
    try:
        # Carregar o modelo e o tokenizer para Llama-3.2-3B-Instruct
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Erro ao carregar o modelo {model_name}: {e}")
        raise


def load_rag_model(model_name):
    """
    Função para carregar o modelo RAG e o retriever.
    """
    if model_name == "facebook/rag-token-nq":
        model = RagSequenceForGeneration.from_pretrained(model_name)
        tokenizer = RagTokenizer.from_pretrained(model_name)
        retriever = RagRetriever.from_pretrained(model_name)
    elif model_name == "facebook/rag-sequence-nq":
        model = RagSequenceForGeneration.from_pretrained(model_name)
        tokenizer = RagTokenizer.from_pretrained(model_name)
        retriever = RagRetriever.from_pretrained(model_name)
    elif model_name == "meta-llama/Llama-3.2-1B":
        # Carregar o modelo Llama-3.2-3B-Instruct
        model, tokenizer = load_llama_instruct_model(model_name)
        retriever = None  # Não é necessário para esse modelo específico
    elif model_name == "facebook/bloom-560m":
        model = RagSequenceForGeneration.from_pretrained("facebook/bloom-560m")
        tokenizer = RagTokenizer.from_pretrained("facebook/bloom-560m")
        retriever = RagRetriever.from_pretrained("facebook/bloom-560m")
    elif model_name == "facebook/opt-1.3b":
        model = RagSequenceForGeneration.from_pretrained("facebook/opt-1.3b")
        tokenizer = RagTokenizer.from_pretrained("facebook/opt-1.3b")
        retriever = RagRetriever.from_pretrained("facebook/opt-1.3b")
    elif model_name == "facebook/opt-2.7b":
        model = RagSequenceForGeneration.from_pretrained("facebook/opt-2.7b")
        tokenizer = RagTokenizer.from_pretrained("facebook/opt-2.7b")
        retriever = RagRetriever.from_pretrained("facebook/opt-2.7b")
    elif model_name == "huggingface/llama-7B":
        model = RagSequenceForGeneration.from_pretrained("huggingface/llama-7B")
        tokenizer = RagTokenizer.from_pretrained("huggingface/llama-7B")
        retriever = RagRetriever.from_pretrained("huggingface/llama-7B")
    elif model_name == "huggingface/llama-13B":
        model = RagSequenceForGeneration.from_pretrained("huggingface/llama-13B")
        tokenizer = RagTokenizer.from_pretrained("huggingface/llama-13B")
        retriever = RagRetriever.from_pretrained("huggingface/llama-13B")
    elif model_name == "deepseek-ai/DeepSeek-R1-0528":
        model = RagSequenceForGeneration.from_pretrained("deepseek-ai/DeepSeek-R1-0528")
        tokenizer = RagTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-0528")
        retriever = RagRetriever.from_pretrained("deepseek-ai/DeepSeek-R1-0528")
    else:
        # Caso o modelo não seja compatível com RAG diretamente, use um fallback
        raise ValueError(f"O modelo {model_name} não é compatível com RAG.")
    
    return model, tokenizer, retriever

def load_gemma2_model(model_name):
    """
    Função para carregar o modelo Gemma2 com question_encoder e generator.
    """
    try:
        # Carregar a configuração do modelo
        config = AutoConfig.from_pretrained(model_name)
        
        # Especificar as configurações adicionais que o Gemma2 precisa
        config.question_encoder = "path_to_question_encoder_config"  # Insira o caminho para a configuração do question_encoder
        config.generator = "path_to_generator_config"  # Insira o caminho para a configuração do generator
        
        # Carregar o modelo e o tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer

    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        raise

def load_deepseek_model(model_name):
    """
    Função para carregar o modelo Gemma2 com question_encoder e generator.
    """
    try:
        # Carregar a configuração do modelo
        config = AutoConfig.from_pretrained(model_name)
        
        # Especificar as configurações adicionais que o Gemma2 precisa
        config.question_encoder = "path_to_question_encoder_config"  # Insira o caminho para a configuração do question_encoder
        config.generator = "path_to_generator_config"  # Insira o caminho para a configuração do generator
        
        # Carregar o modelo e o tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer

    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        raise

# Função para carregar documentos a partir de arquivos
def load_documents_from_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents
