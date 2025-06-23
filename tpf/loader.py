import ollama  # Importando a biblioteca Ollama
from huggingface_hub import login  # Importando para login no Hugging Face
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch
import psutil
import os
import streamlit as st

# Função para fazer login no Hugging Face usando um token de acesso
def login_to_huggingface():
    try:
        # Faça o login no Hugging Face com o token
        login(token="seu_token_aqui")  # Substitua pelo seu token de acesso
    except Exception as e:
        print(f"Erro ao fazer login no Hugging Face: {e}")
        raise

# Função para verificar recursos do sistema
def check_system_resources(ram_limit, cpu_cores, gpu_layers):
    # Verificar a quantidade de RAM disponível
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # GB
    if available_ram < ram_limit:
        st.warning(f"RAM insuficiente: {available_ram:.2f} GB disponíveis. O modelo pode não funcionar corretamente.")
    
    # Verificar a quantidade de núcleos da CPU
    available_cpu = psutil.cpu_count(logical=False)
    if available_cpu < cpu_cores:
        st.warning(f"Núcleos de CPU insuficientes: {available_cpu} disponíveis.")
    
    # Verificar GPU
    if torch.cuda.is_available():
        available_gpu = torch.cuda.device_count()
        if available_gpu < gpu_layers:
            st.warning(f"Camadas da GPU insuficientes: {available_gpu} disponíveis.")
    else:
        st.warning("Nenhuma GPU disponível, utilizando CPU.")

# Função para carregar o modelo DeepSeek-R1-1.5B usando a API Ollama
def load_ollama_model(model_name):
    """
    Função para carregar o modelo do Ollama, como DeepSeek-R1-1.5B.
    """
    try:
        # Usando a API Ollama para carregar o modelo
        model = ollama.load(model_name)  # Aqui é onde a API Ollama carrega o modelo.
        return model  # O modelo carregado é retornado
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
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        retriever = None  # Não é necessário para esse modelo específico
    elif model_name == "deepseek-ai/DeepSeek-R1-1.5B":  # Carregamento do modelo Ollama
        model = load_ollama_model(model_name)  # Não precisa de retriever nem tokenizer
        tokenizer = None  # Ollama cuida do tokenizer
        retriever = None  # Não usa retriever para Ollama
    else:
        model = RagSequenceForGeneration.from_pretrained(model_name)
        tokenizer = RagTokenizer.from_pretrained(model_name)
        retriever = RagRetriever.from_pretrained(model_name)
    
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
    Função para carregar o modelo DeepSeek.
    """
    try:
        # Carregar a configuração do modelo
        config = AutoConfig.from_pretrained(model_name)
        
        # Especificar as configurações adicionais que o modelo precisa
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
