import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import PyPDF2
import io
import os
import time

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="RAG com LLM",
    page_icon="🤖",
    layout="wide"
)

# --- Cache do Modelo e Tokenizer ---
# O cache é refeito se o nome do modelo ou qualquer configuração de recurso for alterado.
@st.cache_resource
def load_model(model_name, device_choice, max_vram_gb, cpu_threads):
    """
    Carrega o modelo de IA e o tokenizer do Hugging Face com controle de recursos.
    Usa quantização de 4 bits para economizar VRAM na GPU.
    """
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        st.info(f"PyTorch configurado para usar no máximo {torch.get_num_threads()} threads de CPU.")

    quantization_config = None
    max_memory_map = None
    torch_dtype = torch.float32 # Padrão para CPU

    if device_choice == 'Auto (GPU se disponível)' and torch.cuda.is_available():
        device = "cuda"
        device_map = "auto"
        st.info(f"Carregando o modelo ({model_name}) em: GPU com quantização de 4 bits.")

        # Configuração de Quantização para 4 bits para economizar VRAM.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        if max_vram_gb > 0:
            max_memory_map = {0: f"{max_vram_gb}GiB"}
            st.info(f"Tentando limitar a VRAM da GPU a {max_vram_gb}GiB.")
        
    else:
        device = "cpu"
        device_map = None 
        st.info(f"Carregando o modelo ({model_name}) em: CPU. Isso pode ser lento...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            quantization_config=quantization_config,
            max_memory=max_memory_map,
            trust_remote_code=True # Necessário para alguns modelos como o Phi-3
        )
        
        if device_map is None:
            model.to(device)

        st.success(f"Modelo '{os.path.basename(model_name)}' carregado com sucesso no dispositivo: {device.upper()}!")
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.error("Isso pode ser causado por falta de memória (RAM ou VRAM). Tente um modelo menor ou force o uso de CPU.")
        st.stop()


def extract_text_from_file(uploaded_file):
    """
    Extrai o texto de um arquivo carregado (.txt, .md, .pdf).
    """
    if uploaded_file is None:
        return ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            return text
        else:
            return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return ""

def generate_response(tokenizer, model, device, context, query, max_new_tokens):
    """
    Gera uma resposta usando o modelo de IA.
    """
    if not query:
        st.warning("Por favor, insira uma pergunta.")
        return ""

    messages = [
        {"role": "user", "content": f'Use o seguinte contexto para responder à pergunta. Se a resposta não estiver no contexto, diga "Não encontrei a resposta no documento."\n\n--- CONTEXTO ---\n{context}\n\n--- PERGUNTA ---\n{query}'}
    ]
    # O tokenizer.apply_chat_template lida com os diferentes formatos de prompt para cada modelo.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id
        )
    response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response_text

# --- Interface do Usuário (UI) ---
st.title("📄🤖 Chat com Documentos (RAG)")
st.markdown("""
Use esta interface para fazer perguntas a um documento, com controle total sobre o modelo de IA e os recursos de hardware.
**A alteração de qualquer configuração reiniciará o modelo.**
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configurações de IA e Recursos")
    
    # --- NOVO: Seletor de Modelo ---
    model_options = {
        "Gemma-2 9B (Poderoso)": "google/gemma-2-9b-it",
        "Gemma 2B (Leve)": "google/gemma-2b-it",
        "Phi-3 Mini (Rápido)": "microsoft/Phi-3-mini-4k-instruct"
    }
    selected_model_key = st.selectbox(
        "Escolha o modelo de IA",
        options=list(model_options.keys()),
        index=0,
        help="Modelos maiores são mais capazes, mas consomem mais recursos."
    )
    model_id = model_options[selected_model_key]

    device_option = st.selectbox(
        "Dispositivo de processamento",
        ('Auto (GPU se disponível)', 'Forçar CPU'),
        help="'Auto' usará a GPU com quantização de 4 bits. 'Forçar CPU' usará o processador."
    )

    max_vram_option = 0
    if device_option == 'Auto (GPU se disponível)':
        st.info("O modo GPU usará quantização de 4 bits para reduzir o uso de VRAM.")
        max_vram_option = st.number_input(
            "Limite de VRAM da GPU (GB)", 0, 128, 0, 4,
            help="Defina o máximo de VRAM que o modelo pode usar. 0 significa sem limite. O que não couber será alocado na RAM."
        )

    available_cpus = os.cpu_count() or 4 
    cpu_threads_option = st.number_input(
        "Limite de Threads da CPU", 1, available_cpus, available_cpus, 1,
        help=f"Defina o número máximo de núcleos de CPU. Sua máquina tem {available_cpus} núcleos."
    )

    # Carrega o modelo com base em todas as opções selecionadas
    tokenizer, model, device = load_model(model_id, device_option, max_vram_option, cpu_threads_option)
    
    st.divider()

    uploaded_file = st.file_uploader("Faça o upload do seu documento", type=['txt', 'md', 'pdf'])
    
    token_options = [30, 50, 100, 500, 1000, 10000]
    max_tokens = st.selectbox("Limite Máximo de Tokens para Resposta", options=token_options, index=2)
    
    query = st.text_area("Faça sua pergunta aqui:", height=150, placeholder="Ex: Qual é o principal objetivo do documento?")
    
    submit_button = st.button("Gerar Resposta", type="primary", use_container_width=True)

context = ""
if uploaded_file:
    with st.spinner("Processando o arquivo..."):
        context = extract_text_from_file(uploaded_file)
        with col2:
             with st.expander("Conteúdo do Documento (Primeiros 1000 caracteres)"):
                st.write(context[:1000] + "...")

with col2:
    st.header("Resposta do Modelo")
    response_placeholder = st.empty()
    time_placeholder = st.empty() 

    if submit_button:
        if not context:
            st.warning("Por favor, faça o upload de um arquivo.")
        elif not query:
            st.warning("Por favor, digite sua pergunta.")
        else:
            with st.spinner(f"🤖 {os.path.basename(model.name_or_path)} está pensando no dispositivo {device.upper()}..."):
                start_time = time.time()
                response = generate_response(tokenizer, model, device, context, query, max_tokens)
                end_time = time.time()
                duration = end_time - start_time
                
                response_placeholder.markdown(response)
                time_placeholder.success(f"Resposta gerada em {duration:.2f} segundos.")
    else:
        response_placeholder.info("A resposta do modelo aparecerá aqui.")
