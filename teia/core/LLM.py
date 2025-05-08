from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate


embeddings = OllamaEmbeddings(model="llama3")

llm = OllamaLLM

retriever_prompt = PromptTemplate.from_template(
    """Generate questions form based on the document uploaded.

New Question: {input}"""
)

qa_prompt = PromptTemplate.from_template(
    """You're a useful assistant which answer based on the provided documents.
        

Context:
{context}

Question: {input}"""
)