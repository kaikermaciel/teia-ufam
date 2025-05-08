from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .LLM import llm, retriever_prompt, qa_prompt

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente útil de documentos."),
    MessagesPlaceholder("chat_history"),
    ("user", "{input}")
])

def criar_chain(retriever):
    retriever_hist = create_history_aware_retriever(
        retriever=retriever,
        llm=llm,
        prompt=retriever_prompt
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt
    )

    return create_retrieval_chain(
        retriever=retriever_hist,
        combine_docs_chain=combine_docs_chain
    )

def adapt_chroma_retriever(chroma_client):
    def chroma_retriever(query, n_results=1):
        result = chroma_client.query(query_texts=[query], n_results=n_results)

        documents = [doc['document'] for doc in result['documents']]
        return documents

    return chroma_retriever
