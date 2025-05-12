import os
import shutil
import streamlit as st
from datetime import datetime
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Templates aprimorados para respostas mais completas e amigáveis
CONDENSE_PROMPT = PromptTemplate.from_template(
    "Resumo do histórico de conversa: {chat_history}\n" 
    "Reformule a seguinte pergunta de forma clara e detalhada: {question}"
)
QA_PROMPT = PromptTemplate.from_template(
    "Use apenas o contexto abaixo para responder à pergunta de forma detalhada e abrangente."
    "\nSe a informação não estiver presente no contexto, responda de forma educada:"
    " \"Desculpe, não encontrei essa informação no documento. Posso ajudar em outra coisa?\"\n\n"
    "Contexto:\n{context}\n\n"
    "Pergunta: {question}\nResposta detalhada:"
)

# Carrega texto de um único PDF (texto ou OCR)
def carregar_documento(caminho):
    texto = ""
    try:
        leitor = PdfReader(caminho)
        for pagina in leitor.pages:
            texto += pagina.extract_text() or ""
    except Exception:
        pass
    if not texto.strip():
        try:
            imagens = convert_from_path(caminho)
            for img in imagens:
                texto += pytesseract.image_to_string(img, lang='eng')
        except Exception:
            pass
    nome = os.path.basename(caminho)
    return [Document(page_content=texto, metadata={"fonte": nome})] if texto.strip() else []

# Cria ou carrega índice FAISS, verificando ambos arquivos .faiss e .pkl
def criar_ou_carregar_indice(docs, index_path="faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")
    # Tenta carregar índice existente somente se ambos arquivos existirem
    if os.path.isdir(index_path) and os.path.exists(faiss_file) and os.path.exists(pkl_file):
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            # Se falhar ao carregar, remove índice corrompido e recria abaixo
            shutil.rmtree(index_path)
    # Cria novo índice
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    base = FAISS.from_documents(chunks, embeddings)
    base.save_local(index_path)
    return base

# Interface do chatbot
def chatbot(base, index_path):
    st.title("🤖 DígitosChatAI")
    if "history" not in st.session_state:
        st.session_state.history = []

    # Botão recarregar base
    if st.sidebar.button("🔄 Recarregar base"):
        if os.path.isdir(index_path):
            shutil.rmtree(index_path)
        st.experimental_rerun()

    # Botão limpar histórico
    if st.sidebar.button("🗑️ Limpar conversa"):
        st.session_state.history = []
        st.sidebar.success("Histórico limpo!")

    modelo = ChatOpenAI(
        temperature=0.5,
        model=st.sidebar.selectbox("Modelo:", ["gpt-3.5-turbo", "gpt-4"]),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    pergunta = st.text_input("Digite sua pergunta:")
    if pergunta:
        chain = ConversationalRetrievalChain.from_llm(
            llm=modelo,
            retriever=base.as_retriever(search_kwargs={"k": 3}),
            condense_question_prompt=CONDENSE_PROMPT,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )
        res = chain({"question": pergunta, "chat_history": st.session_state.history})
        resposta = res["answer"].strip()
        fontes = res.get("source_documents", [])

        st.session_state.history.append((pergunta, resposta))
        st.write("**Resposta:**")
        st.write(resposta)
        if fontes:
            st.write("**Fonte:**", fontes[0].metadata.get('fonte', ''))
        else:
            st.warning("⚠️ Informação não encontrada no documento.")

    # Exibe histórico da conversa
    with st.expander("🕒 Histórico da conversa"):
        for i, (q, a) in enumerate(st.session_state.history, 1):
            st.markdown(f"**{i}. Pergunta:** {q}")
            st.markdown(f"**Resposta:** {a}")
            st.markdown("---")

# Execução principal
if __name__ == "__main__":
    pdfs = [f for f in os.listdir("documentos") if f.lower().endswith('.pdf')]
    if not pdfs:
        st.error("Nenhum PDF encontrado na pasta 'documentos'.")
        st.stop()
    if len(pdfs) > 1:
        st.error("Mais de um PDF encontrado. Deixe apenas um arquivo em 'documentos'.")
        st.stop()

    caminho = os.path.join("documentos", pdfs[0])
    docs = carregar_documento(caminho)
    if not docs:
        st.error("Falha ao extrair texto do PDF.")
        st.stop()

    index_path = "faiss_index"
    base = criar_ou_carregar_indice(docs, index_path)
    chatbot(base, index_path)
