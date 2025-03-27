import os
import streamlit as st
import time
from datetime import datetime
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Função para carregar textos dos PDFs, com OCR quando necessário
def carregar_documentos(pasta):
    documentos = []

    for root, dirs, files in os.walk(pasta):
        for file in files:
            if file.endswith(".pdf"):
                caminho = os.path.join(root, file)
                texto_pdf = ""

                try:
                    leitor = PdfReader(caminho)
                    for pagina in leitor.pages:
                        texto = pagina.extract_text()
                        if texto:
                            texto_pdf += texto
                except Exception as e:
                    print(f"Erro ao ler {file} com PyPDF2: {e}")

                if not texto_pdf.strip():
                    # Aplica OCR se não conseguiu extrair texto
                    try:
                        imagens = convert_from_path(caminho)
                        for img in imagens:
                            texto_pdf += pytesseract.image_to_string(img, lang='eng')
                        print(f"OCR aplicado com sucesso em: {file}")
                    except Exception as ocr_error:
                        print(f"Erro no OCR para {file}: {ocr_error}")
                        continue

                if texto_pdf.strip():
                    documentos.append(Document(page_content=texto_pdf, metadata={"fonte": file}))
                else:
                    print(f"Aviso: Nenhum conteúdo extraído do arquivo {file}.")

    return documentos

# Função para criar ou carregar o banco vetorial FAISS do disco
def criar_ou_carregar_vetores(textos, caminho_index="faiss_index"):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = []
    for texto in textos:
        partes = splitter.split_documents([texto])
        chunks.extend(partes)

    if not chunks:
        raise ValueError("Nenhum conteúdo válido foi encontrado nos documentos.")

    base_vetorial = FAISS.from_documents(chunks, embeddings)
    base_vetorial.save_local(caminho_index)

    # Atualiza o timestamp da última atualização
    with open("ultima_atualizacao.txt", "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    return base_vetorial

# Interface do chatbot com Streamlit
def chatbot(base_vetorial):
    st.title("DígitosChatAI")
    st.write("Faça sua pergunta sobre legislação, manuais, procedimentos e convenções coletivas.")

    # Mostra a data da última atualização
    if os.path.exists("ultima_atualizacao.txt"):
        with open("ultima_atualizacao.txt", "r") as f:
            data = f.read().strip()
            st.sidebar.info(f"📅 Base atualizada em: {data}")

    modelo_escolhido = st.selectbox("Escolha o modelo da OpenAI:", ["gpt-3.5-turbo", "gpt-4"])
    pergunta = st.text_input("Digite sua pergunta:")

    if pergunta:
        modelo = ChatOpenAI(
            temperature=0,
            model=modelo_escolhido,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        qa = RetrievalQA.from_chain_type(
            llm=modelo,
            chain_type="stuff",
            retriever=base_vetorial.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True
        )

        resposta_completa = qa({"query": pergunta})
        resposta = resposta_completa["result"]
        fontes = resposta_completa["source_documents"]

        st.write("**Resposta:**")
        st.write(resposta)

        if not fontes:
            st.warning("\u26a0\ufe0f Não encontrei essa informação nos documentos.")
        else:
            nomes_fontes = set(doc.metadata['fonte'] for doc in fontes)
            st.write("**Fonte(s):**", ", ".join(nomes_fontes))

# Executar
if __name__ == "__main__":
    caminho_index = "faiss_index"
    atualizar_base = st.sidebar.button("🔄 Recarregar base de documentos")

    if os.path.exists(caminho_index) and not atualizar_base:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        base = FAISS.load_local(caminho_index, embeddings, allow_dangerous_deserialization=True)
    else:
        with st.spinner("Carregando e processando os documentos..."):
            docs = carregar_documentos("documentos")
            base = criar_ou_carregar_vetores(docs, caminho_index)
            st.success("Base de documentos atualizada com sucesso!")

    chatbot(base)
