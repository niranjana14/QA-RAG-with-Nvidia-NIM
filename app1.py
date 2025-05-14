import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import tempfile
import time
from dotenv import load_dotenv

load_dotenv()

# Load API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

st.title("📄 Nvidia NIM PDF Q&A Demo")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if "vectors" not in st.session_state:
    st.session_state.vectors = None

def vector_embedding_from_uploads(files):
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_chunks = splitter.split_documents(all_docs)
    embeddings = NVIDIAEmbeddings()
    vectorstore = FAISS.from_documents(final_chunks, embeddings)
    return vectorstore

if st.button("🔄 Generate Embeddings"):
    if uploaded_files:
        st.session_state.vectors = vector_embedding_from_uploads(uploaded_files)
        st.success("✅ Vector DB created and ready!")
    else:
        st.warning("⚠️ Please upload PDF files first.")

question = st.text_input("💬 Ask a question based on uploaded documents")

if question and st.session_state.vectors:
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

    prompt = ChatPromptTemplate.from_template(
       """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions:{input}

        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': question})
    st.write("🧠 Answer:", response['answer'])
    st.write("⏱️ Response time:", round(time.process_time() - start, 2), "sec")

    with st.expander("🔍 Document Chunks Retrieved"):
        for doc in response["context"]:
            st.write(doc.page_content)
            st.markdown("---")
elif question:
    st.warning("⚠️ Please generate embeddings before asking questions.")
