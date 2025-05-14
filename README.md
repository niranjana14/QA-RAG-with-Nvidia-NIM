# 📄 Nvidia NIM PDF Q&A Demo

This is a simple Streamlit app that allows users to upload PDF documents, generate vector embeddings using **NVIDIA NIM embeddings**, and ask questions over the document content using **LLM-powered retrieval-augmented generation (RAG)**.

---

## 🚀 Features

- Upload one or more PDF files directly in the UI
- Generate a FAISS vector store for the uploaded PDFs
- Ask questions based on the PDF contents
- Uses NVIDIA's LLM (`meta/llama3-70b-instruct`) via Langchain
- View the matching document chunks for transparency
- Simple, fast, and interactive Streamlit interface

---

## 🧠 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Langchain](https://www.langchain.com/)
- [NVIDIA NIM Embeddings & LLMs](https://catalog.ngc.nvidia.com/)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [PyMuPDF / PyPDFLoader](https://pymupdf.readthedocs.io/)

---
## Screenshots
![image](https://github.com/user-attachments/assets/040eddc2-d4da-4092-a597-a436a600dad9)
![image](https://github.com/user-attachments/assets/b41ae2e8-6d99-4fe0-b614-d5aede2c0721)



