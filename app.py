import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text could be extracted from the PDF. Please check the document.", icon="🚨")
        return False
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Failed to create vector store: {e}", icon="🚨")
        return False

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    Make sure to provide all the details. If the answer is not in the provided context,
    just say, "The answer is not available in the context." Do not provide a wrong answer.

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_gemini_response(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except FileNotFoundError:
        return "Vector store not found. Please process your PDF files first."
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    st.set_page_config(page_title="PDF Lens", page_icon="🔍", layout="wide")

    st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            border-radius: 0.5rem;
        }
        .st-emotion-cache-1y4p8pa {
            padding-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.image("https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original-wordmark.svg", width=50)
        st.header("PDF Lens 🔍")
        st.write("a RAG application to chat with multiple PDFs with AI")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Submit & Process", use_container_width=True, type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.", icon="⚠️")
            else:
                with st.spinner("Creating embeddings for vector-store..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("Failed to extract text. The PDF might be image-based or empty.", icon="🚨")
                        st.session_state.processed = False
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        if get_vector_store(text_chunks):
                            st.session_state.processed = True
                            st.session_state.messages = [
                                {"role": "assistant", "content": "Your documents are ready! Ask me anything about them."}
                            ]
                            st.success("Documents processed successfully!", icon="✅")
                            st.rerun()
                        else:
                            st.session_state.processed = False

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.processed:
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Chat cleared. Ask a new question about the loaded documents."}
                )
            st.rerun()

        st.markdown(
            """
            <div style="text-align: center;">
                <p>Made with ❤️ by D3vShoaib</p>
                <a href="https://github.com/D3vShoaib" target="_blank" style="text-decoration: none;">
                    <img src="https://github.githubassets.com/assets/apple-touch-icon-144x144-b882e354c005.png" alt="GitHub" width="32" height="32" style="filter: invert(0.1);">
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.title("Chat with multiple PDFs using Gemini Pro 💬")
    
    if not st.session_state.processed:
        st.info("Please upload and process your PDF documents in the sidebar to start chatting.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.processed:
            st.warning("Please upload and process your documents first.", icon="⚠️")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_gemini_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.", icon="🚨")
    else:
        main()