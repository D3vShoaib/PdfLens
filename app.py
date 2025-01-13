import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import google.generativeai as genai

# Sidebar contents
with st.sidebar:
    st.title('👁️‍🗨️ PFD Lens')
    st.markdown('''
    ## About
This app is an LLM-powered chatbot built using:  
- Streamlit
- LangChain  
- Google Gemini
''')
    add_vertical_space(2)
    st.markdown('Made with Excellence by:\n- Shoaib Ahmed\n- Krishna Sahu\n- Mohd Ahmad\n- Abdul Rehman')

load_dotenv()

def get_response(query, vector_store=None):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    try:
        if vector_store:
            docs = vector_store.similarity_search(query=query, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            full_prompt = """
                Context: {context}

                Question: {query}

                Instructions:
                - use natural language, and mix it will your knowledge
                - Answer should be a elaborated paragraph
                - If the answer cannot be found in the context answer or present partially in answer elaborately from your knowledge
                - If can't find any clue in context say "I didn't find the answer in the provided PDF"
                """.format(context=context, query=query)
            response = llm.invoke(full_prompt).content
        else:
            response = llm.invoke(query).content
        return response
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    st.header("Chat with PDF & Gemini 💬")

    if not os.getenv('GOOGLE_API_KEY'):
        st.error("Please set the GOOGLE_API_KEY environment variable")
        return

    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

    vector_store = None
    pdf = st.file_uploader("Upload your PDF (optional)", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
        else:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)

    query = st.text_input("Ask your question:")
    if query:
        response = get_response(query, vector_store)
        st.write(response)

if __name__ == '__main__':
    main()