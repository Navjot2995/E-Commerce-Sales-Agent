import streamlit as st
import os
import time
import requests
import base64
from typing import List
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Configuration ---
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DB_PATH = "vectorstore"
DOCUMENTS_PATH = "documents"

# Available Groq models with their descriptions
GROQ_MODELS = {
    "llama3-70b-8192": {
        "name": "Llama 3 70B",
        "description": "Latest and most powerful Llama model, best for complex reasoning and detailed responses",
        "max_tokens": 8192
    },
    "llama3-8b-8192": {
        "name": "Llama 3 8B",
        "description": "Efficient and fast model for general tasks",
        "max_tokens": 8192
    },
    "mixtral-8x7b-32768": {
        "name": "Mixtral 8x7B",
        "description": "Powerful mixture of experts model, excellent for complex tasks",
        "max_tokens": 32768
    },
    "gemma-7b-it": {
        "name": "Gemma 7B",
        "description": "Google's latest model, optimized for efficiency and speed",
        "max_tokens": 8192
    }
}

MAX_RETRIES = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Initialize Groq Client ---
@st.cache_resource
def init_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in environment variables")
        st.stop()
    return Groq(api_key=api_key)

client = init_groq_client()

# --- Document Processing ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_document(file_path: str):
    try:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".csv"):
            # Handle CSV files as text files
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        return loader.load()
    except Exception as e:
        st.error(f"Error loading document {file_path}: {str(e)}")
        return []

def process_documents():
    try:
        if not os.path.exists(DOCUMENTS_PATH):
            st.error(f"Documents directory not found: {DOCUMENTS_PATH}")
            return False

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        all_docs = []
        progress_bar = st.progress(0)
        files = [f for f in os.listdir(DOCUMENTS_PATH) if os.path.isfile(os.path.join(DOCUMENTS_PATH, f))]
        total_files = len(files)
        
        if total_files == 0:
            st.error("No files found in documents directory")
            return False

        st.info(f"Found {total_files} files to process")
        
        for idx, filename in enumerate(files):
            file_path = os.path.join(DOCUMENTS_PATH, filename)
            st.write(f"Processing {filename}...")
            docs = load_document(file_path)
            if docs:
                splits = text_splitter.split_documents(docs)
                all_docs.extend(splits)
                st.write(f"Added {len(splits)} chunks from {filename}")
            progress_bar.progress((idx + 1) / total_files)
        
        if all_docs:
            st.write(f"Creating vector store with {len(all_docs)} total chunks...")
            vectorstore = FAISS.from_documents(all_docs, OpenAIEmbeddings(model=EMBEDDING_MODEL))
            vectorstore.save_local(VECTOR_DB_PATH)
            st.success(f"Successfully processed {len(all_docs)} chunks from {total_files} files")
            return True
        else:
            st.error("No valid documents were processed")
            return False
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

# --- RAG Setup with Groq ---
@st.cache_resource
def setup_retriever():
    try:
        if not os.path.exists(VECTOR_DB_PATH):
            st.warning("Vector store not found. Please process documents first.")
            return None
            
        st.write("Loading vector store...")
        vectorstore = FAISS.load_local(VECTOR_DB_PATH, OpenAIEmbeddings(model=EMBEDDING_MODEL), allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.success("Vector store loaded successfully")
        return retriever
    except Exception as e:
        st.error(f"Error setting up retriever: {str(e)}")
        return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Sales-specific prompt template optimized for Groq
prompt_template = """
You are an expert sales assistant for an e-commerce company. Use the provided context to answer questions.

Context:
{context}

Question:
{question}

Guidelines:
1. Start with a warm, professional greeting
2. Provide concise, accurate information
3. Highlight product benefits and specifications
4. Include pricing if available
5. End with a call-to-action (e.g., "Shall I check stock availability?")

Response:
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# --- Streamlit UI ---
def img_to_bytes(img_url):
    try:
        response = requests.get(img_url, timeout=5)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return ""

def app_header():
    header_html = f"""
    <div style="background:linear-gradient(135deg, #4b6cb7, #182848);padding:20px;border-radius:10px;margin-bottom:20px;color:white">
        <table>
            <tr>
                <td><img src="data:image/png;base64,{img_to_bytes('https://img.icons8.com/color/480/black-friday.png')}" width=80></td>
                <td><h1 style="color:white;margin-left:20px">Turbo Sales Assistant ⚡</h1></td>
            </tr>
        </table>
        <p style="margin:0;font-size:14px">Powered by Groq's ultra-fast LLM</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm your lightning-fast sales assistant. How can I help you today?"
        })

def display_chat():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Groq Chat Completion ---
@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def groq_chat_completion(context: str, question: str, model: str) -> str:
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt_template.format(context=context, question=question)
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            model=model,
            temperature=0.3,
            max_tokens=GROQ_MODELS[model]["max_tokens"]
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        raise

# --- Main App ---
def main():
    initialize_chat()
    app_header()
    
    # Sidebar for document upload and management
    with st.sidebar:
        st.header("Knowledge Base Setup")
        
        # Model Selection
        st.subheader("Model Selection")
        selected_model = st.selectbox(
            "Choose a model",
            options=list(GROQ_MODELS.keys()),
            format_func=lambda x: f"{GROQ_MODELS[x]['name']} - {GROQ_MODELS[x]['description']}"
        )
        
        # Document upload
        st.subheader("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload product docs (PDF/TXT/DOCX/CSV)",
            type=["pdf", "txt", "docx", "csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Documents"):
            os.makedirs(DOCUMENTS_PATH, exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DOCUMENTS_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.write(f"Saved {uploaded_file.name}")
            
            with st.spinner("Processing documents..."):
                if process_documents():
                    st.success("Documents processed!")
                    st.session_state.retriever = setup_retriever()
                else:
                    st.error("No valid documents found")
        
        # Document management
        st.markdown("---")
        st.subheader("Manage Documents")
        if os.path.exists(DOCUMENTS_PATH):
            files = os.listdir(DOCUMENTS_PATH)
            if files:
                for file in files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(file)
                    with col2:
                        if st.button("🗑️", key=f"delete_{file}"):
                            os.remove(os.path.join(DOCUMENTS_PATH, file))
                            st.rerun()
            else:
                st.info("No documents uploaded yet")
        
        st.markdown("---")
        st.markdown("⚡ **Performance Info**")
        st.markdown(f"Using: `{GROQ_MODELS[selected_model]['name']}`")
        st.markdown(f"Max tokens: {GROQ_MODELS[selected_model]['max_tokens']}")
        st.markdown("Response times typically < 1s")
    
    # Main chat interface
    col1, col2 = st.columns([5, 1])
    with col1:
        display_chat()
    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "Hello! I'm your lightning-fast sales assistant. How can I help you today?"
            }]
            st.rerun()
    
    # Initialize retriever
    if "retriever" not in st.session_state:
        st.session_state.retriever = setup_retriever()
    
    # Chat input
    if prompt := st.chat_input("Ask about products..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if not st.session_state.retriever:
                st.error("Please upload product documents first")
                response = "I need product information to assist you. Please upload documents in the sidebar."
            else:
                start_time = time.time()
                with st.spinner("Generating lightning-fast response..."):
                    try:
                        # Retrieve relevant context
                        docs = st.session_state.retriever.get_relevant_documents(prompt)
                        context = format_docs(docs)
                        
                        # Get response from Groq
                        response = groq_chat_completion(context, prompt, selected_model)
                        
                        # Display performance
                        end_time = time.time()
                        st.caption(f"Generated in {end_time-start_time:.2f}s | {len(docs)} sources used")
                    except Exception as e:
                        response = f"Error: {str(e)}"
            
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()