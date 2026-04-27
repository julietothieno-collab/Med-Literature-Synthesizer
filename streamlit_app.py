import streamlit as st
import os

# --- 1. MODERN LANGCHAIN 1.0+ IMPORTS ---
try:
    from PyPDF2 import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage
except ImportError as e:
    st.error(f"⚠️ Library Error: {e}")
    st.stop()

# --- 2. APP CONFIGURATION ---
st.set_page_config(page_title="Med-Synth AI", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    .stButton>button { 
        background-color: #1B4F72; 
        color: white; 
        border-radius: 20px; 
        border: none;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input {
        border: 2px solid #1B4F72;
    }
    h1 { color: #1B4F72; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Medical Literature Synthesizer")
st.caption("Developed by Juliet Sera Othieno, MBBS Candidate | AI-Powered Research Assistant")
st.markdown("---")

# --- 3. SIDEBAR LOGIC ---
with st.sidebar:
    st.header("Upload & Setup")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get a free key at aistudio.google.com")
    pdf_docs = st.file_uploader("Upload Medical PDFs", accept_multiple_files=True)
    
    if st.button("Process Literature"):
        if api_key and pdf_docs:
            with st.spinner("Processing medical PDFs..."):
                os.environ["GOOGLE_API_KEY"] = api_key
                text = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = splitter.split_text(text)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_texts(chunks, embeddings)
                st.session_state.vector_store = vector_store
                st.success("Analysis Complete!")
        else:
            st.warning("Please enter your API Key and upload at least one PDF.")

# --- 4. MAIN INTERFACE ---
user_query = st.text_input("Ask a research question across the uploaded papers:")

if user_query:
    if "vector_store" not in st.session_state:
        st.info("Please upload and process your PDFs in the sidebar first.")
    else:
        with st.spinner("Synthesizing evidence..."):
            os.environ["GOOGLE_API_KEY"] = api_key
            
            # Retrieve relevant chunks
            docs = st.session_state.vector_store.similarity_search(user_query, k=5)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Modern LLM Call (Bypassing the crashing 'chains' module)
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
            
            clinical_prompt = f"""
            You are a Senior Medical Researcher. Synthesize a professional answer using the context provided below.
            If the answer is not in the context, state that the literature does not provide enough information.
            
            CONTEXT:
            {context_text}
            
            QUESTION:
            {user_query}
            
            ACADEMIC RESPONSE:
            """
            
            response = llm.invoke([HumanMessage(content=clinical_prompt)])
            
            st.markdown("### 📊 Evidence Synthesis Report")
            st.write(response.content)
