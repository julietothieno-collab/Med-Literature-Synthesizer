import streamlit as st
import os

# --- 1. ROBUST IMPORTS ---
try:
    from PyPDF2 import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    # Direct import for better stability
    import google.generativeai as genai
except ImportError as e:
    st.error(f"Library Error: {e}")
    st.stop()

# --- 2. UI DESIGN ---
st.set_page_config(page_title="Med-Synth AI", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    h1, h2, h3 { color: #1B4F72 !important; font-weight: 800 !important; }
    p, li, label, .stMarkdown { color: #000000 !important; font-size: 1.1rem !important; }
    section[data-testid="stSidebar"] { background-color: #F0F4F8 !important; border-right: 2px solid #1B4F72; }
    .stTextInput>div>div>input { border: 2px solid #1B4F72 !important; color: #000000 !important; }
    .stButton>button { background-color: #1B4F72 !important; color: white !important; border-radius: 10px !important; font-weight: bold !important; height: 3em !important; width: 100%; }
    /* Success Box */
    .stSuccess { background-color: #D4EFDF !important; color: #145A32 !important; border: 1px solid #145A32 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR (SETUP) ---
with st.sidebar:
    st.title("🧰 Setup Panel")
    api_key = st.text_input("1. Enter Gemini API Key", type="password")
    pdf_docs = st.file_uploader("2. Upload Medical PDFs", accept_multiple_files=True)
    
    if st.button("3. PROCESS DOCUMENTS"):
        if api_key and pdf_docs:
            with st.spinner("🔍 Analyzing Literature..."):
                text = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted: text += extracted
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
                chunks = splitter.split_text(text)
                
                # Using local HuggingFace embeddings for stability
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_texts(chunks, embeddings)
                
                st.session_state.vector_store = vector_store
                st.session_state.api_key = api_key
                st.success("✅ Ready! Ask your question on the right.")
        else:
            st.warning("⚠️ Missing API Key or PDFs")

# --- 4. MAIN INTERFACE ---
st.title("🧬 Medical Literature Synthesizer")
st.markdown(f"**Research Lead:** Juliet Sera Othieno, MBBS Candidate")
st.markdown("---")

if "vector_store" not in st.session_state:
    st.info("👈 **Welcome!** Please start by entering your API key and uploading PDFs in the sidebar.")
else:
    st.markdown("### 📝 Clinical Question")
    user_query = st.text_input("", placeholder="e.g. Synthesize the findings regarding patient outcomes...", key="query_input")

    if user_query:
        with st.spinner("🔬 AI Synthesizing evidence..."):
            # Setup GenAI directly
            genai.configure(api_key=st.session_state.api_key)
            
            # Retrieve context from local vector store
            docs = st.session_state.vector_store.similarity_search(user_query, k=5)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Use a robust direct call to Gemini Pro
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            You are a Senior Medical Researcher. Synthesize a professional response based ONLY on the evidence provided.
            
            STUDY CONTEXT:
            {context_text}
            
            RESEARCH QUESTION:
            {user_query}
            
            Please provide a structured, high-contrast academic summary.
            """
            
            try:
                # Direct generate content call (More stable than LangChain chains)
                response = model.generate_content(prompt)
                
                st.markdown("---")
                st.markdown("### 📊 Evidence Synthesis Report")
                st.write(response.text)
                st.success("Synthesis complete based on provided documents.")
            except Exception as e:
                st.error("AI Generation failed. This is usually due to regional API restrictions or an invalid key.")
                st.info(f"Technical error details: {str(e)}")
