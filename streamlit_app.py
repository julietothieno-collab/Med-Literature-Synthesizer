import streamlit as st
import os
import time

# --- 1. ROBUST IMPORTS ---
try:
    from PyPDF2 import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    import google.generativeai as genai
except ImportError as e:
    st.error(f"Library Error: {e}")
    st.stop()

# --- 2. PREMIUM MEDICAL UI (BOLD & CLEAN) ---
st.set_page_config(page_title="Med-Synth AI", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #FFFFFF; }
    
    /* Elegant Large Title */
    .main-title {
        font-size: 55px !important;
        font-weight: 800 !important;
        color: #1B4F72 !important;
        text-align: center;
        margin-bottom: 0px;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Subtitle */
    .sub-title {
        font-size: 1.2rem !important;
        color: #5D6D7E !important;
        text-align: center;
        margin-bottom: 30px;
    }

    /* Black Reading Text */
    p, li, label, span, .stMarkdown { color: #000000 !important; font-size: 1.1rem !important; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] { background-color: #F4F6F7 !important; border-right: 2px solid #1B4F72; }
    
    /* Input Box: High Contrast */
    .stTextInput>div>div>input { 
        color: #FFFFFF !important; 
        background-color: #1B4F72 !important; 
        border-radius: 10px;
        padding: 15px;
        font-size: 1.2rem !important;
    }
    
    /* Signature at the bottom */
    .signature {
        position: fixed;
        left: 0;
        bottom: 10px;
        width: 100%;
        text-align: center;
        color: #AEB6BF;
        font-size: 0.8rem;
        font-style: italic;
    }

    .stButton>button { 
        background-color: #1B4F72 !important; 
        color: white !important; 
        border-radius: 10px !important; 
        font-weight: bold !important; 
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    pdf_docs = st.file_uploader("Upload Medical PDFs", accept_multiple_files=True)
    
    if st.button("PROCESS LITERATURE"):
        if api_key and pdf_docs:
            with st.spinner("🔍 Reading PDFs..."):
                text = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted: text += extracted
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_text(text)
                
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_texts(chunks, embeddings)
                
                st.session_state.vector_store = vector_store
                st.session_state.api_key = api_key
                st.success("✅ Analysis Complete!")
        else:
            st.warning("⚠️ Missing setup details.")

# --- 4. MAIN INTERFACE ---
st.markdown('<p class="main-title">🧬 Med-Synth AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Advanced Medical Literature Synthesis & Data Extraction</p>', unsafe_allow_html=True)

if "vector_store" not in st.session_state:
    st.info("👈 Please enter your API key and upload your clinical papers in the sidebar to begin.")
else:
    st.markdown("### 📝 Ask your Research Question")
    user_query = st.text_input("", placeholder="e.g., Summarize the findings and mention the sample sizes...")

    if user_query:
        with st.spinner("🔬 Synthesizing Evidence..."):
            genai.configure(api_key=st.session_state.api_key)
            
            # Retrieve relevant text
            docs = st.session_state.vector_store.similarity_search(user_query, k=5)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # --- THE 404 FIX: AUTO-SELECT THE CORRECT MODEL ---
            try:
                # Find available models for your specific API key
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                # Priority list
                if 'models/gemini-1.5-flash-latest' in models: selection = 'gemini-1.5-flash-latest'
                elif 'models/gemini-1.5-flash' in models: selection = 'gemini-1.5-flash'
                elif 'models/gemini-pro' in models: selection = 'gemini-pro'
                else: selection = models[0].replace('models/', '')

                model = genai.GenerativeModel(selection)
                
                prompt = f"System: Medical Researcher. Context:\n{context_text}\n\nQuestion: {user_query}"
                response = model.generate_content(prompt)
                
                st.markdown("---")
                st.markdown("### 📊 Synthesis Report")
                st.write(response.text)
                st.balloons()
            except Exception as e:
                st.error(f"Generation failed. Error details: {str(e)}")

# --- 5. SIGNATURE ---
st.markdown('<div class="signature">Developed by Juliet Sera Othieno, MBBS Candidate</div>', unsafe_allow_html=True)
