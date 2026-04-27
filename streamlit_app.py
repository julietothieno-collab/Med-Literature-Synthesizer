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

# --- 2. PREMIUM MEDICAL UI ---
st.set_page_config(page_title="Med-Synth AI", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; }
    
    .main-title {
        font-size: 50px !important;
        font-weight: 800 !important;
        color: #1B4F72 !important;
        text-align: center;
        margin-top: -20px;
    }
    
    .sub-title {
        font-size: 1.1rem !important;
        color: #5D6D7E !important;
        text-align: center;
        margin-bottom: 30px;
    }

    p, li, label, span, .stMarkdown { color: #000000 !important; font-size: 1.05rem !important; }
    
    section[data-testid="stSidebar"] { background-color: #F4F6F7 !important; border-right: 2px solid #1B4F72; }
    
    /* Input Box: High Contrast White Font */
    .stTextInput>div>div>input { 
        color: #FFFFFF !important; 
        background-color: #1B4F72 !important; 
        border-radius: 8px;
        font-size: 1.1rem !important;
    }
    
    /* Make signature small and clean */
    .signature {
        text-align: center;
        color: #BDC3C7;
        font-size: 0.75rem;
        margin-top: 50px;
        padding-bottom: 20px;
    }

    .stButton>button { 
        background-color: #1B4F72 !important; 
        color: white !important; 
        border-radius: 8px !important; 
        font-weight: bold !important; 
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
            with st.spinner("🔍 Reading Medical Data..."):
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
            st.warning("⚠️ Enter details first.")

# --- 4. MAIN INTERFACE ---
st.markdown('<p class="main-title">🧬 Med-Synth AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Synthesizing clinical evidence for professional research</p>', unsafe_allow_html=True)

if "vector_store" not in st.session_state:
    st.info("👈 Welcome. Please configure the tool in the sidebar to begin.")
else:
    st.markdown("### 📝 Enter Research Question")
    user_query = st.text_input("", placeholder="e.g. Summarize the clinical implications and primary outcomes...")

    if user_query:
        with st.spinner("🔬 AI is synthesizing..."):
            genai.configure(api_key=st.session_state.api_key)
            docs = st.session_state.vector_store.similarity_search(user_query, k=5)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Smart Model Auto-Detection
            try:
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                selection = 'gemini-1.5-flash' if 'models/gemini-1.5-flash' in models else models[0].replace('models/', '')
                model = genai.GenerativeModel(selection)
                
                prompt = f"Role: Medical Analyst. Context:\n{context_text}\n\nQuestion: {user_query}. Respond with clear bullet points."
                
                # RETRY LOGIC FOR QUOTA
                max_retries = 3
                for i in range(max_retries):
                    try:
                        response = model.generate_content(prompt)
                        st.markdown("---")
                        st.markdown("### 📊 Evidence Synthesis Report")
                        st.write(response.text)
                        st.balloons()
                        break
                    except Exception as e:
                        if "429" in str(e) and i < max_retries - 1:
                            st.warning(f"⏳ Rate limit hit. Retrying automatically in 15 seconds (Attempt {i+1}/3)...")
                            time.sleep(15)
                        else:
                            raise e
            except Exception as e:
                st.error("The API is busy. Please wait 1 minute and re-enter your question.")

# --- 5. SIGNATURE ---
st.markdown('<div class="signature">Developed by Juliet Sera Othieno, MBBS Candidate</div>', unsafe_allow_html=True)
