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

# --- 2. CLEAN WHITE MEDICAL UI (FIXED FONTS) ---
st.set_page_config(page_title="Med-Synth AI", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    /* White Background */
    .stApp { background-color: #FFFFFF; }
    
    /* Pure Black Text for Readability */
    p, li, label, span, .stMarkdown { color: #000000 !important; font-size: 1.1rem !important; }
    
    /* Professional Navy Headlines */
    h1, h2, h3 { color: #1B4F72 !important; font-weight: 800 !important; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] { background-color: #F8F9F9 !important; border-right: 1px solid #1B4F72; }
    
    /* INPUT BOX FIX: White text on dark background OR Black text on light background */
    .stTextInput>div>div>input { 
        color: #FFFFFF !important;  /* text color inside the box */
        background-color: #1B4F72 !important; /* navy blue box */
        border: 2px solid #1B4F72 !important;
    }
    
    /* Make the placeholder text white-ish so you can see it */
    input::placeholder { color: #D5DBDB !important; opacity: 1; }

    /* Button Styling */
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
    st.title("🧰 Tools")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    pdf_docs = st.file_uploader("Upload Medical PDFs", accept_multiple_files=True)
    
    if st.button("PROCESS LITERATURE"):
        if api_key and pdf_docs:
            with st.spinner("⏳ Analyzing PDFs..."):
                text = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted: text += extracted
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_text(text)
                
                # Using local embeddings for stability
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_texts(chunks, embeddings)
                
                st.session_state.vector_store = vector_store
                st.session_state.api_key = api_key
                st.success("✅ Ready!")
        else:
            st.warning("⚠️ Enter Key and PDFs.")

# --- 4. MAIN INTERFACE ---
st.title("🧬 Medical Literature Synthesizer")
st.markdown(f"**Research Lead:** Juliet Sera Othieno, MBBS Candidate")
st.markdown("---")

if "vector_store" not in st.session_state:
    st.info("👈 Please setup via the sidebar to begin clinical synthesis.")
else:
    st.markdown("### 📝 Enter Your Question")
    user_query = st.text_input("", placeholder="Wait 30s between questions to avoid rate limits...")

    if user_query:
        with st.spinner("🔬 Synthesizing..."):
            genai.configure(api_key=st.session_state.api_key)
            
            # Retrieve Evidence
            docs = st.session_state.vector_store.similarity_search(user_query, k=5)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Use the newer model version
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"System: Senior Medical Researcher. Answer based ONLY on context:\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{user_query}"
            
            try:
                response = model.generate_content(prompt)
                st.markdown("---")
                st.markdown("### 📊 Evidence Synthesis Report")
                st.write(response.text)
                st.balloons()
            except Exception as e:
                if "429" in str(e):
                    st.error("⚠️ **API Rate Limit Reached.** Google's free tier only allows 15 questions per minute. Please wait 30 seconds and try again!")
                else:
                    st.error(f"Error: {str(e)}")
