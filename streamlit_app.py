import streamlit as st
import os

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

# --- 2. DARK PREMIUM MEDICAL UI ---
st.set_page_config(page_title="Med-Synth AI", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    /* Dark Theme background */
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    /* White text for all standard elements */
    p, li, label, span, .stMarkdown { color: #FFFFFF !important; font-size: 1.1rem !important; }
    
    /* Blue Glow Headlines */
    h1, h2, h3 { color: #5DADE2 !important; font-weight: 800 !important; text-shadow: 2px 2px 4px #000000; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #1B2631 !important; border-right: 1px solid #5DADE2; }
    
    /* Input Box Styling */
    .stTextInput>div>div>input { 
        border: 1px solid #5DADE2 !important; 
        color: #FFFFFF !important; 
        background-color: #2C3E50 !important; 
    }

    /* Professional Button */
    .stButton>button { 
        background-color: #5DADE2 !important; 
        color: #0E1117 !important; 
        border-radius: 10px !important; 
        font-weight: bold !important; 
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #AED6F1 !important; transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR (SETUP) ---
with st.sidebar:
    st.title("🧰 Control Panel")
    st.markdown("---")
    api_key = st.text_input("Gemini API Key", type="password")
    pdf_docs = st.file_uploader("Upload Medical PDFs", accept_multiple_files=True)
    
    if st.button("PROCESS DOCUMENTS"):
        if api_key and pdf_docs:
            with st.spinner("⏳ Analyzing Literature..."):
                text = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted: text += extracted
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
                chunks = splitter.split_text(text)
                
                # Local safe embeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_texts(chunks, embeddings)
                
                st.session_state.vector_store = vector_store
                st.session_state.api_key = api_key
                st.success("✅ Analysis Complete!")
        else:
            st.warning("⚠️ Missing API Key or PDFs")

# --- 4. MAIN INTERFACE ---
st.title("🧬 Medical Literature Synthesizer")
st.markdown(f"**Research Lead:** Juliet Sera Othieno, Year 4 MBBS Candidate")
st.markdown("---")

if "vector_store" not in st.session_state:
    st.info("👈 **Welcome Doctor.** Please upload your study materials in the sidebar to begin.")
else:
    st.markdown("### 📝 Enter Research Query")
    user_query = st.text_input("", placeholder="e.g. Discuss the statistical significance of outcomes across these trials...")

    if user_query:
        with st.spinner("🔬 AI Synthesizing evidence..."):
            genai.configure(api_key=st.session_state.api_key)
            
            # Retrieve Evidence
            docs = st.session_state.vector_store.similarity_search(user_query, k=5)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # --- MODEL AUTO-DETECTION LOGIC ---
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # Try to pick Gemini 1.5 Flash first, then 1.5 Pro, then fallback to whatever is available
            if 'models/gemini-1.5-flash' in available_models:
                target_model = 'gemini-1.5-flash'
            elif 'models/gemini-1.5-pro' in available_models:
                target_model = 'gemini-1.5-pro'
            else:
                target_model = available_models[0].split('/')[-1] # Automated selection

            model = genai.GenerativeModel(target_model)
            
            prompt = f"""
            You are a Professional Medical Data Analyst. 
            Answer the clinical question based ONLY on the evidence provided.
            
            CONTEXT:
            {context_text}
            
            QUESTION:
            {user_query}
            
            Format the response with headers and bullet points. 
            If conflicting data exists between papers, highlight it clearly.
            """
            
            try:
                response = model.generate_content(prompt)
                st.markdown(f"**Using Model:** `{target_model}`")
                st.markdown("---")
                st.markdown("### 📊 Evidence Synthesis Report")
                st.write(response.text)
                st.balloons() # Visual celebration for success!
            except Exception as e:
                st.error(f"Synthesis failed using {target_model}")
                st.info(f"Details: {str(e)}")
