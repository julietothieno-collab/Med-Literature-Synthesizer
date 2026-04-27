import streamlit as st
import os

# --- 1. ROBUST IMPORTS ---
try:
    from PyPDF2 import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
except ImportError as e:
    st.error(f"Library Error: {e}")
    st.stop()

# --- 2. PREMIUM MEDICAL UI DESIGN ---
st.set_page_config(page_title="Med-Synth AI", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    /* Make all text darker and easier to read */
    .stApp { background-color: #FFFFFF; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    
    /* Strong Navy Blue Headlines */
    h1, h2, h3 { color: #1B4F72 !important; font-weight: 800 !important; }
    
    /* Make body text black (not gray) */
    p, li, label, .stMarkdown { color: #000000 !important; font-size: 1.1rem !important; }

    /* Custom Sidebar styling */
    section[data-testid="stSidebar"] { background-color: #F0F4F8 !important; border-right: 2px solid #1B4F72; }
    
    /* Highlight the "Ask a Question" input box */
    .stTextInput>div>div>input {
        border: 2px solid #1B4F72 !important;
        background-color: #FBFCFE !important;
        font-size: 1.2rem !important;
        color: #000000 !important;
    }

    /* Professional Button */
    .stButton>button { 
        background-color: #1B4F72 !important; 
        color: white !important; 
        border-radius: 10px !important; 
        font-weight: bold !important;
        height: 3em !important;
    }
    
    /* Success Box */
    .stSuccess {
        background-color: #D4EFDF !important;
        color: #145A32 !important;
        border: 1px solid #145A32 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR (SETUP) ---
with st.sidebar:
    st.title("🧰 Control Panel")
    st.markdown("---")
    api_key = st.text_input("1. Enter Gemini API Key", type="password", help="Get your key at aistudio.google.com")
    pdf_docs = st.file_uploader("2. Upload Research PDFs", accept_multiple_files=True)
    
    if st.button("3. START PROCESSING"):
        if api_key and pdf_docs:
            with st.spinner("🔍 Reading Medical Literature..."):
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
                st.success("✅ Analysis Complete! You can now ask questions on the right.")
        else:
            st.warning("⚠️ Missing Key or PDFs")

# --- 4. MAIN INTERFACE (QUESTIONING) ---
st.title("🧬 Medical Literature Synthesizer")
st.markdown(f"**Researcher:** Juliet Sera Othieno, MBBS Candidate")
st.markdown("---")

# Instruction for the user when they first open the app
if "vector_store" not in st.session_state:
    st.info("👈 **How to start:** Enter your API key and upload your study PDFs in the sidebar to begin.")
else:
    st.markdown("### 📝 Ask Your Clinical Question")
    st.markdown("Type below to search across all uploaded evidence:")
    
    user_query = st.text_input("", placeholder="e.g. Compare the P-values and results of these studies...", key="query_input")

    if user_query:
        with st.spinner("🔬 Synthesizing Research Findings..."):
            os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
            
            # Retrieve relevant chunks
            docs = st.session_state.vector_store.similarity_search(user_query, k=4)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Use Gemini-1.5-Flash
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
            
            clinical_prompt = f"""
            You are a Senior Medical Researcher. Synthesize a professional, high-quality response using the context provided.
            Provide a clear summary, highlight data points (like sample sizes or dosages), and note any contradictions.
            
            CONTEXT:
            {context_text}
            
            QUESTION:
            {user_query}
            
            ACADEMIC RESPONSE:
            """
            
            response = llm.invoke([HumanMessage(content=clinical_prompt)])
            
            # Clearer result container
            st.markdown("---")
            st.markdown("### 📊 Evidence Synthesis Report")
            st.success("Targeted insights generated from your literature:")
            st.write(response.content)
            st.markdown("---")
            st.caption("AI-generated synthesis based on uploaded documents. Always verify clinical data.")
