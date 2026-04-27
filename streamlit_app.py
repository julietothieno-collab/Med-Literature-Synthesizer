import streamlit as st
import os

# --- 1. ROBUST IMPORTS ---
try:
    from PyPDF2 import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings # Switched for reliability
    from langchain_google_genai import ChatGoogleGenerativeAI
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
        color: white; border-radius: 20px; width: 100%; font-weight: bold; 
    }
    h1 { color: #1B4F72; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Medical Literature Synthesizer")
st.caption("Developed by Juliet Sera Othieno, MBBS Candidate")

# --- 3. SIDEBAR LOGIC ---
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Google Gemini API Key", type="password")
    pdf_docs = st.file_uploader("Upload Medical PDFs", accept_multiple_files=True)
    
    if st.button("Process Literature"):
        if api_key and pdf_docs:
            with st.spinner("Analyzing papers (this may take a minute)..."):
                # 1. Extract Text
                text = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted: text += extracted
                
                # 2. Chunk text
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_text(text)
                
                # 3. Create Vector Store using FREE HuggingFace Embeddings
                # This bypasses the Google GenerativeAIError
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_texts(chunks, embeddings)
                
                st.session_state.vector_store = vector_store
                st.session_state.api_key = api_key
                st.success("Analysis Complete!")
        else:
            st.warning("Please enter your API Key and upload PDFs.")

# --- 4. MAIN INTERFACE ---
user_query = st.text_input("Ask a research question across the uploaded papers:")

if user_query:
    if "vector_store" not in st.session_state:
        st.info("Please upload and process your PDFs in the sidebar first.")
    else:
        with st.spinner("Synthesizing evidence..."):
            os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
            
            # Retrieve relevant chunks
            docs = st.session_state.vector_store.similarity_search(user_query, k=4)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Use Gemini to answer
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
