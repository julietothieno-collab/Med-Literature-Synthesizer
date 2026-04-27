import streamlit as st
import os

# --- 1. SELF-HEALING IMPORTS ---
try:
    from PyPDF2 import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
except ImportError as e:
    st.error(f"⚠️ **Deployment Header Error:** {e}")
    st.info("Streamlit is still installing libraries from requirements.txt. Please wait 2-3 minutes and refresh.")
    st.stop()

# --- 2. APP CONFIGURATION ---
st.set_page_config(page_title="Med-Synth AI", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stButton>button { background-color: #1B4F72; color: white; border-radius: 8px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Medical Literature Synthesizer")
st.caption("Developed by Juliet Sera Othieno, MBBS Candidate")
st.markdown("---")

# --- 3. SIDEBAR LOGIC ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google Gemini API Key", type="password")
    pdf_docs = st.file_uploader("Upload Medical PDFs", accept_multiple_files=True)
    
    if st.button("Analyze Literature"):
        if api_key and pdf_docs:
            with st.spinner("Analyzing papers..."):
                os.environ["GOOGLE_API_KEY"] = api_key
                text = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = splitter.split_text(text)
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_texts(chunks, embeddings)
                st.session_state.vector_store = vector_store
                st.success("Analysis Complete!")
        else:
            st.warning("Please enter your API Key and upload PDFs.")

# --- 4. MAIN INTERFACE ---
user_query = st.text_input("Enter your clinical research question:")

if user_query and "vector_store" in st.session_state:
    docs = st.session_state.vector_store.similarity_search(user_query)
    
    prompt = PromptTemplate(
        template="Synthesize an answer using the context: {context}\nQuestion: {question}",
        input_variables=["context", "question"]
    )
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    response = chain({"input_documents": docs, "question": user_query}, return_only_outputs=True)
    st.markdown("### 📊 Synthesis Report")
    st.write(response["output_text"])
