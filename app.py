import streamlit as st
import os
import hashlib
import pickle
import pandas as pd
import requests
import matplotlib.pyplot as plt
import altair as alt
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LangChain Imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# PyPDF and Translator Imports
from pypdf import PdfReader
from deep_translator import GoogleTranslator

# Load environment variables if any
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Multi-Source RAG Chatbot Dashboard",
    page_icon="🦜",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: radial-gradient(circle at top right, rgba(99, 102, 241, 0.05), transparent 40%),
                    radial-gradient(circle at bottom left, rgba(168, 85, 247, 0.05), transparent 40%),
                    #0f172a;
        color: #f1f5f9;
    }
    
    /* Custom Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Cards and Glassmorphism */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.2);
    }
    
    /* Styled tags */
    .metric-badge {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
        color: #e0e7ff;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if "pdf_chat_history" not in st.session_state:
    st.session_state.pdf_chat_history = []

if "csv_chat_history" not in st.session_state:
    st.session_state.csv_chat_history = []

if "web_chat_history" not in st.session_state:
    st.session_state.web_chat_history = []

# ----------------- API KEY RESOLUTION -----------------
env_key = os.getenv("OPENAI_API_KEY", "")
api_key = env_key
is_env_loaded = False

if env_key and not env_key.startswith("your_openai"):
    is_env_loaded = True
else:
    # Fallback to session state
    api_key = st.session_state.get("openai_api_key", "")

# ----------------- SIDEBAR CONFIG -----------------
with st.sidebar:
    st.markdown("<h1>🦜 Multi-RAG AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8; font-size:0.9rem;'>Interact with PDFs, CSVs, Excel, and Webpages using LLMs</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # API Key Input (Silenced if in env)
    if is_env_loaded:
        st.success("🔑 API Key: Loaded from `.env`")
        with st.expander("Override API Key"):
            override_key = st.text_input("New API Key", type="password", placeholder="sk-...")
            if override_key:
                api_key = override_key
                st.session_state.openai_api_key = override_key
    else:
        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            placeholder="Paste OpenAI or OpenRouter key...",
            help="Enter your API key. This will be stored securely in session state."
        )
        if api_key_input:
            api_key = api_key_input
            st.session_state.openai_api_key = api_key_input
        
    st.markdown("---")
    
    # Model configuration
    model_choice = st.selectbox(
        "Select Model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="gpt-4o-mini is highly recommended for speed and cost efficiency."
    )
    
    st.markdown("---")
    
    # Reset Buttons
    if st.button("Clear All Chat Histories"):
        st.session_state.pdf_chat_history = []
        st.session_state.csv_chat_history = []
        st.session_state.web_chat_history = []
        st.success("Chat histories cleared!")
        
    st.markdown("<br><br><p style='text-align: center; color: #64748b; font-size: 0.8rem;'>Created by Antigravity AI</p>", unsafe_allow_html=True)

# Helper function to dynamically map OpenRouter vs OpenAI endpoint details
def resolve_model_and_url(api_key_val, model_name):
    if api_key_val.startswith("sk-or-"):
        base_url = "https://openrouter.ai/api/v1"
        if model_name == "gpt-4o-mini":
            resolved_model = "openai/gpt-4o-mini"
        elif model_name == "gpt-4o":
            resolved_model = "openai/gpt-4o"
        else:
            resolved_model = model_name
        return resolved_model, base_url
    else:
        return model_name, None

# Helper function to get LLM instance
def get_llm():
    if not api_key:
        return None
    try:
        model_name, base_url = resolve_model_and_url(api_key, model_choice)
        return ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0
        )
    except Exception as e:
        st.error(f"Error initializing ChatOpenAI: {str(e)}")
        return None

# Helper function to get Embeddings instance
def get_embeddings():
    if not api_key:
        return None
    try:
        if api_key.startswith("sk-or-"):
            return OpenAIEmbeddings(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                model="openai/text-embedding-3-small"
            )
        else:
            return OpenAIEmbeddings(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAIEmbeddings: {str(e)}")
        return None

# Check key check
if not api_key:
    st.warning("⚠️ Please configure your API Key to begin.")
    st.stop()

# Cache directory configuration
CACHE_DIR = ".faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------- TABS SETUP -----------------
tab1, tab2, tab3 = st.tabs([
    "📄 PDF Chatbot & Translator", 
    "📈 CSV/Excel Data Agent", 
    "🌐 Web Page AI Reader"
])

# ==========================================
# TAB 1: PDF CHATBOT & TRANSLATOR
# ==========================================
with tab1:
    st.subheader("📄 Chat with PDF & Auto-Translate")
    
    # Language Translation Selection list
    LANGUAGES = {
        "None (Original English)": "none",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Dutch": "nl",
        "Russian": "ru",
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "Japanese": "ja",
        "Korean": "ko",
        "Arabic": "ar",
        "Hindi": "hi",
        "Turkish": "tr",
        "Vietnamese": "vi"
    }
    
    col_pdf1, col_pdf2 = st.columns([1, 2])
    
    with col_pdf1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 1. Upload Document")
        uploaded_pdf = st.file_uploader("Upload your PDF file", type=["pdf"])
        
        target_lang = st.selectbox(
            "Translate AI responses to:",
            options=list(LANGUAGES.keys()),
            index=0
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    vectorstore_pdf = None
    if uploaded_pdf is not None:
        file_bytes = uploaded_pdf.read()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        db_path = os.path.join(CACHE_DIR, f"pdf_{file_hash}")
        
        embeddings = get_embeddings()
        
        if os.path.exists(db_path):
            with st.spinner("Loading index from cache..."):
                vectorstore_pdf = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
                st.success("Loaded cached embeddings successfully!")
        else:
            with st.spinner("Parsing PDF and generating embeddings..."):
                # Parse PDF
                pdf_reader = PdfReader(uploaded_pdf)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)
                
                # Embed and save
                vectorstore_pdf = FAISS.from_texts(chunks, embedding=embeddings)
                vectorstore_pdf.save_local(db_path)
                st.success("Embeddings created and saved to cache!")
                
        # Status Card
        with col_pdf1:
            st.markdown(f"""
            <div class='glass-card'>
                <h4>Document Insights</h4>
                <span class='metric-badge'>Name: {uploaded_pdf.name}</span>
                <span class='metric-badge'>Size: {len(file_bytes)/1024/1024:.2f} MB</span>
                <span class='metric-badge'>Pages: {len(PdfReader(uploaded_pdf).pages)}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_pdf2:
        st.markdown("### 2. Conversational Chat")
        
        # Display chat messages
        for message in st.session_state.pdf_chat_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(message.content)
                
        if vectorstore_pdf is not None:
            user_query = st.chat_input("Ask a question about the PDF...", key="pdf_query_input")
            
            if user_query:
                # Add to history
                with st.chat_message("user"):
                    st.write(user_query)
                st.session_state.pdf_chat_history.append(HumanMessage(content=user_query))
                
                with st.spinner("Thinking..."):
                    # Similarity search
                    docs = vectorstore_pdf.similarity_search(query=user_query, k=3)
                    
                    # LLM completion
                    llm = get_llm()
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=user_query)
                    
                    # Optional Translation
                    translation_code = LANGUAGES[target_lang]
                    final_response = response
                    
                    if translation_code != "none":
                        try:
                            translator = GoogleTranslator(source='auto', target=translation_code)
                            translated_response = translator.translate(response)
                            final_response = f"{response}\n\n---\n🌐 **Translated ({target_lang}):**\n{translated_response}"
                        except Exception as e:
                            final_response = f"{response}\n\n*(Translation failed: {str(e)})*"
                    
                    with st.chat_message("assistant"):
                        st.write(final_response)
                    st.session_state.pdf_chat_history.append(AIMessage(content=final_response))
                    st.rerun()
        else:
            st.info("Upload a PDF document on the left to activate the chatbot.")

# ==========================================
# TAB 2: CSV/EXCEL DATA AGENT
# ==========================================
with tab2:
    st.subheader("📈 Conversational Data Agent & Visualizer")
    
    col_csv1, col_csv2 = st.columns([1, 2])
    
    df_loaded = None
    with col_csv1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 1. Upload Dataset")
        uploaded_csv = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_csv is not None:
            try:
                if uploaded_csv.name.endswith(".csv"):
                    df_loaded = pd.read_csv(uploaded_csv)
                else:
                    df_loaded = pd.read_excel(uploaded_csv)
                st.success("Dataset loaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                
        if df_loaded is not None:
            # Metadata summary
            st.markdown(f"""
            <div class='glass-card'>
                <h4>Dataset Metrics</h4>
                <span class='metric-badge'>Rows: {df_loaded.shape[0]}</span>
                <span class='metric-badge'>Columns: {df_loaded.shape[1]}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive columns viewer
            with st.expander("Column Details & Types"):
                col_info = pd.DataFrame({
                    "Type": df_loaded.dtypes.astype(str),
                    "Nulls": df_loaded.isnull().sum()
                })
                st.dataframe(col_info, use_container_width=True)
                
            # Quick Stats
            with st.expander("Quick Summary Stats"):
                st.dataframe(df_loaded.describe(include='all').fillna('-'), use_container_width=True)

    with col_csv2:
        if df_loaded is not None:
            st.markdown("### Data Preview")
            st.dataframe(df_loaded.head(5), use_container_width=True)
            
            st.markdown("### 📊 Interactive Visualizations")
            # Auto chart generation
            num_cols = df_loaded.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df_loaded.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if num_cols:
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    x_axis = st.selectbox("X-Axis (Variable)", options=df_loaded.columns.tolist())
                with chart_col2:
                    y_axis = st.selectbox("Y-Axis (Numeric)", options=num_cols)
                    
                chart_type = st.radio("Chart Type", options=["Bar", "Line", "Scatter"], horizontal=True)
                
                try:
                    if chart_type == "Bar":
                        chart = alt.Chart(df_loaded.head(100)).mark_bar().encode(
                            x=x_axis,
                            y=y_axis,
                            tooltip=[x_axis, y_axis]
                        ).interactive()
                    elif chart_type == "Line":
                        chart = alt.Chart(df_loaded.head(100)).mark_line().encode(
                            x=x_axis,
                            y=y_axis,
                            tooltip=[x_axis, y_axis]
                        ).interactive()
                    else:
                        chart = alt.Chart(df_loaded.head(100)).mark_point().encode(
                            x=x_axis,
                            y=y_axis,
                            tooltip=[x_axis, y_axis]
                        ).interactive()
                        
                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to render chart: {str(e)}")
            else:
                st.info("No numeric columns found for auto-visualization.")
                
            st.markdown("### 💬 Chat with Data Agent")
            
            for message in st.session_state.csv_chat_history:
                role = "user" if isinstance(message, HumanMessage) else "assistant"
                with st.chat_message(role):
                    st.write(message.content)
                    
            user_csv_query = st.chat_input("Ask a question about the dataset...", key="csv_query_input")
            
            if user_csv_query:
                # Add to history
                with st.chat_message("user"):
                    st.write(user_csv_query)
                st.session_state.csv_chat_history.append(HumanMessage(content=user_csv_query))
                
                with st.spinner("Analyzing dataset..."):
                    # Custom LLM Sandbox dataframe query builder
                    llm = get_llm()
                    
                    cols = list(df_loaded.columns)
                    dtypes = df_loaded.dtypes.to_dict()
                    preview = df_loaded.head(3).to_string()
                    summary = df_loaded.describe().to_string()
                    
                    prompt = f"""
You are an expert data analyst. You are given a pandas DataFrame named `df`.
Here is the structural details of `df`:
Columns: {cols}
Types: {dtypes}
Preview:
{preview}
Summary stats:
{summary}

User question: "{user_csv_query}"

Task: Write short, efficient Python code that analyzes the dataframe `df` to answer the question.
Guidelines:
1. Respond ONLY with valid, executable Python code. Do NOT wrap it in any comments, explanations, markdown codes or triple backticks.
2. The final result should be stored in a variable named `answer` (which must be a string or a small summary dataframe or float/int).
3. If the user asks for a chart or visualization, write code to create a matplotlib figure object and save it in a variable named `fig`.
4. Ensure code is safe and executes quickly.

Write the code now:
"""
                    try:
                        code_response = llm.predict(prompt).strip()
                        # Clean if LLM included backticks
                        if code_response.startswith("```python"):
                            code_response = code_response[9:]
                        if code_response.endswith("```"):
                            code_response = code_response[:-3]
                        code_response = code_response.strip()
                        
                        # Sandbox environment
                        local_scope = {"df": df_loaded, "plt": plt, "alt": alt, "pd": pd}
                        exec(code_response, {}, local_scope)
                        
                        ans = local_scope.get("answer", "Analysis completed but no answer variable was returned.")
                        fig = local_scope.get("fig") or local_scope.get("chart")
                        
                        final_response = f"**Agent Analysis:**\n{ans}"
                        
                        with st.chat_message("assistant"):
                            st.write(final_response)
                            if fig:
                                st.pyplot(fig)
                        
                        # Save responses
                        st.session_state.csv_chat_history.append(AIMessage(content=final_response))
                    except Exception as e:
                        # Simple fallback correction
                        try:
                            correct_prompt = f"The following code failed: {str(e)}\nCode:\n{code_response}\nPlease write the corrected python code. Return only the code."
                            code_response_corr = llm.predict(correct_prompt).strip()
                            if code_response_corr.startswith("```python"):
                                code_response_corr = code_response_corr[9:]
                            if code_response_corr.endswith("```"):
                                code_response_corr = code_response_corr[:-3]
                            code_response_corr = code_response_corr.strip()
                            
                            local_scope = {"df": df_loaded, "plt": plt, "alt": alt, "pd": pd}
                            exec(code_response_corr, {}, local_scope)
                            ans = local_scope.get("answer", "Analysis completed.")
                            fig = local_scope.get("fig") or local_scope.get("chart")
                            final_response = f"**Agent Analysis:**\n{ans}"
                            with st.chat_message("assistant"):
                                st.write(final_response)
                                if fig:
                                    st.pyplot(fig)
                            st.session_state.csv_chat_history.append(AIMessage(content=final_response))
                        except Exception as e2:
                            err_msg = f"Failed to analyze data: {str(e2)}"
                            with st.chat_message("assistant"):
                                st.error(err_msg)
                            st.session_state.csv_chat_history.append(AIMessage(content=err_msg))
                    st.rerun()
        else:
            st.info("Upload a CSV or Excel dataset to explore and query it using the AI Agent.")

# ==========================================
# TAB 3: WEB PAGE AI READER
# ==========================================
with tab3:
    st.subheader("🌐 Conversational Web Reader")
    
    col_web1, col_web2 = st.columns([1, 2])
    
    vectorstore_web = None
    with col_web1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 1. Configure Website URL")
        website_url = st.text_input("Enter URL:", placeholder="https://example.com")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if website_url:
            url_hash = hashlib.md5(website_url.encode()).hexdigest()
            db_path = os.path.join(CACHE_DIR, f"web_{url_hash}")
            embeddings = get_embeddings()
            
            if os.path.exists(db_path):
                with st.spinner("Loading indexed web page from cache..."):
                    vectorstore_web = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
                    st.success("Web page loaded from cache!")
            else:
                with st.spinner("Scraping webpage and indexing contents..."):
                    try:
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                        response = requests.get(website_url, headers=headers, timeout=10)
                        
                        soup = BeautifulSoup(response.content, 'html.parser')
                        for script in soup(["script", "style"]):
                            script.decompose()
                            
                        # Extract basic metadata
                        page_title = soup.title.string if soup.title else "Untitled Page"
                        
                        text = soup.get_text(separator=' ')
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
                        
                        # Chunking
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        split_docs = text_splitter.split_text(cleaned_text)
                        
                        # Embed
                        vectorstore_web = FAISS.from_texts(split_docs, embedding=embeddings)
                        vectorstore_web.save_local(db_path)
                        st.success(f"Successfully scraped & indexed: '{page_title}'!")
                    except Exception as e:
                        st.error(f"Error loading webpage: {str(e)}")
                        
            if vectorstore_web is not None:
                st.markdown(f"""
                <div class='glass-card'>
                    <h4>Web Scraping Metrics</h4>
                    <span class='metric-badge'>URL: {website_url[:35]}...</span>
                    <span class='metric-badge'>Status: Indexed</span>
                </div>
                """, unsafe_allow_html=True)

    with col_web2:
        st.markdown("### 2. Chat with Web Content")
        
        # Web chat history display
        for message in st.session_state.web_chat_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.write(message.content)
                
        if vectorstore_web is not None:
            user_web_query = st.chat_input("Ask a question about the web page...", key="web_query_input")
            
            if user_web_query:
                # Add to history
                with st.chat_message("user"):
                    st.write(user_web_query)
                st.session_state.web_chat_history.append(HumanMessage(content=user_web_query))
                
                with st.spinner("Fetching answer..."):
                    # Similarity search
                    docs = vectorstore_web.similarity_search(query=user_web_query, k=3)
                    
                    # RAG Answer Generation
                    llm = get_llm()
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=user_web_query)
                    
                    with st.chat_message("assistant"):
                        st.write(response)
                    st.session_state.web_chat_history.append(AIMessage(content=response))
                    st.rerun()
        else:
            st.info("Input a URL on the left to start chatting with its content.")
