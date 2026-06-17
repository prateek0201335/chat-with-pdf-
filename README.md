# 🦜️🔗 AI Multi-Source RAG Chatbot Dashboard

A premium, unified Streamlit application that allows you to upload and chat with multiple data sources (PDFs, CSV/Excel spreadsheets, and URLs) using LangChain, OpenAI, and OpenRouter.

---

## 🚀 Features

1. **📄 PDF Insight Chatbot & Translator**:
   - Upload any PDF document.
   - Text parsing and chunking using `RecursiveCharacterTextSplitter`.
   - Embeddings and indexing stored locally using `FAISS` (avoids duplicate calculation fees).
   - In-app multi-language translation panel supporting 15+ target languages.

2. **📈 CSV/Excel Data Agent & Auto-Visualizer**:
   - Upload CSV or Excel files.
   - Styled dataset metrics (rows, columns) and descriptive statistics.
   - Interactive column details viewer.
   - Automatic Altair chart generation (Bar, Line, Scatter) based on numeric column types.
   - Advanced natural language AI agent that writes and executes safe Python code locally.

3. **🌐 Web Page AI Reader**:
   - Enter any public webpage URL.
   - Crawl and extract the core content recursively.
   - Interact with the website's text using conversational RAG chains.

---

## 🛠️ Files to Push to Git

When committing to GitHub, commit **only** these files to keep your credentials and build assets safe:

- `app.py` (The main application code)
- `requirements.txt` (Python dependencies list)
- `.env.example` (Template for environment variables)
- `.gitignore` (Configured to ignore `.env`, virtual environments, and index caches)
- `README.md` (This documentation file)

*Note: The `.env` file containing your secret API key and the `.faiss_cache/` directory containing your index embeddings should **never** be committed to Git. The `.gitignore` is already set up to exclude them.*

---

## 💻 Getting Started

### 1. Clone the Repository
```bash
git clone <your-repo-link>
cd <repo-name>
```

### 2. Configure Environment Variables
Create a `.env` file from the provided template:
```bash
cp .env.example .env
```
Open `.env` and fill in your OpenAI or OpenRouter key:
```env
OPENAI_API_KEY=your_actual_key_here
```

### 3. Setup Virtual Environment & Install Dependencies
```bash
# Create venv
python3 -m venv venv

# Activate venv (Mac/Linux)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run app.py
```

