# RAG-Supported University Subject Assistant

A Retrieval-Augmented Generation (RAG) chatbot that lets students ask questions about their university course materials. Upload PDFs and PowerPoint slides, and the assistant answers using the actual lecture content — with source citations.

## Features

- Answers questions grounded in your course documents (PDFs and PPTX)
- Streaming responses via a Gradio chat interface
- Semantic chunking using an LLM (GPT-4.1-nano) for high-quality retrieval
- Query rewriting + re-ranking pipeline for more accurate results
- Conversation history awareness
- Shows source documents alongside each answer
- Supports multiple subjects simultaneously

## Supported Subjects

- Calculus
- Computer Techniques and Architecture
- Introduction to Java Programming
- Japanese Culture
- Management

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-large` |
| Vector Store | ChromaDB |
| LLM (answers) | GPT-4.1-mini (streaming) |
| LLM (chunking/reranking) | GPT-4.1-nano |
| Framework | LangChain |
| UI | Gradio |

## Project Structure

```
RagSupportedUniversitySubjectAssistant/
├── subject_documents/          # Put your PDFs and PPTX files here
│   ├── Calculus/
│   ├── Computer Techniques and Architecture/
│   ├── Introduction to Java Programming/
│   └── Japanese Culture and Management/
├── implementation/
│   ├── embeddings.py           # Document ingestion & vector store creation
│   └── query_data_streaming.py # RAG pipeline & streaming query logic
├── prototype_app_streaming.py  # Gradio UI entry point
├── .env                        # API keys (never commit this)
└── vector_db/                  # Auto-generated ChromaDB database
```

## Setup

**1. Clone the repository:**
```bash
git clone https://github.com/dombayj/RagSupportedUniversitySubjectAssistant.git
cd RagSupportedUniversitySubjectAssistant
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Create a `.env` file in the project root:**
```
OPENAI_API_KEY=your-openai-api-key-here
```

**5. Add your course documents:**

Place PDFs and/or PPTX files inside `subject_documents/`, organized by subject folder:
```
subject_documents/
└── Calculus/
    ├── lecture1.pdf
    └── slides.pptx
```

**6. Run the ingestion pipeline** (only needed once, or when you add new documents):
```bash
python -m implementation.embeddings
```

**7. Launch the app:**
```bash
python prototype_app_streaming.py
```

The app will open at `http://127.0.0.1:7860`.

## How It Works

1. **Ingestion** — Documents are loaded, semantically chunked by an LLM, and stored in a ChromaDB vector database.
2. **Query** — When a student asks a question, the query is rewritten for better retrieval, then matched against the vector store.
3. **Re-ranking** — Retrieved chunks are re-ranked by relevance using a second LLM call.
4. **Answer** — The top chunks are passed as context to GPT-4.1-mini, which streams a response back to the UI.

## Notes

- The `vector_db/` folder is auto-generated and gitignored — do not commit it.
- Never commit your `.env` file. Keep your API key private.
- Re-run `embeddings.py` whenever you add new documents.
- Set `WORKERS=1` in `embeddings.py` if you hit OpenAI rate limits during ingestion.
