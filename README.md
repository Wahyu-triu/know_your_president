# Know Your President Chatbot

A web-based Streamlit application that answers questions about speeches by the President of the Republic of Indonesia using Retrieval-Augmented Generation (RAG).

## What this app does

- Loads a CSV dataset of presidential speeches (`pidato_presiden.csv`).
- Splits speech text into chunks and creates semantic embeddings with Google Gemini.
- Stores embeddings in a local FAISS vector index for fast retrieval.
- Uses retrieved speech snippets as context for Gemini to generate accurate answers.
- Displays a chat interface where users can ask questions in Indonesian.

## Key features

- **RAG-based answering**: retrieves relevant speech content before generating responses.
- **Google Gemini integration**: uses Gemini embedding and chat models.
- **Local FAISS index support**: loads an existing vector store from `faiss_index/` if available.
- **Simple Streamlit UI**: chat bubbles, source tags, and an interactive sidebar.

## Requirements

- Python
- Streamlit
- LangChain
- LangChain Google Generative AI bridge
- LangChain Community components
- FAISS CPU
- pandas
- dateparser
- uuid

## Install dependencies

```bash
pip install streamlit langchain langchain-google-genai langchain-community faiss-cpu pandas dateparser uuid
```

## Run the app

From the project folder, run:

```bash
streamlit run chatbotv2.py
```

## Usage

1. Open the Streamlit app in your browser.
2. Enter your Google Gemini API key in the sidebar.
3. Provide the path to the CSV file containing speech data.
4. Click **Muat Knowledge Base** to build or load the FAISS index.
5. Ask questions in the main chat panel.
6. Optionally clear chat history with the sidebar button.

## Files in this project

- `chatbotv2.py` - Main Streamlit app.
- `chatbot.py` - Alternative/previous version of the chatbot.
- `collect_data.py` - Data collection or preprocessing helper.
- `pidato_presiden.csv` - Speech dataset.
- `faiss_index/` - Stored FAISS vector index for faster loading.
- `rag.py` / `rag.ipynb` - RAG experiments and supporting code.

## Notes

- The app is designed for Indonesian language questions about presidential speeches.
- If `faiss_index/` exists, the app will load the saved vector store instead of rebuilding it.
- The system prompt instructs Gemini to answer politely and cite speech dates or titles when appropriate.
