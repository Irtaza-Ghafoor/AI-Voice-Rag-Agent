# 🎙️ AI Voice-Enabled RAG Agent

A production-ready, voice-enabled Retrieval-Augmented Generation (RAG) assistant built using Groq, LangChain, ChromaDB, and Gradio. The pipeline accepts voice queries, transcribes them using phonetic-engineered Speech-to-Text (STT), queries a PDF document vector base, and responds with both text and synthesized audio.

---

## 🚀 Features

* **Phonetic STT Prompting:** Engineered Groq Whisper-large-v3-turbo pipeline optimized to transcribe domain-specific named entities accurately (e.g., handles phonetic errors for names like "Irtaza").
* **Vector Knowledge Base:** Chunked document ingestion using `RecursiveCharacterTextSplitter` stored in `ChromaDB` with `all-MiniLM-L6-v2` embeddings.
* **Low-Latency LLM:** Uses Groq's `llama3-8b-8192` model via LangChain for contextual answers.
* **Voice Output:** Synthesizes LLM responses into audio output using `gTTS`.
* **Interactive UI:** Clean Gradio interface with direct microphone input and automatic response playback.

---

## 📁 Repository Structure

```text
ai-voice-rag-agent/
├── app.py                     # Main Gradio & RAG Execution File
├── requirements.txt           # Project Dependencies
├── README.md                  # Project Documentation
└── data/
    └── Irtaza Ahmed - ML Dev.pdf  # Target Knowledge Base Document