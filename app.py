import os
import gradio as gr
from groq import Groq
from gtts import gTTS

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETUP VECTOR DB & RAG CHAIN ---
pdf_path = "data/Irtaza Ahmed - ML Dev.pdf"
loader = PyPDFLoader(pdf_path)
raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pdf_chunks = text_splitter.split_documents(raw_documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=pdf_chunks,
    embedding=embeddings,
    collection_name="pdf_knowledge_base"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

template = """You are a helpful AI Assistant analyzing a document/resume.

Note: The user's query comes from Speech-to-Text (STT) and may contain phonetic spelling errors or misheard names (e.g., 'Irtura' or 'Artaza' actually means 'Irtaza'). Use context clues from the document to match names correctly.

Context:
{context}

Question: {question}

Answer concisely:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatGroq(
    model_name="llama3-8b-8192",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 2. GRADIO VOICE PIPELINE ---
def gradio_voice_pipeline(user_audio_path):
    if user_audio_path is None:
        return "Please record or upload an audio clip first.", None

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    with open(user_audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(user_audio_path, file.read()),
            model="whisper-large-v3-turbo",
            prompt="Irtaza Ahmed, Machine Learning Developer, email, Lahore",
            response_format="text"
        )
    
    transcribed_text = transcription.strip()
    rag_response_text = rag_chain.invoke(transcribed_text)
    
    output_audio_path = "gradio_response.mp3"
    tts = gTTS(text=rag_response_text, lang="en", slow=False)
    tts.save(output_audio_path)
    
    display_output = f"🗣️ Transcribed Query: '{transcribed_text}'\n\n🤖 AI Answer:\n{rag_response_text}"
    return display_output, output_audio_path

app = gr.Interface(
    fn=gradio_voice_pipeline,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="Speak Your Question"),
    outputs=[
        gr.Textbox(label="Agent Processing & Text Response", lines=6),
        gr.Audio(label="Voice Output Response", autoplay=True)
    ],
    title="🎙️ AI Voice & RAG Knowledge Assistant",
    description="Record your audio query. The app transcribes it using Groq Whisper, queries the vector database using RAG, and responds with synthesized speech.",
    theme="soft"
)

if __name__ == "__main__":
    app.launch()
