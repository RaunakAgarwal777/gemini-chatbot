import streamlit as st
import textwrap
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# --- Configure Gemini ---
api_key = st.secrets["GEMINI"]["API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="GEMINI CHATBOT", page_icon="🔍", layout="centered")
st.title("RETRIEVAL BASED CHATBOT WITH GEMINI")

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "chunks_vectors" not in st.session_state:
    st.session_state.chunks_vectors = None

# --- Utility Functions ---
def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, width=chunk_size)

def extract_text_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_txt(uploaded_txt):
    return uploaded_txt.read().decode("utf-8")

def process_file(uploaded_file, file_type):
    if file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_txt(uploaded_file)

    chunks = chunk_text(text)
    vectorizer = TfidfVectorizer().fit(chunks)
    chunk_vectors = vectorizer.transform(chunks)

    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.chunks_vectors = chunk_vectors

def get_relevant_chunks(query):
    vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(vec, st.session_state.chunks_vectors).flatten()
    top_indices = similarities.argsort()[-3:][::-1]
    return "\n\n".join([st.session_state.chunks[i] for i in top_indices])

def build_prompt(query):
    chat_history = "\n".join(
        [f"User: {q}\nBot: {a}" for q, a in st.session_state.history[-3:]]
    )
    context = get_relevant_chunks(query)
    prompt = f"""You are a helpful assistant. Use the following context to answer the user's question.

Context:
{context}
Chat History:
{chat_history}
User: {query}
Bot:"""
    return prompt

def save_chat_history():
    with open("chat_history.txt", "w", encoding="utf-8") as f:
        for user, bot in st.session_state.history:
            f.write(f"User: {user}\nBot: {bot}\n\n")

# --- Upload Section ---
st.sidebar.title("Upload PDF or TXT")
uploaded_file = st.sidebar.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])

if uploaded_file:
    file_type = uploaded_file.type.split("/")[-1]
    process_file(uploaded_file, file_type)
    st.sidebar.success("File processed successfully!")

# --- Ask a Question Automatically on Enter ---
st.subheader("Ask a question:")
user_query = st.text_input("Your question:", key="user_input")

if user_query and uploaded_file:
    prompt = build_prompt(user_query)
    try:
        response = model.generate_content(prompt)
        bot_reply = response.text.strip()
        st.session_state.history.append((user_query, bot_reply))
        save_chat_history()
        st.rerun()  # ✅ updated for new Streamlit versions
    except Exception as e:
        st.error(f"Error generating response: {e}")

elif user_query and not uploaded_file:
    st.warning("⚠️ Please upload a PDF or TXT file first.")

# --- Display Chat History ---
if st.session_state.history:
    st.subheader("Conversation")
    for user, bot in reversed(st.session_state.history):
        st.markdown(f"**User:** {user}")
        st.markdown(f"**Bot:** {bot}")
        st.markdown("---")

