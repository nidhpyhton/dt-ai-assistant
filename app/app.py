import streamlit as st
from dotenv import load_dotenv
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from pathlib import Path


# ------------------ Load Environment ------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ------------------ Config ------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = ROOT_DIR / "data" / "vector_store"
# adjust if needed
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4"
SIMILARITY_THRESHOLD = 0.5
MAX_CHUNKS = 10

# ------------------ Streamlit Setup ------------------
st.set_page_config(page_title="DT AI Assistant – Capital Gains", page_icon="📘")
st.title("📘 DT AI Assistant")
st.markdown("Ask me anything from the **Capital Gains** chapter of Direct Tax.")

# ------------------ Vector Store Loader ------------------
@st.cache_resource(show_spinner="📚 Loading study material...")
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)

vector_store = load_vectorstore()

# ------------------ Helper Functions ------------------

def retrieve_relevant_chunks(query, vector_store):
    results = vector_store.similarity_search_with_score(query, k=MAX_CHUNKS)
    filtered_docs = [doc for doc, score in results if score > SIMILARITY_THRESHOLD]
    return filtered_docs or [doc for doc, _ in results[:2]]  # fallback

def build_context_with_metadata(chunks):
    context_blocks = []
    for doc in chunks:
        page = doc.metadata.get("start_page", "NA")
        tagged_text = f"[Page {page}] {doc.page_content.strip()}"
        context_blocks.append(tagged_text)
    return "\n\n".join(context_blocks)

def answer_query(query, vector_store):
    relevant_chunks = retrieve_relevant_chunks(query, vector_store)
    context = build_context_with_metadata(relevant_chunks)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a qualified tax expert. Use the following context to answer the user's question.
Each passage in the context is tagged with a page number (e.g., [Page 12]).

Only use the information from the context. If the answer is not found, say "Answer not found in the material."

At the end of your answer, include a line:
📄 Pages Referenced: X, Y, Z

Context:
{context}

Question: {question}

Answer:
"""
    )

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"context": context, "question": query})
    return response, relevant_chunks

# ------------------ Streamlit UI ------------------

# Initialize session state variables
if "query_history" not in st.session_state:
    st.session_state.query_history = []

st.markdown("----")

# UI input
query = st.text_input("💬 Enter your question about Capital Gains", key="main_query")

if query:
    with st.spinner("🔎 Thinking..."):
        answer, sources = answer_query(query, vector_store)

    # Save history
    st.session_state.query_history.append({
        "question": query,
        "answer": answer,
        "pages": sorted(set(doc.metadata.get("start_page", "NA") for doc in sources))
    })

# Display all Q&A history
for idx, qa in enumerate(reversed(st.session_state.query_history), 1):
    st.markdown(f"### ❓ Q{len(st.session_state.query_history) - idx + 1}: {qa['question']}")
    st.write(qa["answer"])
    with st.expander("📄 Pages used"):
        st.write(qa["pages"])
    st.markdown("---")
