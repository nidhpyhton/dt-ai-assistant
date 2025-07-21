import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ------------------------ CONFIG ---------------------------------
VECTOR_STORE_DIR = "../data/vector_store"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
SIMILARITY_THRESHOLD = 0.3
MAX_CHUNKS = 10
# ------------------------------------------------------------------

def load_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)

def retrieve_relevant_chunks(query, vector_store):
    results = vector_store.similarity_search_with_score(query, k=MAX_CHUNKS)

    filtered_docs = []
    for doc, score in results:
        if score > SIMILARITY_THRESHOLD:
            filtered_docs.append(doc)

    if not filtered_docs:
        print("⚠️ No chunks passed the similarity threshold. Using top 2 fallback.")
        filtered_docs = [doc for doc, _ in results[:2]]

    return filtered_docs

def build_context_with_metadata(chunks):
    """Add page metadata to each chunk like [Page 12] ..."""
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

if __name__ == "__main__":
    query = input("💬 Enter your query: ")

    print("🔄 Loading vector store and processing...")
    vector_store = load_vectorstore()

    answer, sources = answer_query(query, vector_store)

    print("\n🧠 Answer:\n", answer)

    print("\n📚 Source Pages Used:")
    pages_used = set(doc.metadata.get("start_page") for doc in sources)
    print("Pages retrieved by retriever:", sorted(pages_used))


