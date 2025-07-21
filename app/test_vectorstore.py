# test_vectorstore.py
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(r"../data/vector_store", embeddings, allow_dangerous_deserialization=True)


query = input("🔎 Enter your query: ")
results_with_scores = vector_store.similarity_search_with_score(query, k=20)

similarity_threshold = 0.5

for doc, score in results_with_scores:
    print(score)
    print("\nResult:")
    print(doc.page_content[:400])
    print("Metadata:", doc.metadata)

