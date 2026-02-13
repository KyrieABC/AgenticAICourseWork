import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. SETUP & DATA INGESTION
docs = [
    {"id": "doc1", "text": "The company's revenue grew by 20% in 2023 due to AI integration."},
    {"id": "doc2", "text": "New health policies were implemented in Q3, focusing on mental wellness."},
    {"id": "doc3", "text": "AI integration has been a key driver for tech sector growth this year."}
]
texts = [d["text"] for d in docs]
metadatas = [{"source": d["id"]} for d in docs]

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# 2. HYBRID RETRIEVAL SETUP
# Combines Semantic (Vector) + Keyword (BM25)
bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
bm25_retriever.k = 2

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], 
    weights=[0.4, 0.6]
)

# 3. RE-RANKING SETUP
# Uses a Cross-Encoder to re-evaluate the top results
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
compressor = CrossEncoderReranker(model=model, top_n=2)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)

# 4. QUERY REWRITE LOGIC
rewrite_prompt = ChatPromptTemplate.from_template(
    "Rewrite the following user question to be an optimized search query for a vector database. "
    "Maintain the original intent. Question: {question}"
)
rewriter = rewrite_prompt | ChatOpenAI(temperature=0) | StrOutputParser()

# 5. GENERATION WITH CITATIONS
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

qa_prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the provided context. 
Every sentence in your answer MUST cite the source ID (e.g., [doc1]).

Context:
{context}

Question: {question}
""")

def format_docs_with_id(docs):
    return "\n\n".join([f"Source: {d.metadata['source']}\nContent: {d.page_content}" for d in docs])

# 6. THE FULL PIPELINE
def run_rag_pipeline(user_query):
    # Step A: Rewrite
    optimized_query = rewriter.invoke({"question": user_query})
    print(f"--- Optimized Query: {optimized_query} ---")
    
    # Step B: Hybrid Retrieve & Re-rank
    retrieved_docs = compression_retriever.get_relevant_documents(optimized_query)
    
    # Step C: Generate with Citations
    chain = (
        {"context": lambda x: format_docs_with_id(x["docs"]), "question": lambda x: x["question"]}
        | qa_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain.invoke({"docs": retrieved_docs, "question": user_query})

# Execution
response = run_rag_pipeline("How much did revenue grow and why?")
print(f"\nFinal Answer:\n{response}")
