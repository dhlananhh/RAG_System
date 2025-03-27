from dotenv import load_dotenv
import os

import streamlit as st

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


# =========================================================================
# 1. Cấu hình Streamlit UI
# =========================================================================

st.title("RAG System with LlamaIndex and Gemini 💬📚")
st.subheader("Demo Q&A about Big Data")

st.sidebar.header("Configuration")
document_folder_path = st.sidebar.text_input(
    "Document folder", "documents"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Notes:** Make sure you have configured the Gemini API key in your `.env` file or environment variable.
    """
)
st.sidebar.markdown("---")


# =========================================================================
# 2. Tải và xử lý nhiều loại tài liệu (PDF, TXT, DOCX)
# =========================================================================

documents_folder = os.path.join(
    os.path.dirname(__file__), document_folder_path
)
document_readers = SimpleDirectoryReader(input_dir=documents_folder)

documents = document_readers.load_data()
st.write(
    f"Uploaded **{len(documents)}** documents successfully!"
)

for doc in documents:
    if ".pdf" in doc.metadata["file_path"].lower():
        doc.metadata["file_type"] = "pdf"
    elif ".txt" in doc.metadata["file_path"].lower():
        doc.metadata["file_type"] = "txt"
    elif ".docx" in doc.metadata["file_path"].lower():
        doc.metadata["file_type"] = "docx"
    else:
        doc.metadata["file_type"] = "unknown"


# =========================================================================
# 3. Chia tài liệu thành Nodes (Chunking)
# =========================================================================

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
st.write(f"Created **{len(nodes)}** nodes successfully!")


# =========================================================================
# 4. Cấu hình LLM và Embedding Model
# =========================================================================

Settings.llm = Gemini(
    api_key=gemini_api_key, model="models/gemini-2.0-flash"
)
Settings.embed_model = GeminiEmbedding(
    api_key=gemini_api_key, model="models/embedding-001"
)
st.write(
    "Configured **LLM and Embedding Model** successfully!"
)


# =========================================================================
# 5. Tạo Index (Vector và Summary)
# =========================================================================

vector_index = VectorStoreIndex(nodes)
summary_index = SummaryIndex(nodes)
st.write(
    "Created **Vector Store Index and Summary Index** successfully!"
)


# =========================================================================
# 6. Tạo Query Engines (Summary và Vector)
# =========================================================================

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

vector_query_engine = vector_index.as_query_engine()
st.write(
    "Created **Query Engines (Summary and Vector)** successfully!"
)  # Hiển thị trên Streamlit UI


# =========================================================================
# 7. Tạo Query Engine Tools (Summary và Vector)
# =========================================================================

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarizing the main content, key points, or overall context of the document in a concise manner.",
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific details, precise information, or answering search-based questions from the document.",
)
st.write("Created **Query Engine Tools** successfully!")


# =========================================================================
# 8. Tạo Router Query Engine
# =========================================================================

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=False,
)
st.write("Created **Router Query Engine** successfully!")


# =========================================================================
# 9. Giao diện nhập câu hỏi và hiển thị kết quả trong Streamlit UI
# =========================================================================

st.markdown("---")
user_question = st.text_area(
    "**Enter your question about Big Data:**", height=100
)


if user_question:
    st.write("**Question:**", user_question)

    with st.spinner(
        "Processing question..."
    ):
        response = query_engine.query(user_question)

    st.write("**Answer:**")
    st.write(response)
    st.markdown("---")

st.success(
    "The RAG system is ready! Please ask your question."
)
