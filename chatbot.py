import streamlit as st
import os
import logging
import sys
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

st.set_page_config(page_title="Hệ thống RAG với Gemini & ChromaDB", layout="wide")
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error(
        "API key của Gemini chưa được đặt. Hãy tạo file .env và thêm GEMINI_API_KEY='YOUR_API_KEY'"
    )
    st.stop()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.vector_stores.chroma import ChromaVectorStore


@st.cache_resource
def initialize_models():
    print("--- Khởi tạo (hoặc lấy từ cache) mô hình LLM và Embedding ---")
    try:
        Settings.llm = Gemini(
            api_key=GEMINI_API_KEY, model_name="models/gemini-2.0-flash"
        )
        Settings.embed_model = GeminiEmbedding(
            api_key=GEMINI_API_KEY, model_name="models/embedding-001"
        )
        print("--- Đã khởi tạo/lấy mô hình ---")
        return True
    except Exception as e:
        st.error(
            f"Lỗi khi khởi tạo model Gemini: {e}. Kiểm tra API Key, tên model và kết nối mạng."
        )
        return False


@st.cache_resource
def load_and_index_data(
    data_dir="data",
    chroma_persist_dir="./chroma_db",
    chroma_collection_name="rag_gemini_collection",
    summary_persist_dir="./storage_summary",
):
    vector_index = None
    summary_index = None
    nodes = []
    print(f"--- Bắt đầu quá trình tải/tạo index (ChromaDB & Summary Index) ---")
    print(f"--- Khởi tạo ChromaDB client tại: {chroma_persist_dir} ---")
    try:
        db = chromadb.PersistentClient(path=chroma_persist_dir)
        print(f"--- Lấy hoặc tạo Chroma collection: {chroma_collection_name} ---")
        chroma_collection = db.get_or_create_collection(chroma_collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    except Exception as e:
        st.error(
            f"Lỗi khi khởi tạo hoặc kết nối tới ChromaDB tại '{chroma_persist_dir}': {e}"
        )
        st.stop()
    collection_exists_and_has_data = False
    if os.path.exists(chroma_persist_dir) and chroma_collection.count() > 0:
        print(
            f"--- Collection '{chroma_collection_name}' đã tồn tại và có {chroma_collection.count()} documents. Đang tải Vector Index từ Chroma. ---"
        )
        collection_exists_and_has_data = True
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=Settings.embed_model
        )
    else:
        print(
            f"--- Collection '{chroma_collection_name}' chưa có dữ liệu hoặc thư mục chưa tồn tại. Sẽ xử lý tài liệu và tạo index mới. ---"
        )
    if not collection_exists_and_has_data:
        print("--- Tải dữ liệu và tạo Vector Index mới vào ChromaDB ---")
        if not os.path.exists(data_dir):
            st.error(f"Thư mục dữ liệu '{data_dir}' không tồn tại.")
            st.stop()
        with st.spinner(
            f"Đang tải, xử lý tài liệu từ '{data_dir}' và tạo index vào ChromaDB..."
        ):
            try:
                print(f"--- Đang đọc tài liệu từ: {data_dir} ---")
                reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
                documents = reader.load_data()
                if not documents:
                    st.warning(f"Không tìm thấy tài liệu trong '{data_dir}'.")
                    return None, None
                print(f"--- Đã tải {len(documents)} tài liệu. Parse nodes... ---")
                for doc in documents:
                    file_path = doc.metadata.get("file_path")
                    if file_path:
                        file_extension = os.path.splitext(file_path)[1].lower()
                        doc.metadata["file_type"] = file_extension.replace(".", "")
                    else:
                        doc.metadata["file_type"] = "unknown"
                node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
                nodes = node_parser.get_nodes_from_documents(
                    documents, show_progress=False
                )
                print(
                    f"--- Đã tạo {len(nodes)} node. Bắt đầu tạo Vector Index vào Chroma ---"
                )
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )
                vector_index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    embed_model=Settings.embed_model,
                    show_progress=True,
                )
                print(
                    f"--- Đã tạo và lưu Vector Index vào Chroma collection: {chroma_collection_name} ---"
                )
            except Exception as e:
                st.error(
                    f"Lỗi nghiêm trọng khi tải tài liệu hoặc tạo Vector Index vào Chroma: {e}"
                )
                st.stop()
    if not nodes:
        if not os.path.exists(data_dir):
            st.warning(
                f"Không tìm thấy thư mục data '{data_dir}' để tạo Summary Index nếu cần."
            )
        else:
            print(
                f"--- Đọc lại tài liệu để chuẩn bị cho Summary Index (nếu cần tạo mới) ---"
            )
            try:
                reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
                documents = reader.load_data()
                if documents:
                    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
                    nodes = node_parser.get_nodes_from_documents(
                        documents, show_progress=False
                    )
                else:
                    print("--- Không tìm thấy tài liệu để tạo Summary Index ---")
            except Exception as e:
                print(f"Lỗi khi đọc lại tài liệu cho Summary Index: {e}")
    summary_storage_exists = os.path.exists(summary_persist_dir)
    if summary_storage_exists:
        try:
            print(f"--- Tải Summary Index từ {summary_persist_dir} (cache) ---")
            storage_context_summary = StorageContext.from_defaults(
                persist_dir=summary_persist_dir
            )
            summary_index = load_index_from_storage(storage_context_summary)
            print("--- Đã tải Summary Index ---")
        except Exception as e:
            print(f"Lỗi khi tải Summary Index từ storage: {e}. Sẽ thử tạo lại.")
            summary_storage_exists = False
            summary_index = None
    if not summary_storage_exists and nodes:
        print("--- Tạo Summary Index mới ---")
        try:
            summary_storage_context = StorageContext.from_defaults()
            summary_index = SummaryIndex(
                nodes, storage_context=summary_storage_context, show_progress=True
            )
            print(f"--- Lưu trữ Summary Index vào: {summary_persist_dir} ---")
            summary_storage_context.persist(persist_dir=summary_persist_dir)
        except Exception as e:
            st.error(f"Lỗi khi tạo hoặc lưu Summary Index: {e}")
            summary_index = None
    elif not nodes:
        print("--- Không có nodes để tạo Summary Index mới. ---")
        summary_index = None
    print("--- Kết thúc quá trình tải/tạo index (Chroma & Summary) ---")
    return vector_index, summary_index


@st.cache_resource
def get_query_engine(_vector_index, _summary_index):
    print("--- Tạo (hoặc lấy từ cache) các Query Engine và Router ---")
    query_engine = None
    has_vector_engine = _vector_index is not None
    has_summary_engine = _summary_index is not None
    if not has_vector_engine and not has_summary_engine:
        st.warning(
            "Không có Vector Index hoặc Summary Index hợp lệ, không thể tạo Query Engine."
        )
        return None
    query_engine_tools = []
    try:
        if has_vector_engine:
            print("--- Tạo Vector Query Engine ---")
            vector_query_engine = _vector_index.as_query_engine(similarity_top_k=3)
            vector_tool = QueryEngineTool.from_defaults(
                query_engine=vector_query_engine,
                name="vector_search_tool",
                description="Hữu ích để tìm kiếm và truy xuất thông tin cụ thể, chi tiết, hoặc trả lời câu hỏi trực tiếp từ nội dung các tài liệu (PDF, TXT, DOCX).",
            )
            query_engine_tools.append(vector_tool)
        else:
            print(
                "--- Bỏ qua việc tạo Vector Query Engine do không có Vector Index ---"
            )
        if has_summary_engine:
            print("--- Tạo Summary Query Engine ---")
            summary_query_engine = _summary_index.as_query_engine(
                response_mode="tree_summarize", use_async=True
            )
            summary_tool = QueryEngineTool.from_defaults(
                query_engine=summary_query_engine,
                name="summary_tool",
                description="Hữu ích để tóm tắt nội dung chính của toàn bộ hoặc một phần lớn các tài liệu.",
            )
            query_engine_tools.append(summary_tool)
        else:
            print(
                "--- Bỏ qua việc tạo Summary Query Engine do không có Summary Index ---"
            )
        if query_engine_tools:
            print("--- Tạo Router Query Engine ---")
            query_engine = RouterQueryEngine(
                selector=LLMSingleSelector.from_defaults(llm=Settings.llm),
                query_engine_tools=query_engine_tools,
                verbose=False,
            )
            print("--- Đã tạo/lấy Router Query Engine ---")
        else:
            st.error("Không có Query Engine nào được tạo thành công.")
            return None
    except Exception as e:
        st.error(f"Lỗi khi tạo Query Engine hoặc Router: {e}")
        return None
    return query_engine


print("--- Bắt đầu khởi tạo ứng dụng Streamlit với ChromaDB ---")
models_initialized = initialize_models()
if models_initialized:
    vector_index, summary_index = load_and_index_data()
    if vector_index is None and summary_index is None:
        st.warning(
            "Không thể tải hoặc tạo bất kỳ index nào (Vector hoặc Summary). Hệ thống có thể không hoạt động."
        )
        query_engine = None
    else:
        query_engine = get_query_engine(vector_index, summary_index)
else:
    st.error(
        "Không thể khởi tạo mô hình LLM/Embedding. Vui lòng kiểm tra lỗi và thử lại."
    )
    query_engine = None
print("--- Hoàn tất khởi tạo ứng dụng (trước phần UI) ---")
st.title("📚 Hệ thống Hỏi Đáp Tài Liệu (Gemini + LlamaIndex + ChromaDB)")
st.caption("Đặt câu hỏi về nội dung các file trong thư mục 'data' (PDF, TXT, DOCX)")
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chào bạn! Bạn muốn hỏi gì về các tài liệu?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    if not models_initialized:
        st.error("Hệ thống chưa sẵn sàng do lỗi khởi tạo model.")
        st.stop()
    if not query_engine:
        st.error(
            "Hệ thống chưa sẵn sàng do không có Query Engine nào hoạt động. Kiểm tra log lỗi index."
        )
        st.stop()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response_placeholder = st.chat_message("assistant").empty()
    try:
        with st.spinner("Đang tìm câu trả lời..."):
            response = query_engine.query(prompt)
            response_text = str(response)
        response_placeholder.write(response_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )
        if (
            response_text
            and hasattr(response, "source_nodes")
            and response.source_nodes
        ):
            with st.expander("Nguồn tài liệu tham khảo"):
                for i, snode in enumerate(response.source_nodes):
                    metadata = snode.metadata
                    file_name = metadata.get("file_path", "N/A")
                    file_type = metadata.get("file_type", "N/A")
                    relevance_score = snode.score
                    display_name = os.path.basename(file_name)
                    st.markdown(f"**Nguồn {i+1}:**")
                    st.markdown(f"- **File:** `{display_name}` ({file_type})")
                    if relevance_score is not None:
                        st.markdown(
                            f"- **Độ liên quan (Score):** `{relevance_score:.4f}`"
                        )
                    else:
                        st.markdown(f"- **Độ liên quan (Score):** `N/A`")
                    st.divider()
    except Exception as e:
        error_message = f"Rất tiếc, đã xảy ra lỗi trong quá trình truy vấn: {e}"
        print(f"Query Error Type: {type(e).__name__}, Message: {e}")
        import traceback

        print(traceback.format_exc())
        response_placeholder.error(error_message)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_message}
        )
