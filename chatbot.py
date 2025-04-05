import streamlit as st
import os
import logging
import sys
from dotenv import load_dotenv

st.set_page_config(page_title="Hệ thống RAG với Gemini", layout="wide")
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


@st.cache_resource
def initialize_models():
    print("--- Khởi tạo (hoặc lấy từ cache) mô hình LLM và Embedding ---")
    try:
        Settings.llm = Gemini(
            api_key=GEMINI_API_KEY, model_name="models/gemini-1.5-pro-latest"
        )
        Settings.embed_model = GeminiEmbedding(
            api_key=GEMINI_API_KEY, model_name="models/embedding-001"
        )
        print("--- Đã khởi tạo/lấy mô hình ---")
        return True
    except Exception as e:
        st.error(
            f"Lỗi khi khởi tạo model Gemini: {e}. Kiểm tra API Key và kết nối mạng."
        )
        return False


@st.cache_resource
def load_and_index_data(
    data_dir="data",
    persist_dir_vector="./storage_vector",
    persist_dir_summary="./storage_summary",
):
    vector_index = None
    summary_index = None
    print(f"--- Bắt đầu quá trình tải/tạo index (cache resource) ---")
    vector_storage_exists = os.path.exists(persist_dir_vector)
    summary_storage_exists = os.path.exists(persist_dir_summary)
    if vector_storage_exists:
        try:
            print(f"--- Tải Vector Index từ {persist_dir_vector} (cache) ---")
            storage_context_vector = StorageContext.from_defaults(
                persist_dir=persist_dir_vector
            )
            vector_index = load_index_from_storage(storage_context_vector)
            print("--- Đã tải Vector Index ---")
        except Exception as e:
            print(f"Lỗi khi tải Vector Index từ storage: {e}. Sẽ thử tạo lại.")
            vector_storage_exists = False
            vector_index = None
    if summary_storage_exists:
        try:
            print(f"--- Tải Summary Index từ {persist_dir_summary} (cache) ---")
            storage_context_summary = StorageContext.from_defaults(
                persist_dir=persist_dir_summary
            )
            summary_index = load_index_from_storage(storage_context_summary)
            print("--- Đã tải Summary Index ---")
        except Exception as e:
            print(f"Lỗi khi tải Summary Index từ storage: {e}. Sẽ thử tạo lại.")
            summary_storage_exists = False
            summary_index = None
    if vector_index is None or summary_index is None:
        print(
            "--- Index chưa tồn tại, không đầy đủ hoặc tải lỗi -> Tải dữ liệu và tạo lại index ---"
        )
        if not os.path.exists(data_dir):
            st.error(
                f"Thư mục dữ liệu '{data_dir}' không tồn tại. Vui lòng tạo và đặt file vào."
            )
            st.stop()
        with st.spinner(
            f"Đang tải và xử lý tài liệu từ '{data_dir}'... Việc này có thể mất vài phút lần đầu."
        ):
            try:
                print(f"--- Đang đọc tài liệu từ: {data_dir} ---")
                reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
                documents = reader.load_data()
                if not documents:
                    st.warning(
                        f"Không tìm thấy tài liệu nào trong '{data_dir}'. Hệ thống sẽ không có dữ liệu để truy vấn."
                    )
                    return None, None
                print(
                    f"--- Đã tải {len(documents)} tài liệu. Bắt đầu thêm metadata và parse nodes ---"
                )
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
                print(f"--- Đã tạo {len(nodes)} node. Bắt đầu tạo index ---")
                if vector_index is None:
                    print("--- Tạo Vector Index mới ---")
                    vector_index = VectorStoreIndex(nodes, show_progress=False)
                    print("--- Lưu trữ Vector Index vào: {persist_dir_vector} ---")
                    vector_index.storage_context.persist(persist_dir=persist_dir_vector)
                if summary_index is None:
                    print("--- Tạo Summary Index mới ---")
                    summary_index = SummaryIndex(nodes, show_progress=False)
                    print("--- Lưu trữ Summary Index vào: {persist_dir_summary} ---")
                    summary_index.storage_context.persist(
                        persist_dir=persist_dir_summary
                    )
            except Exception as e:
                st.error(f"Lỗi nghiêm trọng khi tải hoặc tạo index: {e}")
                st.stop()
    print("--- Kết thúc quá trình tải/tạo index ---")
    return vector_index, summary_index


@st.cache_resource
def get_query_engine(_vector_index, _summary_index):
    print("--- Tạo (hoặc lấy từ cache) các Query Engine và Router ---")
    if _vector_index is None or _summary_index is None:
        st.warning("Index không hợp lệ, không thể tạo Query Engine.")
        return None
    try:
        vector_query_engine = _vector_index.as_query_engine(similarity_top_k=3)
        summary_query_engine = _summary_index.as_query_engine(
            response_mode="tree_summarize", use_async=True
        )
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            name="vector_search_tool",
            description=(
                "Hữu ích để tìm kiếm và truy xuất thông tin cụ thể, chi tiết, hoặc trả lời câu hỏi trực tiếp từ nội dung các tài liệu (PDF, TXT, DOCX)."
            ),
        )
        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            name="summary_tool",
            description=(
                "Hữu ích để tóm tắt nội dung chính của toàn bộ hoặc một phần lớn các tài liệu."
            ),
        )
        query_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(llm=Settings.llm),
            query_engine_tools=[vector_tool, summary_tool],
            verbose=False,
        )
        print("--- Đã tạo/lấy Router Query Engine ---")
        return query_engine
    except Exception as e:
        st.error(f"Lỗi khi tạo Query Engine hoặc Router: {e}")
        return None


print("--- Bắt đầu khởi tạo ứng dụng Streamlit ---")
models_initialized = initialize_models()
if models_initialized:
    vector_index, summary_index = load_and_index_data()
    if vector_index is None or summary_index is None:
        st.warning(
            "Không thể tải hoặc tạo index. Hệ thống có thể không hoạt động đúng."
        )
        query_engine = None
    else:
        query_engine = get_query_engine(vector_index, summary_index)
else:
    st.error(
        "Không thể khởi tạo mô hình LLM/Embedding. Vui lòng kiểm tra lỗi và thử lại."
    )
    vector_index = None
    summary_index = None
    query_engine = None
print("--- Hoàn tất khởi tạo ứng dụng (trước phần UI) ---")
st.title("📚 Hệ thống Hỏi Đáp Tài Liệu sử dụng Gemini & LlamaIndex")
st.caption("Đặt câu hỏi về nội dung các file trong thư mục 'data' (PDF, TXT, DOCX)")
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Chào bạn! Bạn muốn hỏi gì về các tài liệu?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    if not models_initialized:
        st.error(
            "Hệ thống chưa sẵn sàng do lỗi khởi tạo model. Vui lòng kiểm tra log và cấu hình."
        )
        st.stop()
    if not query_engine:
        st.error(
            "Hệ thống chưa sẵn sàng do lỗi tải/tạo index hoặc query engine. Vui lòng kiểm tra log."
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
                    st.markdown(f"- **Độ liên quan (Score):** `{relevance_score:.4f}`")
                    st.divider()
    except Exception as e:
        error_message = f"Rất tiếc, đã xảy ra lỗi trong quá trình truy vấn: {e}"
        print(f"Query Error: {e}")
        response_placeholder.error(error_message)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_message}
        )
