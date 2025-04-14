# chatbot.py (Final Version with Sidebar Upload, Conditional RAG, and Cache Fix)

import streamlit as st
import os
import logging
import sys
import tempfile # Cần thiết để xử lý file tải lên
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai # Import để gọi trực tiếp khi không RAG
import chromadb
import traceback # Để in traceback chi tiết khi có lỗi

# --- 1. Cấu hình ban đầu và API Key ---
# Lệnh này PHẢI là lệnh Streamlit đầu tiên
st.set_page_config(page_title="Chatbot Thông Minh (Gemini + RAG)", layout="wide")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("API key của Gemini chưa được đặt. Hãy tạo file .env và thêm GEMINI_API_KEY='YOUR_API_KEY'")
    st.stop()

# Cấu hình API cho thư viện google.generativeai (dùng khi chat thông thường)
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
     st.error(f"Lỗi khi cấu hình google.generativeai với API Key: {e}. Kiểm tra lại API Key.")
     st.stop()


logging.basicConfig(stream=sys.stdout, level=logging.INFO) # INFO hoặc DEBUG

# --- 2. Import các thư viện LlamaIndex ---
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,         # Vẫn giữ Summary Index nếu muốn
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.gemini import Gemini               # LLM cho LlamaIndex
from llama_index.embeddings.gemini import GeminiEmbedding # Embedding cho LlamaIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.vector_stores.chroma import ChromaVectorStore # Vector store ChromaDB

# --- 3. Khởi tạo Mô hình (Cache Resource) ---
@st.cache_resource
def initialize_llm_and_embedding():
    """Khởi tạo và cache LLM, Embedding Model cho LlamaIndex Settings và GenAI Model."""
    print("--- Khởi tạo (hoặc lấy từ cache) mô hình LLM và Embedding ---")
    genai_chat_model_instance = None
    try:
        # LlamaIndex Settings
        Settings.llm = Gemini(api_key=GEMINI_API_KEY, model_name="models/gemini-1.5-flash-latest")
        Settings.embed_model = GeminiEmbedding(api_key=GEMINI_API_KEY, model_name="models/embedding-001")

        # GenAI Model cho chat thông thường
        genai_chat_model_instance = genai.GenerativeModel('gemini-1.5-flash-latest')

        print("--- Đã khởi tạo/lấy mô hình LlamaIndex Settings và GenAI Model---")
        return True, genai_chat_model_instance
    except Exception as e:
        st.error(f"Lỗi khi khởi tạo model Gemini: {e}. Kiểm tra API Key, tên model và kết nối mạng.")
        print(f"Model Initialization Error: {e}")
        traceback.print_exc()
        # Đảm bảo Settings được reset nếu lỗi
        Settings.llm = None
        Settings.embed_model = None
        return False, None

# --- 4. Khởi tạo ChromaDB Client (Cache Resource) ---
@st.cache_resource
def initialize_chroma_client(persist_dir="./chroma_db"):
    """Khởi tạo và cache ChromaDB client."""
    print(f"--- Khởi tạo (hoặc lấy từ cache) ChromaDB client tại: {persist_dir} ---")
    try:
        os.makedirs(persist_dir, exist_ok=True) # Đảm bảo thư mục tồn tại
        db = chromadb.PersistentClient(path=persist_dir)
        print("--- Khởi tạo ChromaDB client thành công ---")
        return db
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi khởi tạo ChromaDB client tại '{persist_dir}': {e}")
        print(f"ChromaDB Client Initialization Error: {e}")
        traceback.print_exc()
        st.stop() # Dừng nếu không thể khởi tạo DB
        return None

# --- 5. Hàm xử lý file tải lên và tạo/cập nhật Index ---
def process_uploaded_files(uploaded_files, chroma_db_client, collection_name="rag_gemini_collection"):
    """Xử lý PDF tải lên, tạo node, thêm vào ChromaDB. Trả về True nếu thành công."""
    if not uploaded_files:
        st.warning("Không có file nào được chọn để xử lý.")
        return False
    if not Settings.embed_model:
         st.error("Lỗi: Embedding model chưa được khởi tạo thành công. Không thể xử lý tài liệu.")
         return False

    # Sử dụng thư mục tạm thời để lưu file tải lên
    with tempfile.TemporaryDirectory() as temp_dir:
        st.info(f"Đang xử lý {len(uploaded_files)} file PDF...")
        print(f"--- Sử dụng thư mục tạm: {temp_dir} ---")
        try:
            # 1. Lưu và Đọc tài liệu
            print(f"--- Đang đọc tài liệu PDF từ thư mục tạm ---")
            reader = SimpleDirectoryReader(input_dir=temp_dir, input_files=[os.path.join(temp_dir, f.name) for f in uploaded_files], required_exts=[".pdf"])
            # Lưu file vào thư mục tạm TRƯỚC KHI đọc
            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                print(f"Đã lưu file tạm: {temp_file_path}")

            documents = reader.load_data(show_progress=True)
            if not documents:
                st.warning("Không đọc được nội dung từ các file PDF đã tải lên.")
                return False

            # 2. Parse thành Nodes và thêm metadata
            print(f"--- Đã tải {len(documents)} document objects. Parse nodes... ---")
            node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
            nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
            print(f"--- Đã tạo {len(nodes)} nodes ---")
            if not nodes:
                 st.warning("Không tạo được node nào từ tài liệu.")
                 return False

            # Thêm metadata (ví dụ)
            for node in nodes:
                 if 'file_path' in node.metadata:
                      node.metadata['file_type'] = 'pdf'

            # 3. Kết nối tới ChromaDB Collection
            print(f"--- Lấy hoặc tạo Chroma collection: {collection_name} ---")
            chroma_collection = chroma_db_client.get_or_create_collection(collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # 4. Tạo hoặc Cập nhật VectorStoreIndex
            print(f"--- Đang thêm {len(nodes)} nodes vào Vector Index (ChromaDB) ---")
            VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=Settings.embed_model, # Sử dụng embed model đã khởi tạo
                show_progress=True
            )
            print(f"--- Đã cập nhật thành công Vector Index vào Chroma collection: {collection_name} ({chroma_collection.count()} mục) ---")
            st.success(f"Đã xử lý và thêm nội dung từ {len(uploaded_files)} file vào cơ sở tri thức!")
            return True # Xử lý thành công

        except Exception as e:
            st.error(f"Lỗi trong quá trình xử lý file: {e}")
            print(f"File Processing Error: {e}")
            traceback.print_exc() # In traceback chi tiết vào console
            return False # Xử lý thất bại
        # Thư mục tạm sẽ tự động bị xóa

# --- 6. Hàm tải Index và tạo Query Engine (Cache Resource - ĐÃ SỬA LỖI HASH) ---
@st.cache_resource
# Thêm dấu gạch dưới (_) vào trước chroma_db_client để không hash tham số này
def load_indexes_and_get_query_engine(_chroma_db_client, collection_name="rag_gemini_collection", summary_persist_dir="./storage_summary"):
    """
    Tải Vector Index từ Chroma, Summary Index từ disk (nếu có)
    và tạo Router Query Engine.
    LƯU Ý: _chroma_db_client được bỏ qua khi hashing cho cache.
    """
    vector_index = None
    summary_index = None
    query_engine = None
    print("--- Tải (hoặc lấy từ cache) Index và tạo Query Engine ---")

    # Đảm bảo các model đã được khởi tạo
    if not Settings.llm or not Settings.embed_model:
         print("--- Lỗi: LLM hoặc Embedding model chưa được khởi tạo trong Settings. Không thể tải index hoặc tạo engine. ---")
         # Không hiển thị st.error ở đây vì hàm này chạy trong cache
         return None

    # 1. Tải Vector Index từ ChromaDB
    try:
        print(f"--- Kiểm tra Chroma collection: {collection_name} bằng client đã cung cấp ---")
        # Sử dụng _chroma_db_client (tham số với dấu gạch dưới)
        chroma_collection = _chroma_db_client.get_collection(collection_name)
        if chroma_collection.count() == 0:
             print(f"--- Collection '{collection_name}' tồn tại nhưng rỗng. Không thể tạo Vector Index. ---")
             vector_index = None
        else:
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            print(f"--- Collection '{collection_name}' có {chroma_collection.count()} documents. Đang tải Vector Index... ---")
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=Settings.embed_model # Cần embed_model để load
            )
            print("--- Đã tải Vector Index từ Chroma ---")
    except Exception as e:
        # Lỗi phổ biến là collection không tồn tại (chưa có file nào được xử lý)
        print(f"Thông báo: Không thể tải Vector Index từ Chroma Collection '{collection_name}'. Có thể nó chưa được tạo hoặc có lỗi khác. Lỗi: {e}")
        vector_index = None

    # 2. Tải Summary Index từ Disk (Tùy chọn, giữ lại nếu bạn muốn dùng)
    summary_storage_path = Path(summary_persist_dir)
    if summary_storage_path.exists() and summary_storage_path.is_dir():
        try:
            print(f"--- Tải Summary Index từ {summary_persist_dir} ---")
            summary_storage_context = StorageContext.from_defaults(persist_dir=str(summary_storage_path))
            summary_index = load_index_from_storage(summary_storage_context)
            print("--- Đã tải Summary Index ---")
        except Exception as e:
            print(f"Lỗi khi tải Summary Index từ '{summary_persist_dir}': {e}. Bỏ qua Summary Index.")
            summary_index = None
    else:
        print(f"--- Không tìm thấy thư mục Summary Index tại '{summary_persist_dir}'. Bỏ qua. ---")
        summary_index = None

    # 3. Tạo Query Engine Tools và Router
    query_engine_tools = []
    try:
        if vector_index:
            print("--- Tạo Vector Query Engine ---")
            vector_query_engine = vector_index.as_query_engine(similarity_top_k=3, llm=Settings.llm) # Truyền LLM vào engine
            vector_tool = QueryEngineTool.from_defaults(
                query_engine=vector_query_engine,
                name="vector_search_tool",
                description="Hữu ích để tìm kiếm thông tin CỤ THỂ, chi tiết, hoặc trả lời câu hỏi trực tiếp từ nội dung các tài liệu ĐÃ ĐƯỢC TẢI LÊN.",
            )
            query_engine_tools.append(vector_tool)
        else:
            print("--- Bỏ qua Vector Query Engine (không có Vector Index) ---")

        if summary_index:
            print("--- Tạo Summary Query Engine ---")
            summary_query_engine = summary_index.as_query_engine(
                response_mode="tree_summarize", use_async=True, llm=Settings.llm # Truyền LLM vào engine
            )
            summary_tool = QueryEngineTool.from_defaults(
                query_engine=summary_query_engine,
                name="summary_tool",
                description="Hữu ích để TÓM TẮT nội dung chính của tài liệu (dựa trên dữ liệu đã được xử lý trước đó cho Summary Index).",
            )
            query_engine_tools.append(summary_tool)
        else:
             print("--- Bỏ qua Summary Query Engine (không có Summary Index) ---")

        # Chỉ tạo Router nếu có ít nhất 1 tool
        if query_engine_tools:
            print(f"--- Tạo Router Query Engine với {len(query_engine_tools)} tool(s) ---")
            # Settings.llm đã được kiểm tra ở đầu hàm
            query_engine = RouterQueryEngine(
                selector=LLMSingleSelector.from_defaults(llm=Settings.llm),
                query_engine_tools=query_engine_tools,
                verbose=False # Bật True để debug router selection trong console
            )
            print("--- Đã tạo Router Query Engine ---")
            return query_engine
        else:
            print("--- Không có tool nào được tạo, không thể tạo Router Query Engine ---")
            return None

    except Exception as e:
        # Không nên hiển thị st.error ở đây vì chạy trong cache
        print(f"Lỗi khi tạo Query Engine hoặc Router: {e}")
        traceback.print_exc()
        return None

# --- 7. Hàm gọi Gemini thông thường ---
def get_general_gemini_response(user_prompt, chat_model, chat_history):
    """Gửi prompt đến Gemini API thông thường và trả về phản hồi text."""
    if not chat_model:
        print("Lỗi: Mô hình chat genai chưa được khởi tạo.")
        return "Lỗi: Mô hình chat chưa được khởi tạo."
    try:
        print("--- Gọi API Gemini thông thường ---")
        # Chuyển đổi lịch sử chat sang định dạng API yêu cầu
        api_history = []
        for msg in chat_history:
             role = 'user' if msg['role'] == 'user' else 'model'
             # Đảm bảo content là string
             content = str(msg.get('content', ''))
             api_history.append({'role': role, 'parts': [content]})

        # Bắt đầu phiên chat mới mỗi lần (hoặc sử dụng state nếu muốn)
        convo = chat_model.start_chat(history=api_history)
        response = convo.send_message(user_prompt)
        print("--- Nhận phản hồi từ Gemini API ---")
        return response.text
    except Exception as e:
        print(f"Lỗi khi gọi Gemini API: {e}")
        traceback.print_exc()
        # Cung cấp thông tin lỗi rõ ràng hơn cho người dùng
        error_detail = str(e)
        if "API key not valid" in error_detail:
             return "Lỗi: API Key không hợp lệ. Vui lòng kiểm tra lại."
        elif "quota" in error_detail.lower():
             return "Lỗi: Đã hết hạn ngạch sử dụng API Gemini. Vui lòng thử lại sau."
        else:
             return f"Xin lỗi, đã có lỗi xảy ra khi kết nối tới Gemini: {e}"


# --- 8. Khởi tạo ứng dụng Streamlit ---
print("--- === Bắt đầu Khởi tạo Ứng dụng Streamlit === ---")

# Khởi tạo các thành phần chính được cache
models_ok, genai_chat_model = initialize_llm_and_embedding()
chroma_client = initialize_chroma_client()

# Khởi tạo session state nếu chưa có
# Dùng .setdefault() để gọn hơn
st.session_state.setdefault("messages", [{"role": "assistant", "content": "Chào bạn! Tải lên tài liệu PDF ở sidebar hoặc hỏi tôi bất cứ điều gì."}])
st.session_state.setdefault("rag_initialized", False)
st.session_state.setdefault("query_engine", None)
st.session_state.setdefault("engine_loaded", False)
st.session_state.setdefault("processed_files_hash", None) # Để theo dõi file đã xử lý

# Tải query engine MỘT LẦN khi khởi tạo hoặc khi có sự thay đổi cần thiết
# (Ví dụ: sau khi xử lý file)
if models_ok and chroma_client and not st.session_state.engine_loaded:
    print("--- Tải/Làm mới Query Engine ---")
    # Truyền client đã khởi tạo vào hàm cache (với dấu gạch dưới trong định nghĩa hàm)
    st.session_state.query_engine = load_indexes_and_get_query_engine(chroma_client)
    st.session_state.engine_loaded = True # Đánh dấu đã tải xong engine lần này

    # Kiểm tra trạng thái ban đầu của RAG dựa trên ChromaDB
    try:
        collection = chroma_client.get_collection("rag_gemini_collection")
        if collection.count() > 0:
            st.session_state.rag_initialized = True
            print(f"--- Đã phát hiện RAG Index tồn tại ({collection.count()} mục). Chế độ RAG ban đầu: ON ---")
        else:
             st.session_state.rag_initialized = False
             print("--- Không phát hiện dữ liệu trong RAG Index. Chế độ RAG ban đầu: OFF ---")
    except Exception as e: # Collection có thể không tồn tại
        st.session_state.rag_initialized = False
        print(f"--- Collection RAG chưa tồn tại hoặc lỗi khi kiểm tra: {e}. Chế độ RAG ban đầu: OFF ---")

# --- 9. Giao diện Streamlit ---
st.title("📚 Chatbot Thông Minh (Gemini + LlamaIndex + ChromaDB)")

# --- Sidebar ---
with st.sidebar:
    st.title("📁 Quản lý Tài liệu")
    st.markdown("Tải lên các file PDF của bạn tại đây để chatbot trả lời dựa trên nội dung đó.")

    uploaded_files = st.file_uploader(
        "Chọn file PDF",
        accept_multiple_files=True,
        type=["pdf"],
        key="file_uploader" # Thêm key để quản lý state tốt hơn
    )

    if st.button("Xử lý Tài liệu đã tải lên", key="process_button"):
        if uploaded_files:
            # Tính hash của danh sách file để kiểm tra thay đổi
            current_files_hash = hash(tuple(sorted(f.name for f in uploaded_files)))

            # Chỉ xử lý nếu file thay đổi hoặc chưa xử lý lần nào
            # (Giúp tránh xử lý lại cùng file nhiều lần nếu không cần)
            # if current_files_hash != st.session_state.processed_files_hash: -> Logic này có thể phức tạp, tạm thời bỏ qua, xử lý mỗi lần nhấn nút
            print("--- Nhấn nút Xử lý Tài liệu ---")
            if models_ok and chroma_client:
                success = process_uploaded_files(uploaded_files, chroma_client)
                if success:
                    st.session_state.rag_initialized = True
                    st.session_state.engine_loaded = False # Buộc load lại engine
                    # st.session_state.processed_files_hash = current_files_hash # Lưu hash file đã xử lý
                    st.success("Xử lý hoàn tất! Hệ thống đã sẵn sàng cho RAG.")
                    st.rerun() # Chạy lại để load engine mới và cập nhật UI
                else:
                    st.error("Xử lý tài liệu thất bại. Vui lòng kiểm tra log lỗi trong console.")
            else:
                 st.error("Lỗi khởi tạo Model hoặc ChromaDB, không thể xử lý file.")
            # else:
            #     st.info("Các file này đã được xử lý trước đó.")
        else:
            st.warning("Vui lòng chọn ít nhất một file PDF để xử lý.")

    st.divider()
    # Hiển thị trạng thái RAG rõ ràng hơn
    rag_status = st.session_state.get("rag_initialized", False)
    status_text = "Sẵn sàng (dựa trên tài liệu đã xử lý)" if rag_status else "Chưa sẵn sàng (hỏi đáp thông thường)"
    status_icon = "✅" if rag_status else "ℹ️"
    st.markdown(f"**{status_icon} Trạng thái RAG:** {status_text}")
    if rag_status and not st.session_state.get("query_engine"):
         st.warning("RAG đã được kích hoạt nhưng Query Engine chưa tải được. Có thể cần xử lý lại file hoặc kiểm tra lỗi.")


# --- Khu vực Chat chính ---
print(f"--- Trạng thái RAG hiện tại: {st.session_state.rag_initialized}, Query Engine: {'Có' if st.session_state.query_engine else 'Không'} ---")

# Hiển thị lịch sử chat
container = st.container() # Tạo container để chat messages không bị render lại lung tung
with container:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

# Input câu hỏi từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn...", key="chat_input"):
    if not models_ok:
         st.error("Hệ thống chưa sẵn sàng do lỗi khởi tạo model.")
         st.stop()

    # Thêm câu hỏi vào lịch sử và hiển thị ngay lập tức
    st.session_state.messages.append({"role": "user", "content": prompt})
    with container: # Hiển thị lại trong container
         st.chat_message("user").write(prompt)

    # Tạo placeholder cho phản hồi của assistant
    response_placeholder = container.chat_message("assistant").empty()
    response_text = ""

    # Quyết định dùng RAG hay chat thông thường
    use_rag = st.session_state.get("rag_initialized", False)
    current_query_engine = st.session_state.get("query_engine", None)

    if use_rag and current_query_engine:
        print(f"--- Sử dụng RAG Query Engine để trả lời câu hỏi: '{prompt[:50]}...' ---")
        try:
            with st.spinner("🧠 Đang tư duy dựa trên tài liệu..."):
                response = current_query_engine.query(prompt)
                response_text = str(response)

            response_placeholder.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

            # Hiển thị nguồn tham khảo (nếu có)
            if response_text and hasattr(response, 'source_nodes') and response.source_nodes:
                 with st.expander("Nguồn tài liệu tham khảo (từ RAG)"):
                     for i, snode in enumerate(response.source_nodes):
                        metadata = snode.metadata
                        # Cố gắng lấy tên file gốc từ metadata nếu SimpleDirectoryReader thêm vào
                        file_name = metadata.get('file_name', metadata.get('file_path', 'N/A'))
                        display_name = os.path.basename(file_name)
                        file_type = metadata.get('file_type', 'pdf')
                        relevance_score = snode.score

                        st.markdown(f"**Nguồn {i+1}:**")
                        st.markdown(f"- **File:** `{display_name}` ({file_type})")
                        if relevance_score is not None:
                            st.markdown(f"- **Độ liên quan (Score):** `{relevance_score:.4f}`")
                        else:
                            st.markdown(f"- **Độ liên quan (Score):** `N/A`")
                        # Thêm tùy chọn hiển thị nội dung node (hữu ích khi debug)
                        # with st.popover("Xem trích dẫn"):
                        #      st.markdown(f"```\n{snode.get_content()}\n```")
                        st.divider()

        except Exception as e:
            error_message = f"Rất tiếc, đã xảy ra lỗi khi truy vấn RAG: {e}"
            print(f"RAG Query Error: {e}")
            traceback.print_exc()
            response_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": f"Lỗi RAG: {error_message}"}) # Thêm lỗi vào history

    elif use_rag and not current_query_engine:
         warning_message = "Chế độ RAG đang bật nhưng Query Engine chưa sẵn sàng. Có thể cần xử lý lại file hoặc kiểm tra lỗi khởi tạo."
         print(warning_message)
         response_placeholder.warning(warning_message)
         st.session_state.messages.append({"role": "assistant", "content": warning_message}) # Thông báo cho user

    else: # Không dùng RAG -> Chat thông thường
        print(f"--- Sử dụng Gemini thông thường để trả lời câu hỏi: '{prompt[:50]}...' ---")
        if genai_chat_model:
            with st.spinner("💬 Đang trò chuyện với Gemini..."):
                 # Lấy lịch sử chat hiện tại (bỏ câu hỏi user cuối cùng)
                 chat_history_for_api = st.session_state.messages[:-1]
                 response_text = get_general_gemini_response(prompt, genai_chat_model, chat_history_for_api)

            response_placeholder.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
             error_message = "Lỗi: Mô hình chat thông thường chưa được khởi tạo."
             print(error_message)
             response_placeholder.error(error_message)
             st.session_state.messages.append({"role": "assistant", "content": error_message})
