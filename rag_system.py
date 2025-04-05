import os
import logging
import sys
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "API key của Gemini chưa được đặt. Hãy tạo file .env và thêm GEMINI_API_KEY='YOUR_API_KEY'"
    )
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

print("--- Khởi tạo mô hình LLM và Embedding ---")
Settings.llm = Gemini(api_key=GEMINI_API_KEY, model_name="models/gemini-2.0-flash")
Settings.embed_model = GeminiEmbedding(
    api_key=GEMINI_API_KEY, model_name="models/embedding-001"
)
print("--- Đã khởi tạo mô hình LLM và Embedding ---")
DATA_DIR = "data"
print(f"--- Tải tài liệu từ thư mục: {DATA_DIR} ---")
try:
    reader = SimpleDirectoryReader(
        input_dir=DATA_DIR,
        recursive=True,
    )
    documents = reader.load_data()
    if not documents:
        print(
            f"Không tìm thấy tài liệu nào trong thư mục '{DATA_DIR}'. Hãy đảm bảo bạn đã đặt các file PDF, TXT, DOCX vào đó."
        )
        sys.exit()
    print(f"--- Đã tải {len(documents)} tài liệu ---")
    for doc in documents:
        file_path = doc.metadata.get("file_path")
        if file_path:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == ".pdf":
                doc.metadata["file_type"] = "pdf"
            elif file_extension == ".txt":
                doc.metadata["file_type"] = "txt"
            elif file_extension == ".docx":
                doc.metadata["file_type"] = "docx"
            else:
                doc.metadata["file_type"] = "other"
        else:
            print(
                f"   - Cảnh báo: Không tìm thấy 'file_path' trong metadata cho một tài liệu."
            )
except FileNotFoundError:
    print(
        f"Lỗi: Thư mục '{DATA_DIR}' không tồn tại. Hãy tạo thư mục này và đặt tài liệu vào đó."
    )
    sys.exit()
except Exception as e:
    print(f"Lỗi khi tải tài liệu: {e}")
    sys.exit()
print("--- Phân tách tài liệu thành các node ---")
node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
print(f"--- Đã tạo {len(nodes)} node ---")
PERSIST_DIR_VECTOR = "./storage_vector"
PERSIST_DIR_SUMMARY = "./storage_summary"
vector_index = None
summary_index = None
if not os.path.exists(PERSIST_DIR_VECTOR):
    print(f"--- Tạo Vector Index mới (vì chưa có trong {PERSIST_DIR_VECTOR}) ---")
    vector_index = VectorStoreIndex(nodes, show_progress=True)
    print("--- Đang lưu trữ Vector Index... ---")
    vector_index.storage_context.persist(persist_dir=PERSIST_DIR_VECTOR)
else:
    print(f"--- Tải Vector Index từ {PERSIST_DIR_VECTOR} ---")
    storage_context_vector = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR_VECTOR
    )
    vector_index = load_index_from_storage(storage_context_vector)
if not os.path.exists(PERSIST_DIR_SUMMARY):
    print(f"--- Tạo Summary Index mới (vì chưa có trong {PERSIST_DIR_SUMMARY}) ---")
    summary_index = SummaryIndex(nodes, show_progress=True)
    print("--- Đang lưu trữ Summary Index... ---")
    summary_index.storage_context.persist(persist_dir=PERSIST_DIR_SUMMARY)
else:
    print(f"--- Tải Summary Index từ {PERSIST_DIR_SUMMARY} ---")
    storage_context_summary = StorageContext.from_defaults(
        persist_dir=PERSIST_DIR_SUMMARY
    )
    summary_index = load_index_from_storage(storage_context_summary)
print("--- Đã tạo/tải xong các Index ---")
print("--- Tạo các Query Engine ---")
vector_query_engine = vector_index.as_query_engine(similarity_top_k=3)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize", use_async=True
)
print("--- Đã tạo các Query Engine ---")
print("--- Tạo các Query Engine Tool ---")
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    name="vector_search_tool",
    description=(
        "Hữu ích để tìm kiếm và truy xuất thông tin cụ thể, chi tiết, hoặc trả lời câu hỏi trực tiếp từ nội dung các tài liệu (PDF, TXT, DOCX)."
        " Sử dụng công cụ này khi câu hỏi mang tính chất 'là gì', 'ai', 'khi nào', 'ở đâu', 'như thế nào' hoặc yêu cầu tìm một đoạn văn bản cụ thể."
    ),
)
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    name="summary_tool",
    description=(
        "Hữu ích để tóm tắt nội dung chính của toàn bộ hoặc một phần lớn các tài liệu."
        " Sử dụng công cụ này khi câu hỏi yêu cầu 'tóm tắt', 'ý chính', 'tổng quan' về tài liệu."
    ),
)
print("--- Đã tạo các Query Engine Tool ---")
print("--- Tạo Router Query Engine ---")
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(llm=Settings.llm),
    query_engine_tools=[
        vector_tool,
        summary_tool,
    ],
    verbose=True,
)
print("--- Đã tạo Router Query Engine ---")
print("\n=== HỆ THỐNG RAG ĐÃ SẴN SÀNG ===")
if __name__ == "__main__":
    print("Nhập câu hỏi của bạn (hoặc gõ 'exit' để thoát):")
    while True:
        question = input("Câu hỏi: ")
        if question.lower() == "exit":
            break
        if not question.strip():
            continue
        print(f"\nĐang xử lý câu hỏi: '{question}'")
        try:
            response = query_engine.query(question)
            print("\nPhản hồi:")
            print(str(response))
            print("-" * 30)
            if hasattr(response, "source_nodes") and response.source_nodes:
                print("Nguồn tài liệu tham khảo:")
                for snode in response.source_nodes:
                    file_name = snode.metadata.get("file_path", "N/A")
                    print(
                        f"  - File: {os.path.basename(file_name)}, Score: {snode.score:.4f}"
                    )
                print("-" * 30)
        except Exception as e:
            print(f"Lỗi khi truy vấn: {e}")
    print("=== Kết thúc ===")
