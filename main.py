import os
import argparse
import sys
import json
import shutil
import dashscope
from dashscope import TextEmbedding, MultiModalEmbedding
import chromadb
from chromadb.config import Settings
import pypdf
from typing import List, Optional


# ==========================================
# 工具类: 配置与文件管理
# ==========================================

def load_config(config_path="config.json"):
    """读取外部 JSON 配置文件"""
    if not os.path.exists(config_path):
        print(f"Error: 配置文件 {config_path} 未找到。请确保该文件在项目根目录下。")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: 读取配置文件失败 - {e}")
        sys.exit(1)


class FileManager:
    """
    负责将外部文件复制到项目内部存储目录
    """

    def __init__(self, storage_root):
        self.storage_root = storage_root
        self.paper_dir = os.path.join(storage_root, "papers")
        self.image_dir = os.path.join(storage_root, "images")

        # 确保目录存在
        os.makedirs(self.paper_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def save_file(self, source_path, file_type):
        """
        复制文件到内部存储
        :param source_path: 原始文件路径
        :param file_type: 'paper' 或 'image'
        :return: 内部存储的新路径
        """
        filename = os.path.basename(source_path)

        if file_type == 'paper':
            target_dir = self.paper_dir
        elif file_type == 'image':
            target_dir = self.image_dir
        else:
            raise ValueError("Unknown file type")

        target_path = os.path.join(target_dir, filename)

        # 执行复制 (copy2 保留文件元数据)
        try:
            shutil.copy2(source_path, target_path)
            return target_path
        except Exception as e:
            print(f"Error copying file: {e}")
            return None


# ==========================================
# 核心服务类
# ==========================================

class AliyunEmbeddingService:
    """
    封装阿里云 DashScope 的 Embedding 服务
    """

    def __init__(self, api_key, text_model, multimodal_model):
        dashscope.api_key = api_key
        self.text_model = text_model
        self.multimodal_model = multimodal_model

    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """调用文本向量模型"""
        try:
            resp = TextEmbedding.call(
                model=self.text_model,
                input=text,
                dimension=1024
            )
            if resp.status_code == 200:
                return resp.output['embeddings'][0]['embedding']
            else:
                print(f"API Error (Text): {resp}")
                return None
        except Exception as e:
            print(f"Exception during text embedding: {e}")
            return None

    def get_multimodal_embedding(self, text: str = None, image_path: str = None) -> Optional[List[float]]:
        """调用多模态向量模型"""
        input_data = {}
        if text:
            input_data['text'] = text
        if image_path:
            input_data['image'] = image_path

        try:
            resp = MultiModalEmbedding.call(
                model=self.multimodal_model,
                input=input_data
            )
            if resp.status_code == 200:
                return resp.output['embeddings'][0]['embedding']
            else:
                print(f"API Error (Multimodal): {resp}")
                return None
        except Exception as e:
            print(f"Exception during multimodal embedding: {e}")
            return None


class LocalDBManager:
    """
    封装 ChromaDB 本地向量数据库管理
    """

    def __init__(self, db_path):
        self.client = chromadb.PersistentClient(path=db_path)

        # 1. 文献集合
        self.paper_collection = self.client.get_or_create_collection(
            name="paper_collection",
            metadata={"hnsw:space": "cosine"}
        )

        # 2. 图像集合
        self.image_collection = self.client.get_or_create_collection(
            name="image_collection",
            metadata={"hnsw:space": "cosine"}
        )

    def add_paper(self, doc_id: str, embedding: List[float], metadata: dict, text_snippet: str):
        self.paper_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text_snippet]
        )

    def add_image(self, img_id: str, embedding: List[float], metadata: dict):
        self.image_collection.add(
            ids=[img_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def search_paper(self, query_embedding: List[float], top_k: int = 3):
        return self.paper_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    def search_image(self, query_embedding: List[float], top_k: int = 3):
        return self.image_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )


# ==========================================
# 业务逻辑处理
# ==========================================

def extract_text_from_pdf(pdf_path: str, max_chars: int = 2000) -> str:
    """提取PDF前几页的文本"""
    text_content = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i in range(min(2, len(reader.pages))):
            text = reader.pages[i].extract_text()
            if text:
                text_content += text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    return text_content[:max_chars].replace('\n', ' ')


def main():
    # 1. 加载配置
    config = load_config()

    # 2. 参数解析
    parser = argparse.ArgumentParser(description="本地多模态 AI 智能助手 (Powered by Alibaba Cloud)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # add_paper
    add_paper_parser = subparsers.add_parser("add_paper", help="Add a PDF paper")
    add_paper_parser.add_argument("path", type=str, help="Path to the PDF file")

    # search_paper
    search_paper_parser = subparsers.add_parser("search_paper", help="Semantic search for papers")
    search_paper_parser.add_argument("query", type=str, help="Search query")

    # add_image
    add_image_parser = subparsers.add_parser("add_image", help="Add an image")
    add_image_parser.add_argument("path", type=str, help="Path to the image file")

    # search_image
    search_image_parser = subparsers.add_parser("search_image", help="Search images by text")
    search_image_parser.add_argument("query", type=str, help="Image description")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 3. 初始化各模块
    print(">>> 正在初始化系统...")

    # 文件管理器
    file_manager = FileManager(config["storage_root"])

    # 阿里云服务
    ali_service = AliyunEmbeddingService(
        api_key=config["dashscope_api_key"],
        text_model=config["text_embedding_model"],
        multimodal_model=config["multimodal_embedding_model"]
    )

    # 数据库管理器
    db_manager = LocalDBManager(config["db_path"])

    print(">>> 系统初始化完成.\n")

    # 4. 逻辑分发
    if args.command == "add_paper":
        source_path = args.path
        if not os.path.exists(source_path):
            print(f"错误: 文件 {source_path} 不存在。")
            return

        print(f"正在导入论文: {source_path} ...")

        # A. 复制文件到内部存储
        internal_path = file_manager.save_file(source_path, 'paper')
        if not internal_path:
            return
        print(f"已归档至: {internal_path}")

        # B. 提取文本 (从内部文件提取)
        extracted_text = extract_text_from_pdf(internal_path)
        if not extracted_text:
            print("警告: 未能从 PDF 提取到文本，跳过索引。")
            return

        # C. 生成向量
        print("正在生成向量...")
        vector = ali_service.get_text_embedding(extracted_text)

        # D. 存入数据库 (存内部路径)
        if vector:
            filename = os.path.basename(internal_path)
            db_manager.add_paper(
                doc_id=filename,
                embedding=vector,
                metadata={"path": internal_path, "filename": filename, "original_path": source_path},
                text_snippet=extracted_text[:300] + "..."
            )
            print(f"成功: 论文 '{filename}' 已添加至知识库。")  # Input/Output Statement

    elif args.command == "search_paper":
        print(f"正在搜索: '{args.query}' ...")
        query_vector = ali_service.get_text_embedding(args.query)

        if query_vector:
            results = db_manager.search_paper(query_vector)

            print("\n" + "=" * 40)
            print("       文献搜索结果 (Papers)       ")
            print("=" * 40)

            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            documents = results['documents'][0]

            if not ids:
                print("未找到相关论文。")

            for i in range(len(ids)):
                score = 1 - distances[i]
                print(f"[{i + 1}] {metadatas[i]['filename']}")
                print(f"    相似度: {score:.4f}")
                print(f"    内部存储路径: {metadatas[i]['path']}")
                print(f"    摘要片段: {documents[i]}")
                print("-" * 40)

    elif args.command == "add_image":
        source_path = args.path
        if not os.path.exists(source_path):
            print(f"错误: 文件 {source_path} 不存在。")
            return

        print(f"正在导入图片: {source_path} ...")

        # A. 复制文件到内部存储
        internal_path = file_manager.save_file(source_path, 'image')
        if not internal_path:
            return
        print(f"已归档至: {internal_path}")

        # B. 生成向量 (使用内部图片路径)
        print("正在生成多模态向量...")
        vector = ali_service.get_multimodal_embedding(image_path=internal_path)

        # C. 存入数据库
        if vector:
            filename = os.path.basename(internal_path)
            db_manager.add_image(
                img_id=filename,
                embedding=vector,
                metadata={"path": internal_path, "filename": filename, "original_path": source_path}
            )
            print(f"成功: 图片 '{filename}' 已添加至知识库。")  # Input/Output Statement

    elif args.command == "search_image":
        print(f"正在以文搜图: '{args.query}' ...")
        query_vector = ali_service.get_multimodal_embedding(text=args.query)

        if query_vector:
            results = db_manager.search_image(query_vector)

            print("\n" + "=" * 40)
            print("       图像搜索结果 (Images)       ")
            print("=" * 40)

            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]

            if not ids:
                print("未找到相关图片。")

            for i in range(len(ids)):
                score = 1 - distances[i]
                print(f"[{i + 1}] {metadatas[i]['filename']}")
                print(f"    相似度: {score:.4f}")
                print(f"    内部存储路径: {metadatas[i]['path']}")
                print("-" * 40)


if __name__ == "__main__":
    main()