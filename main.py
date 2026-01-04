import os
import transformers
import argparse
import sys
import json
import shutil
import glob
import numpy as np  # 用于计算相似度
import dashscope
from dashscope import TextEmbedding
import chromadb
import pypdf
from typing import List, Optional
# === 设置环境变量，使用 HF-Mirror 镜像加速本地 CLIP 下载 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ======================================================
from sentence_transformers import SentenceTransformer
from PIL import Image


# ==========================================
# 工具类: 配置与文件管理
# ==========================================

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        print(f"Error: 配置文件 {config_path} 未找到。")
        sys.exit(1)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: 读取配置文件失败 - {e}")
        sys.exit(1)


class FileManager:
    """
    负责将外部文件复制到项目内部存储目录，支持按主题分类存储
    """

    def __init__(self, storage_root):
        self.storage_root = storage_root
        self.paper_root = os.path.join(storage_root, "papers")
        self.image_root = os.path.join(storage_root, "images")

        os.makedirs(self.paper_root, exist_ok=True)
        os.makedirs(self.image_root, exist_ok=True)

    def save_paper(self, source_path, topic="Uncategorized"):
        """
        保存论文到对应主题的子文件夹
        """
        filename = os.path.basename(source_path)
        # 创建主题子目录: local_storage/papers/CV/
        target_dir = os.path.join(self.paper_root, topic)
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, filename)

        try:
            shutil.copy2(source_path, target_path)
            return target_path
        except Exception as e:
            print(f"Error copying file: {e}")
            return None

    def save_image(self, source_path):
        """保存图片"""
        filename = os.path.basename(source_path)
        target_path = os.path.join(self.image_root, filename)
        try:
            shutil.copy2(source_path, target_path)
            return target_path
        except Exception as e:
            print(f"Error copying file: {e}")
            return None


# ==========================================
# 核心服务类 (Hybrid: Ali Cloud + Local CLIP)
# ==========================================

class HybridEmbeddingService:
    def __init__(self, api_key, text_model):
        # 1. 初始化阿里云文本服务
        dashscope.api_key = api_key
        self.text_model = text_model

        # 2. 初始化本地 CLIP 模型 (用于图片和以文搜图)
        try:
            # --- 修改开始 ---
            # 临时将 transformers 的日志级别设置为 ERROR，只看报错，不看警告
            transformers.logging.set_verbosity_error()
            self.local_clip = SentenceTransformer('clip-ViT-B-32')
            # (推荐) 加载完成后，将日志级别恢复为 WARNING，以免错过后续可能的其他重要警告
            transformers.logging.set_verbosity_warning()
            # --- 修改结束 ---
        except Exception as e:
            print(f"Warning: CLIP 模型加载失败，图片功能将不可用. {e}")
            self.local_clip = None

    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """调用阿里云 text-embedding-v3"""
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

    def get_clip_embedding(self, text: str = None, image_path: str = None) -> Optional[List[float]]:
        """使用本地 CLIP 模型生成向量"""
        if not self.local_clip: return None
        try:
            if image_path:
                img = Image.open(image_path)
                return self.local_clip.encode(img).tolist()
            elif text:
                return self.local_clip.encode(text).tolist()
            return None
        except Exception as e:
            print(f"Local CLIP Error: {e}")
            return None


class Classifier:
    """
    基于向量相似度的分类器 (不使用 LLM)
    """

    def __init__(self, embedding_service):
        self.service = embedding_service

    def classify(self, text_embedding: List[float], topics: List[str]) -> str:
        """
        原理: 计算文本向量与每个 Topic 向量的相似度，取最大者。
        """
        if not topics:
            return "Uncategorized"

        # 1. 获取所有 Topic 的向量
        topic_embeddings = []
        valid_topics = []

        for t in topics:
            # 这里给 Topic 加一点描述词，增加匹配准确度
            # 例如用户输入 "CV"，我们将其扩展为 "Computer Vision paper" 去做嵌入，效果更好
            # 但为了通用性，我们直接嵌入 Topic 原词
            emb = self.service.get_text_embedding(t)
            if emb:
                topic_embeddings.append(emb)
                valid_topics.append(t)

        if not topic_embeddings:
            return "Uncategorized"

        # 2. 计算余弦相似度
        # Cosine Similarity = (A . B) / (||A|| * ||B||)
        # 由于 API 返回的通常已归一化，简化为点积 A . B
        scores = []
        vec_paper = np.array(text_embedding)

        for vec_topic in topic_embeddings:
            vec_topic = np.array(vec_topic)
            # 计算点积
            score = np.dot(vec_paper, vec_topic)
            scores.append(score)

        # 3. 找出最大值
        best_idx = np.argmax(scores)
        best_topic = valid_topics[best_idx]

        # (可选) 可以设置一个阈值，如果相似度都太低，归为 Other
        # print(f"Debug: Scores {dict(zip(valid_topics, scores))}")

        return best_topic


class LocalDBManager:
    def __init__(self, db_path):
        # 此时假设 CLIP 维度 512，阿里文本维度 1024
        self.client = chromadb.PersistentClient(path=db_path)
        self.paper_collection = self.client.get_or_create_collection(name="paper_collection",
                                                                     metadata={"hnsw:space": "cosine"})
        self.image_collection = self.client.get_or_create_collection(name="image_collection",
                                                                     metadata={"hnsw:space": "cosine"})

    def add_paper(self, doc_id, embedding, metadata, snippet):
        self.paper_collection.add(ids=[doc_id], embeddings=[embedding], metadatas=[metadata], documents=[snippet])

    def add_image(self, img_id, embedding, metadata):
        self.image_collection.add(ids=[img_id], embeddings=[embedding], metadatas=[metadata])

    def search_paper(self, query_vec, top_k=3):
        return self.paper_collection.query(query_embeddings=[query_vec], n_results=top_k)

    def search_image(self, query_vec, top_k=3):
        return self.image_collection.query(query_embeddings=[query_vec], n_results=top_k)


# ==========================================
# 业务逻辑
# ==========================================

def extract_text_from_pdf(pdf_path: str, max_chars: int = 3000) -> str:
    text_content = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i in range(min(3, len(reader.pages))):  # 读前3页以获取更多语义
            text = reader.pages[i].extract_text()
            if text: text_content += text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""
    return text_content[:max_chars].replace('\n', ' ')


def process_single_paper(file_path, topics_str, ali_service, classifier, file_manager, db_manager):
    """处理单个论文的核心逻辑：提取 -> 向量化 -> 分类 -> 移动 -> 索引"""

    # 1. 解析 Topics
    topics = [t.strip() for t in topics_str.split(',')] if topics_str else []

    # 2. 提取文本
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text:
        print(f"跳过: 无法读取文本 {os.path.basename(file_path)}")
        return

    # 3. 生成文本向量 (用于搜索和分类)
    print(f"正在分析语义: {os.path.basename(file_path)} ...")
    vector = ali_service.get_text_embedding(raw_text)
    if not vector:
        return

    # 4. 自动分类 (如果提供了 topics)
    assigned_topic = "Uncategorized"
    if topics:
        assigned_topic = classifier.classify(vector, topics)
        print(f"    -> 自动归类为: [{assigned_topic}]")  # Output Statement

    # 5. 文件归档 (移动到对应文件夹)
    internal_path = file_manager.save_paper(file_path, topic=assigned_topic)
    if not internal_path:
        return

    # 6. 存入数据库
    filename = os.path.basename(internal_path)
    db_manager.add_paper(
        doc_id=filename,
        embedding=vector,
        metadata={
            "path": internal_path,
            "filename": filename,
            "topic": assigned_topic,
            "original_path": file_path
        },
        snippet=raw_text[:300] + "..."
    )
    print(f"成功: 论文已索引至 '{assigned_topic}' 目录。")  # Output Statement


def main():
    config = load_config()

    # 参数解析
    parser = argparse.ArgumentParser(description="本地多模态 AI 助手 (自动分类版)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # add_paper
    ap_parser = subparsers.add_parser("add_paper", help="添加并分类单篇论文")
    ap_parser.add_argument("path", type=str, help="PDF文件路径")
    ap_parser.add_argument("--topics", type=str, default="", help="可选主题列表,逗号分隔, e.g. 'CV,NLP,RL'")

    # organize_folder (批量整理)
    org_parser = subparsers.add_parser("organize_folder", help="一键整理文件夹")
    org_parser.add_argument("folder_path", type=str, help="待整理的文件夹路径")
    org_parser.add_argument("--topics", type=str, required=True, help="分类主题列表, e.g. 'CV,NLP,Physics'")

    # search_paper
    sp_parser = subparsers.add_parser("search_paper", help="语义搜索论文")
    sp_parser.add_argument("query", type=str, help="搜索关键词")

    # add_image
    ai_parser = subparsers.add_parser("add_image", help="添加图片")
    ai_parser.add_argument("path", type=str, help="图片路径")

    # search_image
    si_parser = subparsers.add_parser("search_image", help="以文搜图")
    si_parser.add_argument("query", type=str, help="图片描述")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 初始化
    print(">>> 正在初始化系统...")
    file_manager = FileManager(config["storage_root"])

    # 混合服务: 阿里(Text) + 本地(CLIP)
    ali_service = HybridEmbeddingService(
        api_key=config["dashscope_api_key"],
        text_model=config["text_embedding_model"]
    )

    classifier = Classifier(ali_service)
    db_manager = LocalDBManager(config["db_path"])
    print(">>> 系统初始化完成.\n")

    # === 功能分发 ===

    if args.command == "add_paper":
        if not os.path.exists(args.path):
            print(f"错误: 文件 {args.path} 不存在。")
            return
        process_single_paper(args.path, args.topics, ali_service, classifier, file_manager, db_manager)

    elif args.command == "organize_folder":
        source_folder = args.folder_path
        if not os.path.isdir(source_folder):
            print(f"错误: 目录 {source_folder} 不存在。")
            return

        # 扫描所有 PDF
        pdf_files = glob.glob(os.path.join(source_folder, "*.pdf"))
        print(f"扫描到 {len(pdf_files)} 个 PDF 文件，开始整理...")  # Input Statement

        for pdf_path in pdf_files:
            print(f"--- 处理: {os.path.basename(pdf_path)} ---")
            process_single_paper(pdf_path, args.topics, ali_service, classifier, file_manager, db_manager)

        print("\n>>> 文件夹整理完成！所有文件已移动并建立索引。")  # Output Statement

    elif args.command == "search_paper":
        print(f"正在搜索: '{args.query}' ...")
        q_vec = ali_service.get_text_embedding(args.query)
        if q_vec:
            results = db_manager.search_paper(q_vec)
            # 格式化输出
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            docs = results['documents'][0]
            dists = results['distances'][0]

            if not ids: print("未找到相关论文。")
            for i in range(len(ids)):
                print(f"[{i + 1}] {metadatas[i]['filename']}")
                print(f"    分类: {metadatas[i].get('topic', 'N/A')}")
                print(f"    路径: {metadatas[i]['path']}")
                print(f"    相似度: {1 - dists[i]:.4f}")
                print("-" * 30)

    elif args.command == "add_image":
        if not os.path.exists(args.path): return
        print(f"处理图片: {args.path}")
        internal_path = file_manager.save_image(args.path)
        vec = ali_service.get_clip_embedding(image_path=internal_path)
        if vec:
            filename = os.path.basename(internal_path)
            db_manager.add_image(filename, vec, {"path": internal_path, "filename": filename})
            print(f"成功: 图片已索引。")

    elif args.command == "search_image":
        print(f"搜图: '{args.query}' ...")
        q_vec = ali_service.get_clip_embedding(text=args.query)
        if q_vec:
            results = db_manager.search_image(q_vec)
            ids = results['ids'][0]
            metadatas = results['metadatas'][0]
            dists = results['distances'][0]

            if not ids: print("未找到图片。")
            for i in range(len(ids)):
                print(f"[{i + 1}] {metadatas[i]['filename']}")
                print(f"    路径: {metadatas[i]['path']}")
                print(f"    相似度: {1 - dists[i]:.4f}")
                print("-" * 30)


if __name__ == "__main__":
    main()