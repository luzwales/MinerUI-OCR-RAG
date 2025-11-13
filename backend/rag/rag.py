NULL = None
import faiss
import torch
import faiss
import numpy as np
from llama_index.core.node_parser import SentenceSplitter
import re
from typing import List, Dict, Any, Optional, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil
from openai import OpenAI
import os
import fitz  # PyMuPDF
import chardet  # 用于自动检测编码
import traceback
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config  # 导入配置文件
from task_db import TaskDB

# 创建知识库根目录和临时文件目录
KB_BASE_DIR = Config.kb_base_dir
os.makedirs(KB_BASE_DIR, exist_ok=True)

# 创建默认知识库目录
DEFAULT_KB = Config.default_kb
DEFAULT_KB_DIR = os.path.join(KB_BASE_DIR, DEFAULT_KB)
os.makedirs(DEFAULT_KB_DIR, exist_ok=True)

# 创建临时输出目录
OUTPUT_DIR = Config.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化数据库连接
db = TaskDB()

client = OpenAI(
    api_key=Config.llm_api_key,
    base_url=Config.llm_base_url
)

class DeepSeekClient:
    def generate_answer(self, system_prompt, user_prompt, model=Config.llm_model):
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip()

# 获取知识库列表
# 获取知识库列表
def get_knowledge_bases() -> List[Dict]:
    """获取所有知识库的完整信息列表"""
    try:
        # 确保知识库根目录存在
        if not os.path.exists(KB_BASE_DIR):
            os.makedirs(KB_BASE_DIR, exist_ok=True)
            
        # 从数据库获取知识库完整记录
        kb_records = db.get_knowledge_bases()
        
        # 获取文件系统中的知识库目录
        kb_dirs = [d for d in os.listdir(KB_BASE_DIR) 
                  if os.path.isdir(os.path.join(KB_BASE_DIR, d))]
        
        # 确保默认知识库存在于文件系统和数据库中
        if DEFAULT_KB not in [record['kb_name'] for record in kb_records]:
            # 如果默认知识库目录存在但数据库中没有记录，创建数据库记录
            if DEFAULT_KB in kb_dirs:
                db.create_knowledge_base(DEFAULT_KB)
                # 重新获取完整记录
                kb_records = db.get_knowledge_bases()
            else:
                # 如果默认知识库目录也不存在，创建目录和数据库记录
                os.makedirs(DEFAULT_KB_DIR, exist_ok=True)
                db.create_knowledge_base(DEFAULT_KB)
                # 重新获取完整记录
                kb_records = db.get_knowledge_bases()
        
        # 确保文件系统和数据库同步（可选，根据实际需求）
        for kb_dir in kb_dirs:
            if kb_dir not in [record['kb_name'] for record in kb_records]:
                # 如果文件系统有但数据库没有，创建数据库记录
                db.create_knowledge_base(kb_dir)
                # 重新获取完整记录
                kb_records = db.get_knowledge_bases()
        
        # 按创建时间排序返回完整记录
        return sorted(kb_records, key=lambda x: x['created_at'])
    except Exception as e:
        print(f"获取知识库列表失败: {str(e)}")
        # 出错时返回包含默认知识库的最小记录
        return [{"id": 1, "kb_name": DEFAULT_KB, "system_prompt": "", 
                "created_at": "", "updated_at": "", "created_by": "system", "updated_by": "system"}]

# 更新知识库
def update_knowledge_base(kb_name: str, system_prompt: str = '', updated_by: str = 'system') -> str:
    """更新知识库的system_prompt和update_by字段"""
    try:
        # 检查知识库是否存在
        kb_record = db.get_knowledge_base_by_name(kb_name)
        if not kb_record:
            return f"知识库 '{kb_name}' 不存在"
            
        # 更新知识库记录，更新update_by为当前时间
        import datetime
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if db.update_knowledge_base(kb_name, system_prompt, updated_by="system"):
            return f"知识库 '{kb_name}' 已更新"
        else:
            return f"更新知识库 '{kb_name}' 失败"
    except Exception as e:
        return f"更新知识库失败: {str(e)}"

# 创建新知识库
def create_knowledge_base(kb_name: str, system_prompt: str = '', created_by: str = 'system') -> str:
    """创建新的知识库"""
    try:
        if not kb_name or not kb_name.strip():
            return "错误：知识库名称不能为空"
            
        # 净化知识库名称，只允许字母、数字、下划线和中文
        kb_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', kb_name.strip())
        
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if os.path.exists(kb_path):
            return f"知识库 '{kb_name}' 已存在"
            
        # 先创建文件系统目录
        os.makedirs(kb_path, exist_ok=True)
        
        # 再创建数据库记录
        if db.create_knowledge_base(kb_name, system_prompt, created_by):
            return f"知识库 '{kb_name}' 创建成功"
        else:
            # 如果数据库记录创建失败，删除已创建的目录
            if os.path.exists(kb_path):
                shutil.rmtree(kb_path)
            return f"创建知识库数据库记录失败"
    except Exception as e:
        return f"创建知识库失败: {str(e)}"

# 删除知识库
def delete_knowledge_base(kb_name: str) -> str:
    """删除指定的知识库"""
    try:
        if kb_name == DEFAULT_KB:
            return f"无法删除默认知识库 '{DEFAULT_KB}'"
            
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        # 先从数据库删除记录
        if db.delete_knowledge_base(kb_name):
            # 如果数据库记录删除成功，再删除文件系统目录
            if os.path.exists(kb_path):
                shutil.rmtree(kb_path)
            return f"知识库 '{kb_name}' 已删除"
        else:
            # 数据库记录不存在或删除失败
            if os.path.exists(kb_path):
                shutil.rmtree(kb_path)
                return f"知识库记录不存在，文件目录刚刚删除"
            else:
                return f"知识库 '{kb_name}' 不存在"
    except Exception as e:
        return f"删除知识库失败: {str(e)}"

# 获取知识库文件列表
def get_kb_files(kb_name: str) -> List[str]:
    """获取指定知识库中的文件列表"""
    try:
        kb_path = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_path):
            return []
            
        # 获取所有文件（排除索引文件和元数据文件）
        files = [f for f in os.listdir(kb_path) 
                if os.path.isfile(os.path.join(kb_path, f)) and 
                not f.endswith(('.index', '.json'))]
        
        return sorted(files)
    except Exception as e:
        print(f"获取知识库文件列表失败: {str(e)}")
        return []

# 语义分块函数
def semantic_chunk(text: str, chunk_size=800, chunk_overlap=20) -> List[dict]:
    class EnhancedSentenceSplitter(SentenceSplitter):
        def __init__(self, *args, **kwargs):
            custom_seps = ["；", "!", "?", "\n"]
            separators = [kwargs.get("separator", "。")] + custom_seps
            kwargs["separator"] = '|'.join(map(re.escape, separators))
            super().__init__(*args, **kwargs)

        def _split_text(self, text: str, **kwargs) -> List[str]:
            splits = re.split(f'({self.separator})', text)
            chunks = []
            current_chunk = []
            for part in splits:
                part = part.strip()
                if not part:
                    continue
                if re.fullmatch(self.separator, part):
                    if current_chunk:
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                else:
                    current_chunk.append(part)
            if current_chunk:
                chunks.append("".join(current_chunk))
            return [chunk.strip() for chunk in chunks if chunk.strip()]

    text_splitter = EnhancedSentenceSplitter(
        separator="。",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n"
    )

    paragraphs = []
    current_para = []
    current_len = 0

    for para in text.split("\n\n"):
        para = para.strip()
        para_len = len(para)
        if para_len == 0:
            continue
        if current_len + para_len <= chunk_size:
            current_para.append(para)
            current_len += para_len
        else:
            if current_para:
                paragraphs.append("\n".join(current_para))
            current_para = [para]
            current_len = para_len

    if current_para:
        paragraphs.append("\n".join(current_para))

    chunk_data_list = []
    chunk_id = 0
    for para in paragraphs:
        chunks = text_splitter.split_text(para)
        for chunk in chunks:
            if len(chunk) < 20:
                continue
            chunk_data_list.append({
                "id": f'chunk{chunk_id}',
                "chunk": chunk,
                "method": "semantic_chunk"
            })
            chunk_id += 1
    return chunk_data_list

# 构建Faiss索引
def build_faiss_index(vector_file, index_path, metadata_path):
    try:
        with open(vector_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            raise ValueError("向量数据为空，请检查输入文件。")
            
        # 确认所有数据项都有向量
        valid_data = []
        for item in data:
            if 'vector' in item and item['vector']:
                valid_data.append(item)
            else:
                print(f"警告: 跳过没有向量的数据项 ID: {item.get('id', '未知')}")
                
        if not valid_data:
            raise ValueError("没有找到任何有效的向量数据。")
            
        # 提取向量
        vectors = [item['vector'] for item in valid_data]
        vectors = np.array(vectors, dtype=np.float32)
        
        if vectors.size == 0:
            raise ValueError("向量数组为空，转换失败。")
            
        # 检查向量维度
        dim = vectors.shape[1]
        n_vectors = vectors.shape[0]
        print(f"构建索引: {n_vectors} 个向量，每个向量维度: {dim}")
        
        # 确定索引类型和参数
        max_nlist = n_vectors // 39
        nlist = min(max_nlist, 128) if max_nlist >= 1 else 1

        if nlist >= 1 and n_vectors >= nlist * 39:
            print(f"使用 IndexIVFFlat 索引，nlist={nlist}")
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            if not index.is_trained:
                index.train(vectors)
            index.add(vectors)
        else:
            print(f"使用 IndexFlatIP 索引")
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)

        faiss.write_index(index, index_path)
        print(f"成功写入索引到 {index_path}")
        
        # 创建元数据
        metadata = [{'id': item['id'], 'chunk': item['chunk'], 'method': item['method']} for item in valid_data]
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        print(f"成功写入元数据到 {metadata_path}")
        
        return True
    except Exception as e:
        print(f"构建索引失败: {str(e)}")
        traceback.print_exc()
        raise

# 向量化文件内容
def vectorize_file(data_list, output_file_path, field_name="chunk"):
    """向量化文件内容，处理长度限制并确保输入有效"""
    if not data_list:
        print("警告: 没有数据需要向量化")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump([], outfile, ensure_ascii=False, indent=4)
        return
        
    # 准备查询文本，确保每个文本有效且长度适中
    valid_data = []
    valid_texts = []
    
    for data in data_list:
        text = data.get(field_name, "")
        # 确保文本不为空且长度合适
        if text and 1 <= len(text) <= 8000:  # 略小于API限制的8192，留出一些余量
            valid_data.append(data)
            valid_texts.append(text)
        else:
            # 如果文本太长，截断它
            if len(text) > 8000:
                truncated_text = text[:8000]
                print(f"警告: 文本过长，已截断至8000字符。原始长度: {len(text)}")
                data[field_name] = truncated_text
                valid_data.append(data)
                valid_texts.append(truncated_text)
            else:
                print(f"警告: 跳过空文本或长度为0的文本")
    
    if not valid_texts:
        print("错误: 所有文本都无效，无法进行向量化")
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump([], outfile, ensure_ascii=False, indent=4)
        return
    
    # 向量化有效文本
    vectors = vectorize_query(valid_texts)
    
    # 检查向量化是否成功
    if vectors.size == 0 or len(vectors) != len(valid_data):
        print(f"错误: 向量化失败或向量数量({len(vectors) if vectors.size > 0 else 0})与数据条目({len(valid_data)})不匹配")
        # 保存原始数据，但不含向量
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(valid_data, outfile, ensure_ascii=False, indent=4)
        return
    
    # 添加向量到数据中
    for data, vector in zip(valid_data, vectors):
        data['vector'] = vector.tolist()
    
    # 保存结果
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(valid_data, outfile, ensure_ascii=False, indent=4)
    
    print(f"成功向量化 {len(valid_data)} 条数据并保存到 {output_file_path}")


# 向量化查询 - 通用函数，被多处使用
def vectorize_query(query, model_name=Config.model_name, batch_size=Config.batch_size) -> np.ndarray:
    """向量化文本查询，返回嵌入向量，改进错误处理和批处理"""
    embedding_client = OpenAI(
        api_key=Config.api_key,
        base_url=Config.base_url
    )
    
    if not query:
        print("警告: 传入向量化的查询为空")
        return np.array([])
        
    if isinstance(query, str):
        query = [query]
    
    # 验证所有查询文本，确保它们符合API要求
    valid_queries = []
    for q in query:
        if not q or not isinstance(q, str):
            print(f"警告: 跳过无效查询: {type(q)}")
            continue
            
        # 清理文本并检查长度
        clean_q = clean_text(q)
        if not clean_q:
            print("警告: 清理后的查询文本为空")
            continue
            
        # 检查长度是否在API限制范围内
        if len(clean_q) > 8000:
            print(f"警告: 查询文本过长 ({len(clean_q)} 字符)，截断至 8000 字符")
            clean_q = clean_q[:8000]
        
        valid_queries.append(clean_q)
    
    if not valid_queries:
        print("错误: 所有查询都无效，无法进行向量化")
        return np.array([])
    
    # 分批处理有效查询
    all_vectors = []
    for i in range(0, len(valid_queries), batch_size):
        batch = valid_queries[i:i + batch_size]
        try:
            # 记录批次信息便于调试
            print(f"正在向量化批次 {i//batch_size + 1}/{(len(valid_queries)-1)//batch_size + 1}, "
                  f"包含 {len(batch)} 个文本，第一个文本长度: {len(batch[0][:50])}...")
                  
            completion = embedding_client.embeddings.create(
                model=model_name,
                input=batch,
                dimensions=Config.dimensions,
                encoding_format="float"
            )
            vectors = [embedding.embedding for embedding in completion.data]
            all_vectors.extend(vectors)
            print(f"批次 {i//batch_size + 1} 向量化成功，获得 {len(vectors)} 个向量")
        except Exception as e:
            print(f"向量化批次 {i//batch_size + 1} 失败：{str(e)}")
            print(f"问题批次中的第一个文本: {batch[0][:100]}...")
            traceback.print_exc()
            # 如果是第一批就失败，直接返回空数组
            if i == 0:
                return np.array([])
            # 否则返回已处理的向量
            break
    
    # 检查是否获得了任何向量
    if not all_vectors:
        print("错误: 向量化过程没有产生任何向量")
        return np.array([])
        
    return np.array(all_vectors)

# 简单的向量搜索，用于基本对比
def vector_search(query, index_path, metadata_path, limit):
    """基本向量搜索函数"""
    query_vector = vectorize_query(query)
    if query_vector.size == 0:
        return []
        
    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

    index = faiss.read_index(index_path)
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except UnicodeDecodeError:
        print(f"警告：{metadata_path} 包含非法字符，使用 UTF-8 忽略错误重新加载")
        with open(metadata_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            metadata = json.loads(content)

    D, I = index.search(query_vector, limit)
    results = [metadata[i] for i in I[0] if i < len(metadata)]
    return results

def clean_text(text):
    """清理文本中的非法字符，控制文本长度"""
    if not text:
        return ""
    # 移除控制字符，保留换行和制表符
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # 移除重复的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 确保文本长度在合理范围内
    return text.strip()

# PDF文本提取
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text()
            # 清理不可打印字符，尝试用 UTF-8 解码，失败时忽略非法字符
            text += page_text.encode('utf-8', errors='ignore').decode('utf-8')
        if not text.strip():
            print(f"警告：PDF文件 {pdf_path} 提取内容为空")
        return text
    except Exception as e:
        print(f"PDF文本提取失败：{str(e)}")
        return ""

# 处理单个文件
def process_single_file(file_path: str) -> str:
    try:
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            if not text:
                return f"PDF文件 {file_path} 内容为空或无法提取"
        else:
            with open(file_path, "rb") as f:
                content = f.read()
            result = chardet.detect(content)
            detected_encoding = result['encoding']
            confidence = result['confidence']
            
            # 尝试多种编码方式
            if detected_encoding and confidence > 0.7:
                try:
                    text = content.decode(detected_encoding)
                    print(f"文件 {file_path} 使用检测到的编码 {detected_encoding} 解码成功")
                except UnicodeDecodeError:
                    text = content.decode('utf-8', errors='ignore')
                    print(f"文件 {file_path} 使用 {detected_encoding} 解码失败，强制使用 UTF-8 忽略非法字符")
            else:
                # 尝试多种常见编码
                encodings = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin-1', 'utf-16', 'cp936', 'big5']
                text = None
                for encoding in encodings:
                    try:
                        text = content.decode(encoding)
                        print(f"文件 {file_path} 使用 {encoding} 解码成功")
                        break
                    except UnicodeDecodeError:
                        continue
                
                # 如果所有编码都失败，使用忽略错误的方式解码
                if text is None:
                    text = content.decode('utf-8', errors='ignore')
                    print(f"警告：文件 {file_path} 使用 UTF-8 忽略非法字符")
        
        # 确保文本是干净的，移除非法字符
        text = clean_text(text)
        return text
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        traceback.print_exc()
        return f"处理文件 {file_path} 失败：{str(e)}"

# 批量处理并索引文件 - 修改为支持指定知识库
def process_and_index_files(file_objs: List, kb_name: str = DEFAULT_KB) -> str:
    """处理并索引文件到指定的知识库"""
    # 确保知识库目录存在
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    os.makedirs(kb_dir, exist_ok=True)
    
    # 设置临时处理文件路径
    semantic_chunk_output = os.path.join(OUTPUT_DIR, f"{kb_name}/semantic_chunk_output.json")
    semantic_chunk_vector = os.path.join(OUTPUT_DIR, f"{kb_name}/semantic_chunk_vector.json")
    
    # 设置知识库索引文件路径
    semantic_chunk_index = os.path.join(kb_dir, "semantic_chunk.index")
    semantic_chunk_metadata = os.path.join(kb_dir, "semantic_chunk_metadata.json")

    all_chunks = []
    error_messages = []
    try:
        if not file_objs or len(file_objs) == 0:
            return "错误：没有选择任何文件"
            
        print(f"开始处理 {len(file_objs)} 个文件，目标知识库: {kb_name}...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_single_file, file_obj.name): file_obj for file_obj in file_objs}
            for future in as_completed(future_to_file):
                result = future.result()
                file_obj = future_to_file[future]
                file_name = file_obj.name
                
                if isinstance(result, str) and result.startswith("处理文件"):
                    error_messages.append(result)
                    print(result)
                    continue
                
                # 检查结果是否为有效文本
                if not result or not isinstance(result, str) or len(result.strip()) == 0:
                    error_messages.append(f"文件 {file_name} 处理后内容为空")
                    print(f"警告: 文件 {file_name} 处理后内容为空")
                    continue
                    
                print(f"对文件 {file_name} 进行语义分块...")
                chunks = semantic_chunk(result)
                
                if not chunks or len(chunks) == 0:
                    error_messages.append(f"文件 {file_name} 无法生成任何分块")
                    print(f"警告: 文件 {file_name} 无法生成任何分块")
                    continue
                
                # 将处理后的文件保存到知识库目录
                file_basename = os.path.basename(file_name)
                dest_file_path = os.path.join(kb_dir, file_basename)
                try:
                    shutil.copy2(file_name, dest_file_path)
                    print(f"已将文件 {file_basename} 复制到知识库 {kb_name}")
                except Exception as e:
                    print(f"复制文件到知识库失败: {str(e)}")
                
                all_chunks.extend(chunks)
                print(f"文件 {file_name} 处理完成，生成 {len(chunks)} 个分块")

        if not all_chunks:
            return "所有文件处理失败或内容为空\n" + "\n".join(error_messages)

        # 确保分块内容干净且长度合适
        valid_chunks = []
        for chunk in all_chunks:
            # 深度清理文本
            clean_chunk_text = clean_text(chunk["chunk"])
            
            # 检查清理后的文本是否有效
            if clean_chunk_text and 1 <= len(clean_chunk_text) <= 8000:
                chunk["chunk"] = clean_chunk_text
                valid_chunks.append(chunk)
            elif len(clean_chunk_text) > 8000:
                # 如果文本太长，截断它
                chunk["chunk"] = clean_chunk_text[:8000]
                valid_chunks.append(chunk)
                print(f"警告: 分块 {chunk['id']} 过长已被截断")
            else:
                print(f"警告: 跳过无效分块 {chunk['id']}")

        if not valid_chunks:
            return "所有生成的分块内容无效或为空\n" + "\n".join(error_messages)
            
        print(f"处理了 {len(all_chunks)} 个分块，有效分块数: {len(valid_chunks)}")

        # 保存语义分块
        with open(semantic_chunk_output, 'w', encoding='utf-8') as json_file:
            json.dump(valid_chunks, json_file, ensure_ascii=False, indent=4)
        print(f"语义分块完成: {semantic_chunk_output}")

        # 向量化语义分块
        print(f"开始向量化 {len(valid_chunks)} 个分块...")
        vectorize_file(valid_chunks, semantic_chunk_vector)
        print(f"语义分块向量化完成: {semantic_chunk_vector}")

        # 验证向量文件是否有效
        try:
            with open(semantic_chunk_vector, 'r', encoding='utf-8') as f:
                vector_data = json.load(f)
                
            if not vector_data or len(vector_data) == 0:
                return f"向量化失败: 生成的向量文件为空\n" + "\n".join(error_messages)
                
            # 检查向量数据结构
            if 'vector' not in vector_data[0]:
                return f"向量化失败: 数据中缺少向量字段\n" + "\n".join(error_messages)
                
            print(f"成功生成 {len(vector_data)} 个向量")
        except Exception as e:
            return f"读取向量文件失败: {str(e)}\n" + "\n".join(error_messages)

        # 构建索引
        print(f"开始为知识库 {kb_name} 构建索引...")
        build_faiss_index(semantic_chunk_vector, semantic_chunk_index, semantic_chunk_metadata)
        print(f"知识库 {kb_name} 索引构建完成: {semantic_chunk_index}")

        status = f"知识库 {kb_name} 更新成功！共处理 {len(valid_chunks)} 个有效分块。\n"
        if error_messages:
            status += "以下文件处理过程中出现问题：\n" + "\n".join(error_messages)
        return status
    except Exception as e:
        error = f"知识库 {kb_name} 索引构建过程中出错：{str(e)}"
        print(error)
        traceback.print_exc()
        return error + "\n" + "\n".join(error_messages)

# 核心联网搜索功能
def get_search_background(query: str, max_length: int = 1500) -> str:
    try:
        from retrievor import q_searching
        search_results = q_searching(query)
        cleaned_results = re.sub(r'\s+', ' ', search_results).strip()
        return cleaned_results[:max_length]
    except Exception as e:
        print(f"联网搜索失败：{str(e)}")
        return ""

# 基本的回答生成
def generate_answer_from_deepseek(question: str, system_prompt: str = "你是一名专业医疗助手，请根据背景知识回答问题。", background_info: Optional[str] = None) -> str:
    deepseek_client = DeepSeekClient()
    user_prompt = f"问题：{question}"
    if background_info:
        user_prompt = f"背景知识：{background_info}\n\n{user_prompt}"
    try:
        answer = deepseek_client.generate_answer(system_prompt, user_prompt)
        return answer
    except Exception as e:
        return f"生成回答时出错：{str(e)}"

# 多跳推理RAG系统 - 核心创新点
class ReasoningRAG:
    """
    多跳推理RAG系统，通过迭代式的检索和推理过程回答问题，支持流式响应
    """
    
    def __init__(self, 
                 index_path: str, 
                 metadata_path: str,
                 max_hops: int = 3,
                 initial_candidates: int = 5,
                 refined_candidates: int = 3,
                 reasoning_model: str = Config.llm_model,
                 verbose: bool = False):
        """
        初始化推理RAG系统
        
        参数:
            index_path: FAISS索引的路径
            metadata_path: 元数据JSON文件的路径
            max_hops: 最大推理-检索跳数
            initial_candidates: 初始检索候选数量
            refined_candidates: 精炼检索候选数量
            reasoning_model: 用于推理步骤的LLM模型
            verbose: 是否打印详细日志
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.max_hops = max_hops
        self.initial_candidates = initial_candidates
        self.refined_candidates = refined_candidates
        self.reasoning_model = reasoning_model
        self.verbose = verbose
        
        # 加载索引和元数据
        self._load_resources()
        
    def _load_resources(self):
        """加载FAISS索引和元数据"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except UnicodeDecodeError:
                with open(self.metadata_path, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                    self.metadata = json.loads(content)
        else:
            raise FileNotFoundError(f"Index or metadata not found at {self.index_path} or {self.metadata_path}")
    
    def _vectorize_query(self, query: str) -> np.ndarray:
        """将查询转换为向量"""
        return vectorize_query(query).reshape(1, -1)
    
    def _retrieve(self, query_vector: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """使用向量相似性检索块"""
        if query_vector.size == 0:
            return []
            
        D, I = self.index.search(query_vector, limit)
        results = [self.metadata[i] for i in I[0] if i < len(self.metadata)]
        return results
    
    def _generate_reasoning(self, 
                           query: str, 
                           retrieved_chunks: List[Dict[str, Any]], 
                           previous_queries: Optional[List[str]] = None,
                           hop_number: int = 0) -> Dict[str, Any]:
        """
        为检索到的信息生成推理分析并识别信息缺口
        
        返回包含以下字段的字典:
            - analysis: 对当前信息的推理分析
            - missing_info: 已识别的缺失信息
            - follow_up_queries: 填补信息缺口的后续查询列表
            - is_sufficient: 表示信息是否足够的布尔值
        """
        if previous_queries is None:
            previous_queries = []
            
        # 为模型准备上下文
        chunks_text = "\n\n".join([f"[Chunk {i+1}]: {chunk['chunk']}" 
                                 for i, chunk in enumerate(retrieved_chunks)])
        
        previous_queries_text = "\n".join([f"Q{i+1}: {q}" for i, q in enumerate(previous_queries)])
        
        system_prompt = """
        你是医疗信息检索的专家分析系统。
        你的任务是分析检索到的信息块，识别缺失的内容，并提出有针对性的后续查询来填补信息缺口。
        
        重点关注医疗领域知识，如:
        - 疾病诊断和症状
        - 治疗方法和药物
        - 医学研究和临床试验
        - 患者护理和康复
        - 医疗法规和伦理
        """
        
        user_prompt = f"""
        ## 原始查询
        {query}
        
        ## 先前查询（如果有）
        {previous_queries_text if previous_queries else "无"}
        
        ## 检索到的信息（跳数 {hop_number}）
        {chunks_text if chunks_text else "未检索到信息。"}
        
        ## 你的任务
        1. 分析已检索到的信息与原始查询的关系
        2. 确定能够更完整回答查询的特定缺失信息
        3. 提出1-3个针对性的后续查询，以检索缺失信息
        4. 确定当前信息是否足够回答原始查询
        
        以JSON格式回答，包含以下字段:
        - analysis: 对当前信息的详细分析
        - missing_info: 特定缺失信息的列表
        - follow_up_queries: 1-3个具体的后续查询
        - is_sufficient: 表示信息是否足够的布尔值
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            reasoning_text = response.choices[0].message.content.strip()
            
            # 解析JSON响应
            try:
                reasoning = json.loads(reasoning_text)
                # 确保预期的键存在
                required_keys = ["analysis", "missing_info", "follow_up_queries", "is_sufficient"]
                for key in required_keys:
                    if key not in reasoning:
                        reasoning[key] = [] if key != "is_sufficient" else False
                return reasoning
            except json.JSONDecodeError:
                # 如果JSON解析失败，则回退
                if self.verbose:
                    print(f"无法从模型输出解析JSON: {reasoning_text[:100]}...")
                return {
                    "analysis": "无法分析检索到的信息。",
                    "missing_info": ["无法识别缺失信息"],
                    "follow_up_queries": [],
                    "is_sufficient": False
                }
                
        except Exception as e:
            if self.verbose:
                print(f"推理生成错误: {e}")
                print(traceback.format_exc())
            return {
                "analysis": "分析过程出错。",
                "missing_info": [],
                "follow_up_queries": [],
                "is_sufficient": False
            }
    
    def _synthesize_answer(self, 
                          query: str, 
                          all_chunks: List[Dict[str, Any]],
                          reasoning_steps: List[Dict[str, Any]],
                          use_table_format: bool = False) -> str:
        """从所有检索到的块和推理步骤中合成最终答案"""
        # 合并所有块，去除重复
        unique_chunks = []
        chunk_ids = set()
        for chunk in all_chunks:
            if chunk["id"] not in chunk_ids:
                unique_chunks.append(chunk)
                chunk_ids.add(chunk["id"])
        
        # 准备上下文
        chunks_text = "\n\n".join([f"[Chunk {i+1}]: {chunk['chunk']}" 
                                  for i, chunk in enumerate(unique_chunks)])
        
        # 准备推理跟踪
        reasoning_trace = ""
        for i, step in enumerate(reasoning_steps):
            reasoning_trace += f"\n\n推理步骤 {i+1}:\n"
            reasoning_trace += f"分析: {step['analysis']}\n"
            reasoning_trace += f"缺失信息: {', '.join(step['missing_info'])}\n"
            reasoning_trace += f"后续查询: {', '.join(step['follow_up_queries'])}"
        
        system_prompt = """
        你是医疗领域的专家。基于检索到的信息块，为用户的查询合成一个全面的答案。
        
        重点提供有关医疗和健康的准确、基于证据的信息，包括诊断、治疗、预防和医学研究等方面。
        
        逻辑地组织你的答案，并在适当时引用块中的具体信息。如果信息不完整，请承认限制。
        """
        
        output_format_instruction = ""
        if use_table_format:
            output_format_instruction = """
            请尽可能以Markdown表格格式组织你的回答。如果信息适合表格形式展示，请使用表格；
            如果不适合表格形式，可以先用文本介绍，然后再使用表格总结关键信息。
            
            表格语法示例：
            | 标题1 | 标题2 | 标题3 |
            | ----- | ----- | ----- |
            | 内容1 | 内容2 | 内容3 |
            
            确保表格格式符合Markdown标准，以便正确渲染。
            """
        
        user_prompt = f"""
        ## 原始查询
        {query}
        
        ## 检索到的信息块
        {chunks_text}
        
        ## 推理过程
        {reasoning_trace}
        
        ## 你的任务
        使用提供的信息块为原始查询合成一个全面的答案。你的答案应该:
        
        1. 直接回应查询
        2. 结构清晰，易于理解
        3. 基于检索到的信息
        4. 承认可用信息中的任何重大缺口
        
        {output_format_instruction}
        
        以直接回应提出原始查询的用户的方式呈现你的答案。
        """
        
        try:
            response = client.chat.completions.create(
                model=Config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"答案合成错误: {e}")
                print(traceback.format_exc())
            return "由于出错，无法生成答案。"
    
    def stream_retrieve_and_answer(self, query: str, use_table_format: bool = False):
        """
        执行多跳检索和回答生成的流式方法，逐步返回结果
        
        这是一个生成器函数，会在处理的每个阶段产生中间结果
        """
        all_chunks = []
        all_queries = [query]
        reasoning_steps = []
        
        # 生成状态更新
        yield {
            "status": "正在将查询向量化...",
            "reasoning_display": "",
            "answer": None,
            "all_chunks": [],
            "reasoning_steps": []
        }
        
        # 初始检索
        try:
            query_vector = self._vectorize_query(query)
            if query_vector.size == 0:
                yield {
                    "status": "向量化失败",
                    "reasoning_display": "由于嵌入错误，无法处理查询。",
                    "answer": "由于嵌入错误，无法处理查询。",
                    "all_chunks": [],
                    "reasoning_steps": []
                }
                return
                
            yield {
                "status": "正在执行初始检索...",
                "reasoning_display": "",
                "answer": None,
                "all_chunks": [],
                "reasoning_steps": []
            }
            
            initial_chunks = self._retrieve(query_vector, self.initial_candidates)
            all_chunks.extend(initial_chunks)
            
            if not initial_chunks:
                yield {
                    "status": "未找到相关信息",
                    "reasoning_display": "未找到与您的查询相关的信息。",
                    "answer": "未找到与您的查询相关的信息。",
                    "all_chunks": [],
                    "reasoning_steps": []
                }
                return
            
            # 更新状态，展示找到的初始块
            chunks_preview = "\n".join([f"- {chunk['chunk'][:100]}..." for chunk in initial_chunks[:2]])
            yield {
                "status": f"找到 {len(initial_chunks)} 个相关信息块，正在生成初步分析...",
                "reasoning_display": f"### 检索到的初始信息\n{chunks_preview}\n\n### 正在分析...",
                "answer": None,
                "all_chunks": all_chunks,
                "reasoning_steps": []
            }
            
            # 初始推理
            reasoning = self._generate_reasoning(query, initial_chunks, hop_number=0)
            reasoning_steps.append(reasoning)
            
            # 生成当前的推理显示
            reasoning_display = "### 多跳推理过程\n"
            reasoning_display += f"**推理步骤 1**\n"
            reasoning_display += f"- 分析: {reasoning['analysis'][:200]}...\n"
            reasoning_display += f"- 缺失信息: {', '.join(reasoning['missing_info'])}\n"
            if reasoning['follow_up_queries']:
                reasoning_display += f"- 后续查询: {', '.join(reasoning['follow_up_queries'])}\n"
            reasoning_display += f"- 信息是否足够: {'是' if reasoning['is_sufficient'] else '否'}\n\n"
            
            yield {
                "status": "初步分析完成",
                "reasoning_display": reasoning_display,
                "answer": None,
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }
            
            # 检查是否需要额外的跳数
            hop = 1
            while (hop < self.max_hops and 
                  not reasoning["is_sufficient"] and 
                  reasoning["follow_up_queries"]):
                
                follow_up_status = f"执行跳数 {hop}，正在处理 {len(reasoning['follow_up_queries'])} 个后续查询..."
                yield {
                    "status": follow_up_status,
                    "reasoning_display": reasoning_display + f"\n\n### {follow_up_status}",
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }
                
                hop_chunks = []
                
                # 处理每个后续查询
                for i, follow_up_query in enumerate(reasoning["follow_up_queries"]):
                    all_queries.append(follow_up_query)
                    
                    query_status = f"处理后续查询 {i+1}/{len(reasoning['follow_up_queries'])}: {follow_up_query}"
                    yield {
                        "status": query_status,
                        "reasoning_display": reasoning_display + f"\n\n### {query_status}",
                        "answer": None,
                        "all_chunks": all_chunks,
                        "reasoning_steps": reasoning_steps
                    }
                    
                    # 为后续查询检索
                    follow_up_vector = self._vectorize_query(follow_up_query)
                    if follow_up_vector.size > 0:
                        follow_up_chunks = self._retrieve(follow_up_vector, self.refined_candidates)
                        hop_chunks.extend(follow_up_chunks)
                        all_chunks.extend(follow_up_chunks)
                        
                        # 更新状态，显示新找到的块数量
                        yield {
                            "status": f"查询 '{follow_up_query}' 找到了 {len(follow_up_chunks)} 个相关块",
                            "reasoning_display": reasoning_display + f"\n\n为查询 '{follow_up_query}' 找到了 {len(follow_up_chunks)} 个相关块",
                            "answer": None,
                            "all_chunks": all_chunks,
                            "reasoning_steps": reasoning_steps
                        }
                
                # 为此跳数生成推理
                yield {
                    "status": f"正在为跳数 {hop} 生成推理分析...",
                    "reasoning_display": reasoning_display + f"\n\n### 正在为跳数 {hop} 生成推理分析...",
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }
                
                reasoning = self._generate_reasoning(
                    query, 
                    hop_chunks, 
                    previous_queries=all_queries[:-1],
                    hop_number=hop
                )
                reasoning_steps.append(reasoning)
                
                # 更新推理显示
                reasoning_display += f"\n**推理步骤 {hop+1}**\n"
                reasoning_display += f"- 分析: {reasoning['analysis'][:200]}...\n"
                reasoning_display += f"- 缺失信息: {', '.join(reasoning['missing_info'])}\n"
                if reasoning['follow_up_queries']:
                    reasoning_display += f"- 后续查询: {', '.join(reasoning['follow_up_queries'])}\n"
                reasoning_display += f"- 信息是否足够: {'是' if reasoning['is_sufficient'] else '否'}\n"
                
                yield {
                    "status": f"跳数 {hop} 完成",
                    "reasoning_display": reasoning_display,
                    "answer": None,
                    "all_chunks": all_chunks,
                    "reasoning_steps": reasoning_steps
                }
                
                hop += 1
            
            # 合成最终答案
            yield {
                "status": "正在合成最终答案...",
                "reasoning_display": reasoning_display + "\n\n### 正在合成最终答案...",
                "answer": "正在处理您的问题，请稍候...",
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }
            
            answer = self._synthesize_answer(query, all_chunks, reasoning_steps, use_table_format)
            
            # 为最终显示准备检索内容汇总
            all_chunks_summary = "\n\n".join([f"**检索块 {i+1}**:\n{chunk['chunk']}" 
                                           for i, chunk in enumerate(all_chunks[:10])])  # 限制显示前10个块
            
            if len(all_chunks) > 10:
                all_chunks_summary += f"\n\n...以及另外 {len(all_chunks) - 10} 个块（总计 {len(all_chunks)} 个）"
                
            enhanced_display = reasoning_display + "\n\n### 检索到的内容\n" + all_chunks_summary + "\n\n### 回答已生成"
            
            yield {
                "status": "回答已生成",
                "reasoning_display": enhanced_display,
                "answer": answer,
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }
            
        except Exception as e:
            error_msg = f"处理过程中出错: {str(e)}"
            if self.verbose:
                print(error_msg)
                print(traceback.format_exc())
            
            yield {
                "status": "处理出错",
                "reasoning_display": error_msg,
                "answer": f"处理您的问题时遇到错误: {str(e)}",
                "all_chunks": all_chunks,
                "reasoning_steps": reasoning_steps
            }
    
    def retrieve_and_answer(self, query: str, use_table_format: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        执行多跳检索和回答生成的主要方法
        
        返回:
            包含以下内容的元组:
            - 最终答案
            - 包含推理步骤和所有检索到的块的调试字典
        """
        all_chunks = []
        all_queries = [query]
        reasoning_steps = []
        debug_info = {"reasoning_steps": [], "all_chunks": [], "all_queries": all_queries}
        
        # 初始检索
        query_vector = self._vectorize_query(query)
        if query_vector.size == 0:
            return "由于嵌入错误，无法处理查询。", debug_info
            
        initial_chunks = self._retrieve(query_vector, self.initial_candidates)
        all_chunks.extend(initial_chunks)
        debug_info["all_chunks"].extend(initial_chunks)
        
        if not initial_chunks:
            return "未找到与您的查询相关的信息。", debug_info
        
        # 初始推理
        reasoning = self._generate_reasoning(query, initial_chunks, hop_number=0)
        reasoning_steps.append(reasoning)
        debug_info["reasoning_steps"].append(reasoning)
        
        # 检查是否需要额外的跳数
        hop = 1
        while (hop < self.max_hops and 
               not reasoning["is_sufficient"] and 
               reasoning["follow_up_queries"]):
            
            if self.verbose:
                print(f"开始跳数 {hop}，有 {len(reasoning['follow_up_queries'])} 个后续查询")
            
            hop_chunks = []
            
            # 处理每个后续查询
            for follow_up_query in reasoning["follow_up_queries"]:
                all_queries.append(follow_up_query)
                debug_info["all_queries"].append(follow_up_query)
                
                # 为后续查询检索
                follow_up_vector = self._vectorize_query(follow_up_query)
                if follow_up_vector.size > 0:
                    follow_up_chunks = self._retrieve(follow_up_vector, self.refined_candidates)
                    hop_chunks.extend(follow_up_chunks)
                    all_chunks.extend(follow_up_chunks)
                    debug_info["all_chunks"].extend(follow_up_chunks)
            
            # 为此跳数生成推理
            reasoning = self._generate_reasoning(
                query, 
                hop_chunks, 
                previous_queries=all_queries[:-1],
                hop_number=hop
            )
            reasoning_steps.append(reasoning)
            debug_info["reasoning_steps"].append(reasoning)
            
            hop += 1
        
        # 合成最终答案
        answer = self._synthesize_answer(query, all_chunks, reasoning_steps, use_table_format)
        
        return answer, debug_info

# 基于选定知识库生成索引路径
def get_kb_paths(kb_name: str) -> Dict[str, str]:
    """获取指定知识库的索引文件路径"""
    kb_dir = os.path.join(KB_BASE_DIR, kb_name)
    return {
        "index_path": os.path.join(kb_dir, "semantic_chunk.index"),
        "metadata_path": os.path.join(kb_dir, "semantic_chunk_metadata.json")
    }

def multi_hop_generate_answer(query: str, kb_name: str, use_table_format: bool = False, system_prompt: str = "你是一名医疗专家。") -> Tuple[str, Dict]:
    """使用多跳推理RAG生成答案，基于指定知识库"""
    kb_paths = get_kb_paths(kb_name)
    
    reasoning_rag = ReasoningRAG(
        index_path=kb_paths["index_path"],
        metadata_path=kb_paths["metadata_path"],
        max_hops=3,
        initial_candidates=5,
        refined_candidates=3,
        reasoning_model=Config.llm_model,
        verbose=True
    )
    
    answer, debug_info = reasoning_rag.retrieve_and_answer(query, use_table_format)
    return answer, debug_info

# 使用简单向量检索生成答案，基于指定知识库
def simple_generate_answer(query: str, kb_name: str, use_table_format: bool = False) -> str:
    """使用简单的向量检索生成答案，不使用多跳推理"""
    try:
        kb_paths = get_kb_paths(kb_name)
        
        # 使用基本向量搜索
        search_results = vector_search(query, kb_paths["index_path"], kb_paths["metadata_path"], limit=5)
        
        if not search_results:
            return "未找到相关信息。"
        
        # 准备背景信息
        background_chunks = "\n\n".join([f"[相关信息 {i+1}]: {result['chunk']}" 
                                       for i, result in enumerate(search_results)])
        
        # 生成答案
        system_prompt = "你是一名医疗专家。基于提供的背景信息回答用户的问题。"
        
        if use_table_format:
            system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
        
        user_prompt = f"""
        问题：{query}
        
        背景信息：
        {background_chunks}
        
        请基于以上背景信息回答用户的问题。
        """
        
        response = client.chat.completions.create(
            model=Config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"生成答案时出错：{str(e)}"

# 修改主要的问题处理函数以支持指定知识库
def ask_question_parallel(
    question: str, 
    kb_name: str = DEFAULT_KB,
    system_prompt:str = "你是一名医疗专家，请整合网络搜索和本地知识库提供全面的解答。", 
    use_search: bool = True, 
    use_table_format: bool = False, 
    multi_hop: bool = False,
    model=Config.llm_model,
    IsRagAnswer=True) -> str:   
    """基于指定知识库回答问题"""
    if not IsRagAnswer:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
        )
        
        return response.choices[0].message.content.strip()
    else:        
        try:
            kb_paths = get_kb_paths(kb_name)
            index_path = kb_paths["index_path"]
            metadata_path = kb_paths["metadata_path"]

            search_background = ""
            local_answer = ""
            debug_info = {}

            # 并行处理
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                
                if use_search:
                    futures[executor.submit(get_search_background, question)] = "search"
                    
                if os.path.exists(index_path):
                    if multi_hop:
                        # 使用多跳推理
                        futures[executor.submit(multi_hop_generate_answer, question, kb_name, use_table_format)] = "rag"
                    else:
                        # 使用简单向量检索
                        futures[executor.submit(simple_generate_answer, question, kb_name, use_table_format)] = "simple"
                    
                for future in as_completed(futures):
                    result = future.result()
                    if futures[future] == "search":
                        search_background = result or ""
                    elif futures[future] == "rag":
                        local_answer, debug_info = result
                    elif futures[future] == "simple":
                        local_answer = result

            # 如果同时有搜索和本地结果，合并它们
            if search_background and local_answer:            
                table_instruction = ""
                if use_table_format:
                    table_instruction = """
                    请尽可能以Markdown表格的形式呈现你的回答，特别是对于症状、治疗方法、药物等结构化信息。
                    
                    请确保你的表格遵循正确的Markdown语法：
                    | 列标题1 | 列标题2 | 列标题3 |
                    | ------- | ------- | ------- |
                    | 数据1   | 数据2   | 数据3   |
                    """
                    
                user_prompt = f"""
                问题：{question}
                
                网络搜索结果：{search_background}
                
                本地知识库分析：{local_answer}
                
                {table_instruction}
                
                请根据以上信息，提供一个综合的回答。
                """
                
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    combined_answer = response.choices[0].message.content.strip()
                    return combined_answer
                except Exception as e:
                    # 如果合并失败，回退到本地答案
                    return local_answer
            elif local_answer:
                return local_answer
            elif search_background:
                # 仅从搜索结果生成答案
                system_prompt = system_prompt
                if use_table_format:
                    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
                return generate_answer_from_deepseek(question, system_prompt=system_prompt, background_info=f"[联网搜索结果]：{search_background}")
            else:
                return "未找到相关信息。"
                
        except Exception as e:
            return f"查询失败：{str(e)}"

# 修改以支持多知识库的流式响应函数
def process_question_with_reasoning(
    question: str, 
    kb_name: str = DEFAULT_KB, 
    use_search: bool = True, 
    use_table_format: bool = False, 
    multi_hop: bool = False,  
    chat_history: Optional[List] = None,
    model=Config.llm_model,
    system_prompt:str="你是一名医疗专家。请考虑对话历史并回答用户的问题") -> Generator[Tuple[str, str], None, None]:
    """增强版process_question，支持流式响应，实时显示检索和推理过程，支持多知识库和对话历史"""
    if chat_history is None:
        chat_history = []   
    try:
        kb_paths = get_kb_paths(kb_name)
        index_path = kb_paths["index_path"]
        metadata_path = kb_paths["metadata_path"]

        # 构建带对话历史的问题
        if chat_history and len(chat_history) > 0:
            # 构建对话上下文
            context = "之前的对话内容：\n"
            for user_msg, assistant_msg in chat_history[-3:]:  # 只取最近3轮对话
                context += f"用户：{user_msg}\n"
                context += f"助手：{assistant_msg}\n"
            context += f"\n当前问题：{question}"
            enhanced_question = f"基于以下对话历史，回答用户的当前问题。\n{context}"
        else:
            enhanced_question = question

        # 初始状态
        search_result = "联网搜索进行中..." if use_search else "未启用联网搜索"
        
        if multi_hop:
            reasoning_status = f"正在准备对知识库 '{kb_name}' 进行多跳推理检索..."
            search_display = f"### 联网搜索结果\n{search_result}\n\n### 推理状态\n{reasoning_status}"
            yield search_display, "正在启动多跳推理流程..."
        else:
            reasoning_status = f"正在准备对知识库 '{kb_name}' 进行向量检索..."
            search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n{reasoning_status}"
            yield search_display, "正在启动简单检索流程..."

        # 如果启用，并行运行搜索
        search_future = None
        with ThreadPoolExecutor(max_workers=1) as executor:
            if use_search:
                search_future = executor.submit(get_search_background, question)
                
        # 检查索引是否存在
        if not (os.path.exists(index_path) and os.path.exists(metadata_path)):
            # 如果索引不存在，提前返回
            if search_future:
                # 等待搜索结果
                search_result = "等待联网搜索结果..."
                search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n知识库 '{kb_name}' 中未找到索引"
                yield search_display, "等待联网搜索结果..."
                
                search_result = search_future.result() or "未找到相关网络信息"
                if use_table_format:
                    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
                answer = generate_answer_from_deepseek(enhanced_question, system_prompt=system_prompt, background_info=f"[联网搜索结果]：{search_result}")
                
                search_display = f"### 联网搜索结果\n{search_result}\n\n### 检索状态\n无法在知识库 '{kb_name}' 中进行本地检索（未找到索引）"
                yield search_display, answer
            else:
                yield f"知识库 '{kb_name}' 中未找到索引，且未启用联网搜索", "无法回答您的问题。请先上传文件到该知识库或启用联网搜索。"
            return

        # 开始流式处理
        current_answer = "正在分析您的问题..."
        
        if multi_hop:
            # 使用多跳推理的流式接口
            reasoning_rag = ReasoningRAG(
                index_path=index_path,
                metadata_path=metadata_path,
                max_hops=3,
                initial_candidates=5,
                refined_candidates=3,
                verbose=True
            )
            
            # 使用enhanced_question进行检索
            for step_result in reasoning_rag.stream_retrieve_and_answer(enhanced_question, use_table_format):
                # 更新当前状态
                status = step_result["status"]
                reasoning_display = step_result["reasoning_display"]
                
                # 如果有新的答案，更新
                if step_result["answer"]:
                    current_answer = step_result["answer"]
                
                # 如果搜索结果已返回，更新搜索结果
                if search_future and search_future.done():
                    search_result = search_future.result() or "未找到相关网络信息"
                
                # 构建并返回当前状态
                current_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 推理状态\n{status}\n\n{reasoning_display}"
                yield current_display, current_answer
        else:
            # 简单向量检索的流式处理
            yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n正在执行向量相似度搜索...", "正在检索相关信息..."
            
            # 执行简单向量搜索，使用enhanced_question
            try:
                search_results = vector_search(enhanced_question, index_path, metadata_path, limit=5)
                
                if not search_results:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n未找到相关信息", f"知识库 '{kb_name}' 中未找到相关信息。"
                    current_answer = f"知识库 '{kb_name}' 中未找到相关信息。"
                else:
                    # 显示检索到的信息
                    chunks_detail = "\n\n".join([f"**相关信息 {i+1}**:\n{result['chunk']}" for i, result in enumerate(search_results[:5])])
                    chunks_preview = "\n".join([f"- {result['chunk'][:100]}..." for i, result in enumerate(search_results[:3])])
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n找到 {len(search_results)} 个相关信息块\n\n### 检索到的信息预览\n{chunks_preview}", "正在生成答案..."
                    
                    # 生成答案
                    background_chunks = "\n\n".join([f"[相关信息 {i+1}]: {result['chunk']}" 
                                                   for i, result in enumerate(search_results)])
                    if use_table_format:
                        system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
                    
                    user_prompt = f"""
                    {enhanced_question}
                    
                    背景信息：
                    {background_chunks}
                    
                    请基于以上背景信息和对话历史回答用户的问题。
                    """
                    
                    response = client.chat.completions.create(
                        model=Config.llm_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    
                    current_answer = response.choices[0].message.content.strip()
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n检索完成，已生成答案\n\n### 检索到的内容\n{chunks_detail}", current_answer
                    
            except Exception as e:
                error_msg = f"检索过程中出错: {str(e)}"
                yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n{error_msg}", f"检索过程中出错: {str(e)}"
                current_answer = f"检索过程中出错: {str(e)}"
        
        # 检索完成后，如果有搜索结果，可以考虑合并知识
        if search_future and search_future.done():
            search_result = search_future.result() or "未找到相关网络信息"
            
            # 如果同时有搜索结果和本地检索结果，可以考虑合并
            if search_result and current_answer and current_answer not in ["正在分析您的问题...", "本地知识库中未找到相关信息。"]:
                status_text = "正在合并联网搜索和知识库结果..."
                if multi_hop:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 推理状态\n{status_text}", current_answer
                else:
                    yield f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 检索状态\n{status_text}", current_answer
                
                # 合并结果
                system_prompt = "你是一名医疗专家，请整合网络搜索和本地知识库提供全面的解答。请考虑对话历史。"
                
                if use_table_format:
                    system_prompt += "请尽可能以Markdown表格的形式呈现结构化信息。"
                
                user_prompt = f"""
                {enhanced_question}
                
                网络搜索结果：{search_result}
                
                本地知识库分析：{current_answer}
                
                请根据以上信息和对话历史，提供一个综合的回答。确保使用Markdown表格来呈现适合表格形式的信息。
                """
                
                try:
                    response = client.chat.completions.create(
                        model="qwen-plus",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                    combined_answer = response.choices[0].message.content.strip()
                    
                    final_status = "已整合联网和知识库结果"
                    if multi_hop:
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成多跳推理分析，检索到的内容已在上方显示\n\n### 综合分析\n{final_status}"
                    else:
                        # 获取之前检索到的内容
                        chunks_info = "".join([part.split("### 检索到的内容\n")[-1] if "### 检索到的内容\n" in part else "" for part in search_display.split("### 联网搜索结果")])
                        if not chunks_info.strip():
                            chunks_info = "检索内容已在上方显示"
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成向量检索分析\n\n### 检索到的内容\n{chunks_info}\n\n### 综合分析\n{final_status}"
                    
                    yield final_display, combined_answer
                except Exception as e:
                    # 如果合并失败，使用现有答案
                    error_status = f"合并结果失败: {str(e)}"
                    if multi_hop:
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成多跳推理分析，检索到的内容已在上方显示\n\n### 综合分析\n{error_status}"
                    else:
                        # 获取之前检索到的内容
                        chunks_info = "".join([part.split("### 检索到的内容\n")[-1] if "### 检索到的内容\n" in part else "" for part in search_display.split("### 联网搜索结果")])
                        if not chunks_info.strip():
                            chunks_info = "检索内容已在上方显示"
                        final_display = f"### 联网搜索结果\n{search_result}\n\n### 知识库: {kb_name}\n### 本地知识库分析\n已完成向量检索分析\n\n### 检索到的内容\n{chunks_info}\n\n### 综合分析\n{error_status}"
                        
                    yield final_display, current_answer
        
    except Exception as e:
        error_msg = f"处理失败：{str(e)}\n{traceback.format_exc()}"
        yield f"### 错误信息\n{error_msg}", f"处理您的问题时遇到错误：{str(e)}"

# 添加处理函数，批量上传文件到指定知识库  
def batch_upload_to_kb(file_objs: List, kb_name: str) -> str:
    """批量上传文件到指定知识库并进行处理"""
    try:
        if not kb_name or not kb_name.strip():
            return "错误：未指定知识库"
            
        # 确保知识库目录存在
        kb_dir = os.path.join(KB_BASE_DIR, kb_name)
        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir, exist_ok=True)
            
        if not file_objs or len(file_objs) == 0:
            return "错误：未选择任何文件"
            
        return process_and_index_files(file_objs, kb_name)
    except Exception as e:
        return f"上传文件到知识库失败: {str(e)}"