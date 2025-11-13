"""RAG模块初始化文件"""

from .rag import (
    get_knowledge_bases,
    create_knowledge_base,
    update_knowledge_base,
    delete_knowledge_base,
    get_kb_files,
    ask_question_parallel,
    process_and_index_files,
    batch_upload_to_kb,
    simple_generate_answer,
    multi_hop_generate_answer,
    get_kb_paths,

)
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

__all__ = [
    'get_knowledge_bases',
    'create_knowledge_base',
    'update_knowledge_base',
    'delete_knowledge_base',
    'get_kb_files',
    'ask_question_parallel',
    'process_and_index_files',
    "batch_upload_to_kb",
    'simple_generate_answer',
    'multi_hop_generate_answer',
    'get_kb_paths',

]