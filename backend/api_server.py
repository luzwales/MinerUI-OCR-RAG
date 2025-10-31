"""
MinerU Tianshu - API Server
天枢 API 服务器

企业级 AI 数据预处理平台
支持文档、图片、音频、视频等多模态数据处理
提供 RESTful API 接口用于任务提交、查询和管理
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from pathlib import Path
from loguru import logger
import uvicorn
from typing import Optional
from datetime import datetime
import os
import sys
import re
import uuid
from minio import Minio
import RAG.rag as rag  # 新增：导入 RAG 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'rag')))
from config import Config  # 导入配置文件

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import uvicorn
from decouple import config as env_config


from task_db import TaskDB

# 初始化 FastAPI 应用
app = FastAPI(
    title="Flex AI API",
    description="Flex MinerU Tianshu - AI data preprocessing platform for documents, images, audio, and video.",
    version="1.0.0",
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
db = TaskDB()

# 配置输出目录
OUTPUT_DIR = Path(Config.ocr_output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MinIO 配置
MINIO_CONFIG = {
    "endpoint": os.getenv("MINIO_ENDPOINT", ""),
    "access_key": os.getenv("MINIO_ACCESS_KEY", ""),
    "secret_key": os.getenv("MINIO_SECRET_KEY", ""),
    "secure": True,
    "bucket_name": os.getenv("MINIO_BUCKET", ""),
}


def get_minio_client():
    """获取MinIO客户端实例"""
    return Minio(
        MINIO_CONFIG["endpoint"],
        access_key=MINIO_CONFIG["access_key"],
        secret_key=MINIO_CONFIG["secret_key"],
        secure=MINIO_CONFIG["secure"],
    )


def process_markdown_images(md_content: str, image_dir: Path, upload_images: bool = False):
    """
    处理 Markdown 中的图片引用

    Args:
        md_content: Markdown 内容
        image_dir: 图片所在目录
        upload_images: 是否上传图片到 MinIO 并替换链接

    Returns:
        处理后的 Markdown 内容
    """
    if not upload_images:
        return md_content

    try:
        minio_client = get_minio_client()
        bucket_name = MINIO_CONFIG["bucket_name"]
        minio_endpoint = MINIO_CONFIG["endpoint"]

        # 查找所有 markdown 格式的图片
        img_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"

        def replace_image(match):
            alt_text = match.group(1)
            image_path = match.group(2)

            # 构建完整的本地图片路径
            full_image_path = image_dir / Path(image_path).name

            if full_image_path.exists():
                # 获取文件后缀
                file_extension = full_image_path.suffix
                # 生成 UUID 作为新文件名
                new_filename = f"{uuid.uuid4()}{file_extension}"

                try:
                    # 上传到 MinIO
                    object_name = f"images/{new_filename}"
                    minio_client.fput_object(bucket_name, object_name, str(full_image_path))

                    # 生成 MinIO 访问 URL
                    scheme = "https" if MINIO_CONFIG["secure"] else "http"
                    minio_url = f"{scheme}://{minio_endpoint}/{bucket_name}/{object_name}"

                    # 返回 HTML 格式的 img 标签
                    return f'<img src="{minio_url}" alt="{alt_text}">'
                except Exception as e:
                    logger.error(f"Failed to upload image to MinIO: {e}")
                    return match.group(0)  # 上传失败，保持原样

            return match.group(0)

        # 替换所有图片引用
        new_content = re.sub(img_pattern, replace_image, md_content)
        return new_content

    except Exception as e:
        logger.error(f"Error processing markdown images: {e}")
        return md_content  # 出错时返回原内容


@app.get("/")
async def root():
    """API根路径"""
    return {
        "service": "Flex MinerU Tianshu API",
        "version": "1.0.0",
        "description": "Flex MinerU Tianshu - AI data preprocessing platform",
        "features": "Document, Image, Audio, Video processing with OCR and RAG capabilities",
        "docs": "/docs",
    }


@app.post("/api/v1/tasks/submit")
async def submit_task(
    file: UploadFile = File(..., description="Document: PDF/Picture/Office/HTML/Audio/Video ect."),
    backend: str = Form(
        "pipeline", description="Handling: pipeline/deepseek-ocr/paddleocr-vl (Document) | sensevoice (Audio) | video "
    ),
    lang: str = Form("auto", description="Language: auto/ch/en/korean/japan等"),
    method: str = Form("auto", description="example: auto/txt/ocr"),
    formula_enable: bool = Form(True, description="是否启用公式识别"),
    table_enable: bool = Form(True, description="是否启用表格识别"),
    priority: int = Form(0, description="优先级，数字越大越优先"),
    # DeepSeek OCR 专用参数
    deepseek_resolution: str = Form("base", description="DeepSeek OCR 分辨率: tiny/small/base/large/dynamic"),
    deepseek_prompt_type: str = Form("document", description="DeepSeek OCR 提示词类型: document/image/free/figure"),
    # 视频处理专用参数
    keep_audio: bool = Form(False, description="视频处理时是否保留提取的音频文件"),
    enable_keyframe_ocr: bool = Form(False, description="是否启用视频关键帧OCR识别（实验性功能）"),
    ocr_backend: str = Form("paddleocr-vl", description="关键帧OCR引擎: paddleocr-vl/deepseek-ocr"),
    keep_keyframes: bool = Form(False, description="是否保留提取的关键帧图像"),
    # 水印去除专用参数
    remove_watermark: bool = Form(False, description="是否启用水印去除（支持 PDF/图片）"),
    watermark_conf_threshold: float = Form(0.35, description="水印检测置信度阈值（0.0-1.0，推荐 0.35）"),
    watermark_dilation: int = Form(10, description="水印掩码膨胀大小（像素，推荐 10）"),
):
    """
    提交文档解析任务

    立即返回 task_id，任务在后台异步处理
    """
    try:
        # 保存上传的文件到临时目录
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)

        # 流式写入文件到磁盘，避免高内存使用
        while True:
            chunk = await file.read(1 << 23)  # 8MB chunks
            if not chunk:
                break
            temp_file.write(chunk)

        temp_file.close()

        # 创建任务
        task_id = db.create_task(
            file_name=file.filename,
            file_path=temp_file.name,
            backend=backend,
            options={
                "lang": lang,
                "method": method,
                "formula_enable": formula_enable,
                "table_enable": table_enable,
                # DeepSeek OCR 参数
                "deepseek_resolution": deepseek_resolution,
                "deepseek_prompt_type": deepseek_prompt_type,
                # 视频处理参数
                "keep_audio": keep_audio,
                "enable_keyframe_ocr": enable_keyframe_ocr,
                "ocr_backend": ocr_backend,
                "keep_keyframes": keep_keyframes,
                # 水印去除参数
                "remove_watermark": remove_watermark,
                "watermark_conf_threshold": watermark_conf_threshold,
                "watermark_dilation": watermark_dilation,
            },
            priority=priority,
        )

        logger.info(f"✅ Task submitted: {task_id} - {file.filename}")
        logger.info(f"   Backend: {backend}")
        logger.info(f"   Priority: {priority}")
        if backend == "deepseek-ocr":
            logger.info(f"   DeepSeek Resolution: {deepseek_resolution}")
            logger.info(f"   DeepSeek Prompt Type: {deepseek_prompt_type}")

        return {
            "success": True,
            "task_id": task_id,
            "status": "pending",
            "message": "Task submitted successfully",
            "file_name": file.filename,
            "created_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Failed to submit task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    upload_images: bool = Query(False, description="是否上传图片到MinIO并替换链接（仅当任务完成时有效）"),
    format: str = Query("markdown", description="返回格式: markdown(默认)/json/both"),
):
    """
    查询任务状态和详情

    当任务完成时，会自动返回解析后的内容（data 字段）
    - format=markdown: 只返回 Markdown 内容（默认）
    - format=json: 只返回 JSON 结构化数据（MinerU 和 PaddleOCR-VL 支持）
    - format=both: 同时返回 Markdown 和 JSON
    可选择是否上传图片到 MinIO 并替换为 URL
    """
    task = db.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    response = {
        "success": True,
        "task_id": task_id,
        "status": task["status"],
        "file_name": task["file_name"],
        "backend": task["backend"],
        "priority": task["priority"],
        "error_message": task["error_message"],
        "created_at": task["created_at"],
        "started_at": task["started_at"],
        "completed_at": task["completed_at"],
        "worker_id": task["worker_id"],
        "retry_count": task["retry_count"],
    }
    logger.info(f"✅ Task status: {task['status']} - (result_path: {task['result_path']})")

    # 如果任务已完成，尝试返回解析内容
    if task["status"] == "completed":
        if not task["result_path"]:
            # 结果文件已被清理
            response["data"] = None
            response["message"] = "Task completed but result files have been cleaned up (older than retention period)"
            return response

        result_dir = Path(task["result_path"])
        logger.info(f"📂 Checking result directory: {result_dir}")

        if result_dir.exists():
            logger.info("✅ Result directory exists")
            # 递归查找 Markdown 文件（MinerU 输出结构：task_id/filename/auto/*.md）
            md_files = list(result_dir.rglob("*.md"))
            # 递归查找 JSON 文件 (排除调试用的 page_*.json)
            json_files = [
                f
                for f in result_dir.rglob("*.json")
                if not f.parent.name.startswith("page_") and f.name in ["content.json", "result.json"]
            ]
            logger.info(f"📄 Found {len(md_files)} markdown files and {len(json_files)} json files")

            if md_files:
                try:
                    # 初始化 data 字段
                    response["data"] = {}

                    # 标记 JSON 是否可用
                    response["data"]["json_available"] = len(json_files) > 0

                    # 根据 format 参数决定返回内容
                    if format in ["markdown", "both"]:
                        # 读取 Markdown 内容
                        md_file = md_files[0]
                        logger.info(f"📖 Reading markdown file: {md_file}")
                        with open(md_file, "r", encoding="utf-8") as f:
                            md_content = f.read()

                        logger.info(f"✅ Markdown content loaded, length: {len(md_content)} characters")

                        # 查找图片目录（在 markdown 文件的同级目录下）
                        image_dir = md_file.parent / "images"

                        # 处理图片（如果需要）
                        if upload_images and image_dir.exists():
                            logger.info(f"🖼️  Processing images for task {task_id}, upload_images={upload_images}")
                            md_content = process_markdown_images(md_content, image_dir, upload_images)

                        # 添加 Markdown 相关字段
                        response["data"]["markdown_file"] = md_file.name
                        response["data"]["content"] = md_content
                        response["data"]["images_uploaded"] = upload_images
                        response["data"]["has_images"] = image_dir.exists() if not upload_images else None

                    # 如果用户请求 JSON 格式
                    if format in ["json", "both"] and json_files:
                        import json as json_lib

                        json_file = json_files[0]
                        logger.info(f"📖 Reading JSON file: {json_file}")
                        try:
                            with open(json_file, "r", encoding="utf-8") as f:
                                json_content = json_lib.load(f)
                            response["data"]["json_file"] = json_file.name
                            response["data"]["json_content"] = json_content
                            logger.info("✅ JSON content loaded successfully")
                        except Exception as json_e:
                            logger.warning(f"⚠️  Failed to load JSON: {json_e}")
                    elif format == "json" and not json_files:
                        # 用户请求 JSON 但没有 JSON 文件
                        logger.warning("⚠️  JSON format requested but no JSON file available")
                        response["data"]["message"] = "JSON format not available for this backend"

                    # 如果没有返回任何内容，添加提示
                    if not response["data"]:
                        response["data"] = None
                        logger.warning(f"⚠️  No data returned for format: {format}")
                    else:
                        logger.info(f"✅ Response data field added successfully (format={format})")

                except Exception as e:
                    logger.error(f"❌ Failed to read content: {e}")
                    logger.exception(e)
                    # 读取失败不影响状态查询，只是不返回 data
                    response["data"] = None
            else:
                logger.warning(f"⚠️  No markdown files found in {result_dir}")
        else:
            logger.error(f"❌ Result directory does not exist: {result_dir}")
    elif task["status"] == "completed":
        logger.warning("⚠️  Task completed but result_path is empty")
    else:
        logger.info(f"ℹ️  Task status is {task['status']}, skipping content loading")

    return response


@app.delete("/api/v1/tasks/{task_id}")
async def cancel_task(task_id: str):
    """
    取消任务（仅限 pending 状态）
    """
    task = db.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] == "pending":
        db.update_task_status(task_id, "cancelled")

        # 删除临时文件
        file_path = Path(task["file_path"])
        if file_path.exists():
            file_path.unlink()

        logger.info(f"⏹️  Task cancelled: {task_id}")
        return {"success": True, "message": "Task cancelled successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Cannot cancel task in {task['status']} status")


@app.get("/api/v1/queue/stats")
async def get_queue_stats():
    """
    获取队列统计信息
    """
    stats = db.get_queue_stats()

    return {"success": True, "stats": stats, "total": sum(stats.values()), "timestamp": datetime.now().isoformat()}

# -----------------------------
# RAG REST 接口（将 Gradio 替换为 FastAPI 路由）
# -----------------------------
@app.get("/api/kbs")
async def list_kbs():
    """列出所有知识库"""
    try:
        kbs = rag.get_knowledge_bases()
        return {"success": True, "kbs": kbs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kb")
async def create_kb(kb_name: str = Form(...)):
    """创建知识库"""
    try:
        res = rag.create_knowledge_base(kb_name)
        return {"success": True, "message": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/kb/{kb_name}")
async def delete_kb(kb_name: str):
    """删除知识库"""
    try:
        res = rag.delete_knowledge_base(kb_name)
        return {"success": True, "message": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kb/{kb_name}/files")
async def list_kb_files(kb_name: str):
    """列出指定知识库中的文件和索引状态"""
    try:
        files = rag.get_kb_files(kb_name)
        kb_dir = os.path.join(rag.KB_BASE_DIR, kb_name)
        has_index = os.path.exists(os.path.join(kb_dir, "semantic_chunk.index"))
        return {"success": True, "files": files, "has_index": has_index}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/kb/{kb_name}/upload")
async def upload_files_to_kb(kb_name: str, files: List[UploadFile] = File(...)):
    """
    上传多个文件到指定知识库并进行处理（支持 .txt 和 .pdf）
    文件会被暂存为临时文件，传给 RAG 模块处理，处理完成后临时文件会被清理。
    """
    tmp_paths = []
    try:
        if not files:
            raise HTTPException(status_code=400, detail="没有上传任何文件")
        # 保存每个上传文件到临时路径
        for up in files:
            # 保持原始后缀
            _, ext = os.path.splitext(up.filename or "")
            if not ext:
                ext = ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmpf:
                content = await up.read()
                tmpf.write(content)
                tmp_paths.append(tmpf.name)
        # 调用 rag 模块进行处理（支持文件路径列表）
        result = rag.process_and_index_files(tmp_paths, kb_name)
        return {"success": True, "message": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

@app.post("/api/rag/ask")
async def rag_ask(
    question: str = Form(...),
    kb_name: str = Form(rag.DEFAULT_KB),
    use_search: bool = Form(True),
    use_table_format: bool = Form(False),
    multi_hop: bool = Form(False),
):
    """
    使用 RAG 模块回答问题（同步接口）
    - question: 待问问题
    - kb_name: 使用的知识库
    - use_search: 是否启用联网搜索
    - use_table_format: 是否要求表格格式输出
    - multi_hop: 是否启用多跳推理（当前 ask_question_parallel 会根据参数选择）
    """
    try:
        # 使用 rag.ask_question_parallel（内部会根据 multi_hop/use_search 决定策略）
        answer = rag.ask_question_parallel(question, kb_name=kb_name, use_search=use_search, use_table_format=use_table_format, multi_hop=multi_hop)
        return {"success": True, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/queue/tasks")
async def list_tasks(
    status: Optional[str] = Query(None, description="筛选状态: pending/processing/completed/failed"),
    limit: int = Query(100, description="返回数量限制", le=1000),
):
    """
    获取任务列表
    """
    if status:
        tasks = db.get_tasks_by_status(status, limit)
    else:
        # 返回所有任务（需要修改 TaskDB 添加这个方法）
        with db.get_cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM tasks
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit,),
            )
            tasks = [dict(row) for row in cursor.fetchall()]

    return {"success": True, "count": len(tasks), "tasks": tasks}


@app.post("/api/v1/admin/cleanup")
async def cleanup_old_tasks(days: int = Query(7, description="清理N天前的任务")):
    """
    清理旧任务记录（管理接口）
    """
    deleted_count = db.cleanup_old_tasks(days)

    logger.info(f"🧹 Cleaned up {deleted_count} old tasks")

    return {"success": True, "deleted_count": deleted_count, "message": f"Cleaned up tasks older than {days} days"}


@app.post("/api/v1/admin/reset-stale")
async def reset_stale_tasks(timeout_minutes: int = Query(60, description="超时时间（分钟）")):
    """
    重置超时的 processing 任务（管理接口）
    """
    reset_count = db.reset_stale_tasks(timeout_minutes)

    logger.info(f"🔄 Reset {reset_count} stale tasks")

    return {
        "success": True,
        "reset_count": reset_count,
        "message": f"Reset tasks processing for more than {timeout_minutes} minutes",
    }


@app.get("/api/v1/health")
async def health_check():
    """
    健康检查接口
    """
    try:
        # 检查数据库连接
        stats = db.get_queue_stats()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "queue_stats": stats,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


if __name__ == "__main__":
    # 从环境变量读取端口，默认为8000
    api_port = int(os.getenv("API_PORT", "8000"))

    logger.info("🚀 Starting Flex AI API Server...")
    logger.info(f"📖 API Documentation: http://localhost:{api_port}/docs")

    uvicorn.run(app, host="127.0.0.1", port=api_port, log_level="info")
