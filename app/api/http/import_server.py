"""
导入服务 HTTP 入口模块，直接承载导入接口与相关接口业务逻辑。
"""
import shutil
import sys
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
import uvicorn

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.import_schema import UploadResponseSchema, StatusResponseSchema
from app.infra.config.providers import infra_config
from app.process.import_.agent.main_graph import import_app
from app.shared.runtime.logger import PROJECT_ROOT, logger

from app.process.import_.agent.state import get_default_state, ImportGraphState, create_default_state

from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status, add_running_task, add_done_task,
)

from app.shared.runtime.logger import logger
def invoke_import_graph(task_id,local_dir,local_file_path):
    try:
        #正在执行
        update_task_status(task_id,status_name=TASK_STATUS_PROCESSING)
        state:ImportGraphState = create_default_state(
            task_id=task_id,
            local_dir=local_dir,
            local_file_path=local_file_path
        )
        import_app.invoke(state)
        #完成
        update_task_status(task_id,status_name=TASK_STATUS_COMPLETED)
    except Exception as e:
        #失败
        update_task_status(task_id,status_name=TASK_STATUS_FAILED)
        logger.exception(f'导入模块执行发生异常，异常信息{str(e)}')


app = FastAPI(
    title=infra_config.settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.2.0",
)

#CORS跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(infra_config.settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/html")
def import_html():
    """
    返回导入演示页面。
    Returns:
        FileResponse: 本地导入演示页面文件响应。
    """
    html_path = PROJECT_ROOT / "app" / "process" / "import_" / "page" / "import.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])

# 导入文件

@app.post('/upload')
def upload(backgroundtasks:BackgroundTasks,files:list[UploadFile]):
    # 获取关键参数 task_id local_dir  local_file_path
    task_id = str(uuid.uuid4())

    time_now = datetime.now().strftime('%Y%m%d')

    local_dir_obj = PROJECT_ROOT / 'output' / time_now /task_id
    #创建目录
    local_dir_obj.mkdir(exist_ok=True,parents=True)

    upload_file = files[0]
    local_file_path_obj = local_dir_obj / upload_file.filename

    #上传文档存到有效地址
    add_running_task(task_id,'upload_file')

    #分批次读取 文件 然后写入  window默认是 1mb 1024 * 1024  非window 64kb
    # 函数内指定每次 10 MB = length = 1 * 1025 kb * 1024 mb * 10
    with local_file_path_obj.open('wb') as f:

        shutil.copyfileobj(upload_file.file,f)

    add_done_task(task_id,'upload_file')

    #异步执行
    backgroundtasks.add_task(
        invoke_import_graph,
        task_id=task_id,
        local_file_path = str(local_file_path_obj),
        local_dir = str(local_dir_obj)
    )

    return  UploadResponseSchema(
        code=200,
        message=f'文件{upload_file.filename}上传成功',
        task_ids=[task_id]
    )

@app.get('/status/{task_id}')
def task_status(task_id):
    done_list =get_done_task_list(task_id)
    running_list = get_running_task_list(task_id)

    status = get_task_status(task_id)

    return StatusResponseSchema(
        code=200,
        task_id=task_id,
        status=status,
        done_list=done_list,
        running_list=running_list
    )

if __name__ == '__main__':
    uvicorn.run(
        app,
        host=infra_config.settings.app_host,
        port=infra_config.settings.import_app_port
    )