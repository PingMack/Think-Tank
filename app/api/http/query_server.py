from mimetypes import guess_type
import uvicorn
import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.query_shema import HealthResponseSchema, QueryRequestSchema, QueryStreamResponse, \
    QueryNotStreamResponse, HistoryClearResponseSchema, HistoryListResponseSchema, HistoryItemResponseSchema
from app.infra.persistence.history_repository import history_repository
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_app
from app.process.query.agent.state import create_query_default_state
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator

from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status,
)

import datetime
# 定义fastapi对象
app = FastAPI(
    title=settings.query_app_name,
    description="描述,进行rag查询的服务对象",
    version="0.2.0"
)

# 跨域处理
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)

#接口1 页面入口
@app.get("/html")
def query_html():
    html_path = PROJECT_ROOT / "app" / "resources" / "html" / "chat.html"
    return FileResponse(path=str(html_path), media_type=guess_type(html_path.name)[0])


#接口2 健康检查
@app.get('/health')
def health():
    logger.info(f'完成健康检查{datetime.datetime.now()}')
    return HealthResponseSchema(
        code=200,
        message=f'完成健康检查{datetime.datetime.now()}'
    )

def invoke_query_graph(session_id,original_query,is_stream):
    try:
        #清空task字典,都需要先清空
        clear_task(session_id)
        #如果是流式 提前创建队列
        if is_stream:
            create_sse_queue(session_id)

        update_task_status(session_id,TASK_STATUS_PROCESSING,push_queue=is_stream)
        #封装进state
        query_state = create_query_default_state(session_id=session_id,original_query=original_query,is_stream=is_stream)

        #图执行
        res_state = query_app.invoke(query_state)
        #完成
        update_task_status(session_id,TASK_STATUS_COMPLETED,push_queue=is_stream)

        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": res_state.get('answer'),
                    "status": "completed",
                    "image_urls": res_state.get('image_urls',[])
                }
            )
        return  res_state

    except Exception as e:
        update_task_status(session_id,TASK_STATUS_FAILED,push_queue=is_stream)
        logger.info(f'查询出错,{str(e)}')



#接口3 查询接口
@app.post('/query')
def query(backgroundtasks:BackgroundTasks,query_params:QueryRequestSchema):
    #获取参数
    query = query_params.query
    session_id = query_params.session_id or str(uuid.uuid4())
    is_stream = query_params.is_stream

    #是不是流式响应
    if is_stream:
        backgroundtasks.add_task(
            invoke_query_graph,
            session_id=session_id,
            original_query=query,
            is_stream=is_stream
        )
        return QueryStreamResponse(
            message=f'{query}开始查询',
            session_id = session_id
        )

    else:
        state:QueryStreamResponse= invoke_query_graph(
            session_id=session_id,
            original_query=query,
            is_stream=is_stream
        )
        done_list = get_done_task_list(session_id)

        return QueryNotStreamResponse(
            message=f'{query}查询完成',
            session_id=session_id,
            answer=state.get('answer'),
            done_list=done_list,
            image_urls=state.get('image_urls',[])
        )

#接口4：流式响应接口
@app.get('/stream/{session_id}')
def stream_query(session_id,request:Request):
    #request.is_disconnected()
    # 通过request对象可以检查前端是否已经断开
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type='text/event-stream',
        # headers={}
    )
#接口5 清空会话历史
@app.delete('/history/{session_id}')
def clear_history(session_id):
    delete_count = history_repository.clear_session(session_id=session_id)

    return HistoryClearResponseSchema(
        message=f'{session_id}的历史聊天记录已经删除',
        delete_count=delete_count
    )

#接口6
@app.get('/history/{session_id}')
def get_history(session_id,limit):
    #history_list 结构是list[dict]
    history_list = (history_repository.list_recent(
        session_id=session_id,
        limit=limit
    ))
    return  HistoryListResponseSchema(
        session_id=session_id,
        items=[
            HistoryItemResponseSchema(
                id = item.get('id'),
                session_id=session_id,
                role=item.get('role'),
                text=item.get('text'),
                rewritten_query=item.get('rewritten_quer                                                                                    y'),
                item_names=item.get('item_names',[]),
                image_urls=item.get('image_urls',[]),
                ts = item.get('ts')
            )
            for item in history_list
        ]
    )


if __name__ == '__main__':
    uvicorn.run(
        app,
        host=settings.app_host,
        port = settings.query_app_port
    )






