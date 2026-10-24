from typing import Any
from pydantic import BaseModel,Field



#查询接口的请求参数结构
class QueryRequestSchema(BaseModel):

    session_id: str = None
    query: str | None
    is_stream: bool = False

#健康检查的响应结构
class HealthResponseSchema(BaseModel):
    code:int =200
    message:str = None

#查询模块流式响应结果
class QueryStreamResponse(BaseModel):

    session_id: str
    message: str


#非流式
class QueryNotStreamResponse(BaseModel):

    session_id: str
    message: str
    answer: str #"答案"
    image_urls: list[str]
    done_list: list[str]


#会话记录
class HistoryClearResponseSchema(BaseModel):
    message:str
    delete_count:int


class HistoryItemResponseSchema(BaseModel):
    id:str
    session_id:str
    role:str
    text:str
    rewritten_query:str = None
    #避免脏数据
    item_names:list[str] = Field(description='已完成任务列表',default_factory=list)
    image_urls:list[str] = Field(description='关联的图片地址',default_factory=list)
    ts:Any

class HistoryListResponseSchema(BaseModel):
    session_id:str
    items:list[HistoryItemResponseSchema] = Field(description='查询的数据记录列表',default_factory=list)




