
from pydantic import BaseModel

class UploadResponseSchema(BaseModel):
    code:int = 200
    message:str|None =  None
    task_ids:list[str]


class StatusResponseSchema(BaseModel):
    code:int = 200
    task_id:str
    status:str #当前任务task_id解析文件的整体状态 正在进行 已经完成  失败
    done_list:list[str] #获取  task_id对应的已经完成的的节点
    running_list:list[str] #获取  task_id对应的正在进行的的节点