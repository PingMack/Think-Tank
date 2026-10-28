import re

from langchain_core.messages import HumanMessage

from app.infra.llm.providers import llm_providers
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task,add_running_task,push_to_session
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger
import time
import sys


def state_exist_answer(state):
    answer = state.get('answer')
    if not answer:
        logger.info("answer内容为空,业务正常进行查询的!跳入回答环节!")
        return False

    #判断是不是流式
    is_stream = state.get('is_stream',False)
    if is_stream:
        #是流式 把结果推送到队列中
        push_to_session(state.get('session_id'),SSEEvent.DELTA,{'delta':answer})


    logger.info("answer内容不为空,前期没有识别出item_name,提前给与回答!")

    return True


def load_answer_prompt(state):

    question = state.get('rewritten_query')

    reranked_docs = state.get('reranked_docs',[])
    context=''

    for i ,doc in enumerate(reranked_docs,start=1):
        context += (f"第{i}部分，标题{doc.get('title')},来源{'向量搜索' if doc.get('type') == 'web' else '向量库'}"
                    f"置信度{doc.get('score')},内容{doc.get('text')} \n ")

    item_names = f"{','.join(state.get('item_names',[]))}"

    history_text = ''
    message_list = history_repository.list_recent(state.get('session_id'),limit=6)

    if not message_list or len(message_list) == 0:
        logger.warning(f"当前会话:{state.get('session_id')}没有历史对话记录!提前跳出,history_text为空!")
        history_text = '无对话记录'

    else:
        final_message_list = [
            item for item in message_list if len(item.get('item_names',[])) >0
        ]

        if not final_message_list or len(final_message_list) == 0:
            logger.warning(f"当前会话:{state.get('session_id')}没有有效的历史对话记录!提前跳出,history_text为空!")
            history_text = "无有效对话记录!"
        else:
            for i,item in enumerate(final_message_list,start=1):
                history_text += (f"序号:{i},{'提问:' if item.get('role') == 'user' else '回答:'}"
                                 f"{item.get('rewritten_query') if item.get('role') == 'user' else item.get('text')[:50]},"
                                 f"关联主体: {','.join(item.get('item_names'))} \n")


    #加载提示词
    answer_out_str = load_prompt('answer_out',question = question,context=context,item_names=item_names,history = history_text)

    return answer_out_str


def call_llm(state, answer_prompt_text):
    is_stream= state.get('is_stream',False)
    answer=''
    #准备模型
    llm_client = llm_providers.chat()
    messages= [
        HumanMessage(content=answer_prompt_text)
    ]

    if is_stream:

        stream = llm_client.stream(messages)
        for chunk in stream:
            push_to_session(state.get('session_id'),SSEEvent.DELTA,{'delta':chunk.content})
            answer += chunk.content

    else:
        response = llm_client.invoke(messages)
        answer= response.content

    state['answer'] = answer


def extract_text_image_url(state):

    reranked_docs = state.get('reranked_docs',[])

    image_re = re.compile(r"\!\[.*?\]\((.*?)\)")
    image_urls = []

    for doc in reranked_docs:
        url = doc.get('url')
        text = doc.get('text')
        if url and url.endswith(SUPPORTED_IMAGE_EXTENSIONS):
            image_urls.append(url)
        url_list = image_re.findall(text)

        if url_list and len(url_list)>0:
            image_urls.extend(url_list)

    state['image_urls'] = image_urls




def save_answer_message_history(state):

    history_repository.save_message(
        session_id=state.get("session_id"),
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", [])
    )


def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    答案生成服务：
    1. 检查前置答案（如有追问或拒绝回答，直接输出）
    2. 构建 Prompt（用户问题 + 历史对话 + TopK 文档）
    3. 调用 LLM 生成最终答案（支持流式推送）
    4. 从引用文档中提取图片 URL
    5. 写入 MongoDB 历史记录
    6. 回写 answer 和 image_urls
    """
    ""
    #有answer
    has_answer = state_exist_answer(state)

    if not has_answer:
        # 没有answer 模型润色answer 提取image_urls
        # answer_prompt_text =  声明一个提示词拼接的方法(state) 拼接
        answer_prompt_text: str = load_answer_prompt(state)
        #调用模型处理answer字符串的问题
        call_llm(state,answer_prompt_text)
        # 使用正则或者图片url匹配获取image_urls
        extract_text_image_url(state)

    # 3.历史聊天记录记录
    save_answer_message_history(state)
    logger.info("完成第一遍 2026.7.01")
    return state
