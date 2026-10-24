from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.infra.llm.providers import llm_providers
from app.infra.persistence.history_repository import history_repository
from app.infra.vector_storage.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

@step_log('get_data_and_validates')
def get_data_and_validates(state):
    #参数获取
    session_id =state.get('session_id')
    original_query = state.get('original_query')
    #校验
    if not session_id or not original_query:
        logger.error(f'session_id或original_query为空，业务提前终止')
        raise ValueError(f'session_id或original_query为空，业务提前终止')
    return session_id,original_query


@step_log('get_history_messages_and_context')
def get_history_messages_and_context(session_id):

    message_list = history_repository.list_recent(session_id=session_id,limit=10)

    if not message_list or len(message_list)==0:
        logger.warning(f'当前会话{session_id}没有历史对话记录，不做处理')
        return '无对话记录'

    final_message_list=[ item for item in message_list if len(item.get('item_names',[]))>0]

    if not final_message_list or len(final_message_list)==0:
        logger.warning(f'当前会话{session_id}没有历史对话记录，不做处理')
        return '无有效对话记录'

    history_text=''
    for index,item in enumerate(final_message_list,start=1):
        history_text+=(f"序号:{index},{'提问:' if item.get('role') == 'user' else '回答:'}"
                         f"{item.get('rewritten_query') if item.get('role') =='user' else item.get('text')[:50]},"
                         f"关联主体: {','.join(item.get('item_names'))} \n")


    return history_text



@step_log('call_llm_item_name_and_rewritten')
def call_llm_item_name_and_rewritten(history_text, original_query):

    #获取模型
    chat_llm_client = llm_providers.chat(json_mode=True)
    #加载提示词
    history_prompt_text = load_prompt('rewritten_query_and_itemnames',query=original_query,history_text=history_text)

    messages = [
        HumanMessage(content=history_prompt_text)
    ]

    #封装chains
    chains = chat_llm_client | JsonOutputParser()

    res_dict:dict = chains.invoke(messages)

    #校验参数 获取默认值
    if 'item_names' not in res_dict:
        logger.warning("模型识别商品失败,给item_names赋予空列表")
        res_dict['item_names'] = []
    if 'rewritten_query' not in res_dict:
        logger.warning(f"模型重写问题失败,给rewritten_query赋予原始问题:{original_query}")
        res_dict['rewritten_query'] = original_query

    return res_dict

@step_log('search_by_item_names')
def search_by_item_names(item_names):
    #准备一个空字典来获取最终的结果
    final_res = {}
    #循环查询  获得item_names的列表
    item_names_vector = llm_providers.generate_embeddings(item_names)
    for index in range(len(item_names)):
        #获得稠密向量和稀疏向量
        item_name = item_names[index]
        item_name_dense = item_names_vector['dense'][index]
        item_name_sparse = item_names_vector['sparse'][index]
        #先创建annSearchRequest      然后使用混合检索
        reqs = milvus_gateway.create_requests(item_name_dense,item_name_sparse,limit=5*2)

        results = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_name_collection_name,
            reqs=reqs,
            ranker_weights=(0.5,0.5),
            norm_score=True,
            output_fields=['item_name']
        )

        #处理检索结果
        item_name_milvus_list=[]
        if len(results[0]) >0 :
            # {id:主键,distance:0.9,entity:{item_name:具体的name}}
            for a in results[0]:
                item_name_milvus_list.append({
                    'item_name':a.get('entity').get('item_name'),'score':a.get('distance')
                })

        #获得最终结果
        final_res[item_name] = item_name_milvus_list
    return final_res


@step_log('select_item_names')
def select_item_names(milvus_res):

    confirmed_list = []
    option_list = []

    for i , milvus_result_list_dict in milvus_res.items():
        high_score_list = [ item  for item in milvus_result_list_dict if item.get('score',0)>=0.70]
        md_score_list = [ item for item in milvus_result_list_dict if 0.6<= item.get('score',0)<0.70]

        if len(high_score_list)>0:
            confirmed_list.append(high_score_list[0])
            logger.info(f'识别item_name{i},有对应向量数据库中确认的item_name{high_score_list[0].get('item_name')}')
            continue

        if len(md_score_list)>0:
            option_list.extend(md_score_list[:2])
            logger.info(f"模型识别item_name:{i},没有对应向量数据库中确认的item_name,但是有可选的{','.join([ item.get('item_name') for item in md_score_list[:2]])}")
            continue

    return {
            'confirmed_list':confirmed_list,
            'option_list':option_list
        }




@step_log('apply_item_name_res')
def apply_item_name_res(state,list_dict:dict[str,list], rewritten_query:str):

    confirmed_list = list_dict.get('confirmed_list',[])
    option_list = list_dict.get('option_list',[])

    if len(confirmed_list)>0:
        state['item_names'] = [ item.get('item_name') for item in confirmed_list]
        state['rewritten_query'] = rewritten_query

        if 'answer' in state:
            state['answer'] = None

        return


    if len(option_list)>0:
        state["answer"] = f"本次提问没有确认主体,但是有相似可选的: {','.join([item.get('item_name') for item in option_list])},请您再次确认!"
        return

    state['answer'] ="本次问题没有关联到任何主体,有没有相似可选的主体! 请您明确主体再提问!"


@step_log('save_history_message')
def save_history_message(state):
    history_repository.save_message(
        session_id=state.get('session_id'),
        role='user',
        text=state.get('original_query'),
        rewritten_query=state.get('rewritten_query', ''),
        item_names=state.get('item_names', []),
        image_urls=[]
    )


@step_log('confirm_item_name')
def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    """
    意图确认服务：
    1. 结合历史对话提取商品名
    2. 将模糊问题改写为完整独立的精准问题
    3. 在 Milvus 向量库中进行混合搜索
    4. 根据评分高低自动对齐标准型号，或生成反问让用户手动确认
    5. 同步历史记录到 MongoDB
    """
    #获取参数并校验
    session_id,original_query = get_data_and_validates(state)
    #获取历史上下文 并且拼接成提示词
    history_text = get_history_messages_and_context(session_id)
    #调用模型
    res:dict=call_llm_item_name_and_rewritten(history_text,original_query)

    list_dict= {}
    if len(res.get('item_names',[]))>0:
        #进行向量检索{检索对象{‘item_name’:'score‘,‘item_name’:'score‘}，}
        milvus_res:dict[str,list[dict]] = search_by_item_names(res.get('item_names',[]))
        #确认不可信列表和可信的列表
        list_dict:dict[str,list] = select_item_names(milvus_res)

    #修改state
    apply_item_name_res(state,list_dict,res.get('rewritten_query'))
    #记录聊天问题
    save_history_message(state)

    return state