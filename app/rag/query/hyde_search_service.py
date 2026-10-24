from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.infra.vector_storage.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.infra.llm.providers import llm_providers


#校验参数
def get_data_and_validates(state):

    item_names = state.get('item_names',[])
    rewritten_query = state.get('rewritten_query')

    if not rewritten_query or len(item_names) == 0:
        logger.error('关联的主体item_names或者rewritten_query重写的问题为空 ，业务中止')
        raise  ValueError('关联的主体item_names或者rewritten_query重写的问题为空 ，业务中止')

    return item_names,rewritten_query


#调用模型
def call_llm_answer(rewritten_query):
    llm_client = llm_providers.chat()

    hyde_prompt_text = load_prompt('hyde_prompt',rewritten_query=rewritten_query)

    messages = [HumanMessage(content=hyde_prompt_text)]

    chains = llm_client | StrOutputParser()

    res = chains.invoke(messages)

    return res


#向量化
def search_by_milvus(item_names, rewritten_query,answer):
    #rewritten_query向量化
    result = llm_providers.generate_embeddings([rewritten_query+':'+answer])
    #reqs请求构造(AnnSearchRequest)
    reqs = milvus_gateway.create_requests(
        dense_vector=result['dense'][0],
        sparse_vector=result['sparse'][0],
        expr=f"item_name in  {item_names}",
        limit=5*2
    )

    # 进行混合检索
    milvus_res = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        ranker_weights=(0.6,0.4),
        reqs=reqs,
        norm_score=True,
        limit=5,
        output_fields=[
            'chunk_id',
            'title',
            'item_name',
            'parent_title',
            'file_title',
            'part',
            'content'
        ]
    )
    #返回结果
    return milvus_res[0] if milvus_res and len(milvus_res)>0 else []

#整理格式
def deal_milvus_list(milvus_list):

    #向量处理后的milvus_list 结构是 {id / chunk_id,distance,entity:{}}
    hyde_embedding_chunks = []
    for item in milvus_list:
        entity = item.get('entity',{})
        hyde_embedding_chunks.append({
            'chunk_id':entity.get('chunk_id'),
            'score':item.get('distance',0.0),
            'title':entity.get('title'),
            'file_title':entity.get('file_title'),
            'parent_title':entity.get('parent_title'),
            'part':entity.get('part'),
            'item_name':entity.get('item_name'),
            'content':entity.get('content'),
            'source':'milvus',
            'url':''
        })
    return hyde_embedding_chunks


def search_by_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    HyDE 检索服务：
    1. 让 LLM 基于问题虚构一个"理想答案"
    2. 对这个假设性答案进行向量化
    3. 用答案向量在 Milvus 中检索真实文档
    4. 回写 hyde_embedding_chunks
    """
    #获取参数并校验
    item_names,rewritten_query = get_data_and_validates(state)
    #调用模型获取答案
    llm_answer = call_llm_answer(rewritten_query)
    #向量混合检索
    milvus_list = search_by_milvus(item_names,rewritten_query,llm_answer)
    #结果格式化生成
    hyde_embedding_chunks = deal_milvus_list(milvus_list)

    return hyde_embedding_chunks
