from app.infra.llm.providers import llm_providers
from app.infra.vector_storage.milvus_gateway import milvus_gateway
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger

def get_data_and_validates(state):

    item_names = state.get('item_names',[])
    rewritten_query = state.get('rewritten_query')

    if not rewritten_query or len(item_names) == 0:
        logger.error('关联的主体item_names或者rewritten_query重写的问题为空 ，业务中止')
        raise  ValueError('关联的主体item_names或者rewritten_query重写的问题为空 ，业务中止')

    return item_names,rewritten_query


def search_by_milvus(item_names, rewritten_query):
    #rewritten_query向量化
    result = llm_providers.generate_embeddings([rewritten_query])
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


def deal_milvus_list(milvus_list):

    #向量处理后的milvus_list 结构是 {id / chunk_id,distance,entity:{}}
    embedding_chunks = []
    for item in milvus_list:
        entity = item.get('entity',{})
        embedding_chunks.append({
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
    return embedding_chunks


def search_by_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    向量检索服务：
    1. 根据改写后的问题和限定的商品范围
    2. 利用 BGEM3 混合检索（稠密+稀疏）技术
    3. 从 Milvus 向量数据库中召回 Top-K 最相关的知识切片
    4. 回写 embedding_chunks
    """
    #获取参数并校验
    item_names,rewritten_query = get_data_and_validates(state)
    #向量混合检索
    milvus_list = search_by_milvus(item_names,rewritten_query)
    #结果格式化生成
    embedding_chunks = deal_milvus_list(milvus_list)

    return embedding_chunks