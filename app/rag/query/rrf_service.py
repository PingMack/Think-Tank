from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import NODE_RRF_TOP, NODE_RRF_K
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state):

    embedding_chunks = state.get('embedding_chunks',[])
    hyde_embedding_chunks = state.get('hyde_embedding_chunks',[])

    if len(embedding_chunks)==0 or len(hyde_embedding_chunks)==0:
        logger.error(f'embedding_chunks和hyde_embedding_chunks值为空 业务无法继续进行，立即终止')
        raise ValueError(f'embedding_chunks和hyde_embedding_chunks值为空 业务无法继续进行，立即终止')

    return embedding_chunks,hyde_embedding_chunks


@step_log("use_rrf")
def use_rrf(rrf_list,top = NODE_RRF_TOP,k=NODE_RRF_K):

    #提前准备两个字典
    score_dict = {}
    chunk_dict = {}
    #先循环每一路
    for weight,current_chunks_list in rrf_list:
        for rank,chunk in enumerate(current_chunks_list,start=1):
            # 上一次计算的得分  + 权重 * 1 / k + rank
            score_dict[chunk.get('chunk_id')] = score_dict.get(chunk.get('chunk_id'),0.0)+weight*(1/(k+rank))
            # 每次检查 保留第一次数据
            chunk_dict.setdefault(chunk.get('chunk_id'),chunk)


    chunk_list=[]
    for chunk_id,score in score_dict.items():

        chunk= chunk_dict.get(chunk_id)
        # milvus打的分替换成rrf排序的分析
        chunk['score']= score
        chunk_list.append(chunk)

    #排序处理 从大到小
    chunk_list.sort(key=lambda x:x.get('score',0),reverse=True)

    rrf_chunks = chunk_list[:top]

    return rrf_chunks

@step_log("fuse_by_rrf")
def fuse_by_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF 融合服务：
    1. 合并来自不同检索源的文档列表
    2. 应用 RRF 算法消除分数差异
    3. 给出综合排名最高的文档列表（Top 10）
    4. 回写 rrf_chunks
    """
    #获取数据并校验
    embedding_chunks,hyde_embedding_chunks = get_data_and_validates(state)
    #封装数据结构
    rrf_list = [(1.0,embedding_chunks),(1.0,hyde_embedding_chunks)]
    #使用rrf算法进行数据处理
    rrf_chunks = use_rrf(rrf_list)

    state['rrf_chunks'] = rrf_chunks
    return state