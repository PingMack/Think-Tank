from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from tokenizers.decoders import Strip

from app.infra.llm.providers import llm_providers
from app.process.query.agent.state import QueryGraphState
from app.rag.query.config import RERANK_MAX_INPUT_TOKENS, RERANK_MIN_SUMMARY_CHARS, RERANK_MAX_TOPK, RERANK_MIN_TOPK, \
    RERANK_GAP_ABS, RERANK_GAP_RATIO
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

@step_log("get_data_and_validates")
def get_data_and_validates(state):
    rewritten_query=state.get('rewritten_query')
    rrf_chunks = state.get('rrf_chunks',[])
    web_search_docs = state.get('web_search_docs',[])

    if not rewritten_query or len(rrf_chunks)==0 or len(web_search_docs)==0:
        logger.error('rewritten_query,rrf_chunks,web_search_docs可能为空,业务无法继续进行,提前终止')
        raise ValueError('rewritten_query,rrf_chunks,web_search_docs可能为空,业务无法继续进行,提前终止')
    return rewritten_query,rrf_chunks,web_search_docs



@step_log("deal_rrf_and_web_search")
def deal_rrf_and_web_search(rrf_chunks,web_search_docs):
    #定义一个列表
    reranker_docs = []

    #分别遍历rrf_chunks  web_search_docs
    for chunk in rrf_chunks:
        reranker_docs.append({
            'chunk_id':chunk.get('chunk_id'),
            'text':chunk.get('content'),
            'title':chunk.get('title'),
            'score':0,
            'type':'milvus',
            'url':None
        })

    for doc in web_search_docs:
        reranker_docs.append({
            'chunk_id':None,
            'text':doc.get('snippet'),
            'title':doc.get('title'),
            'score':0,
            'type':'web',
            'url':doc.get('url')
        })

    return reranker_docs



@step_log("create_question_answer_list")
def create_question_answer_list(rewritten_query, reranker_docs):


    question_answer_pair_list = []

    #获取rewritten_query并计算token数量
    reranker_model = llm_providers.reranker_model()
    tokenizer = reranker_model.tokenizer

    rewritten_query_token_list = tokenizer.encode(rewritten_query,add_special_tokens=False)
    rewritten_query_token_len = len(rewritten_query_token_list)

    #循环reranker_docs获取每个text答案
    for doc in reranker_docs:
        answer = doc.get('text')
        answer_token_len =len(tokenizer.encode(answer,add_special_tokens=False))
        # 超长了调用模型进行压缩
        if rewritten_query_token_len + answer_token_len +4 >RERANK_MAX_INPUT_TOKENS:

            limit = max(
                RERANK_MIN_SUMMARY_CHARS,
                int((RERANK_MAX_INPUT_TOKENS - rewritten_query_token_len -4) / RERANK_MIN_SUMMARY_CHARS)
            )
            #加载提示词
            rerank_text_refine_str = load_prompt('rerank_text_refine',question=rewritten_query,answer=answer,limit=limit)
            #封装message
            messages=[
                HumanMessage(content=rerank_text_refine_str)
            ]

            chains = llm_providers.chat() | StrOutputParser()

            answer = chains.invoke(messages)

        question_answer_pair_list.append([rewritten_query,answer])

    return question_answer_pair_list


@step_log("use_reranker_deal_score")
def use_reranker_deal_score(question_answer_pair_list, reranker_docs):
    #调用reranker打分
    reranker_model = llm_providers.reranker_model()
    # normalize=True 归一化 避免负分  0 - 1分之间
    scores_list =reranker_model.compute_score(question_answer_pair_list,normalize=True)

    for score, doc in zip(scores_list,reranker_docs):
        doc['score']=score
    #倒序排序
    reranker_docs.sort(key=lambda  x:x.get('score',0),reverse=True)



@step_log("dyn_limit_reranker_docs")
def dyn_limit_reranker_docs(reranker_docs):
    #定义常量
    top_max:int = RERANK_MAX_TOPK
    top_min:int = RERANK_MIN_TOPK
    gap_abs:float = RERANK_GAP_ABS
    gap_ratio:float = RERANK_GAP_RATIO

    #情况1 max > len
    top_max = min(top_max,len(reranker_docs))

    #情况2 没有出现断崖
    topk= top_max

    #情况3 出现断崖 且 min< max | max = len
    if top_max > top_min:
        #循环双指针
        for pre in range(top_min-1,top_max-1):
            pre_score = reranker_docs[pre].get('score',0.0)
            post_score = reranker_docs[pre+1].get('score',0.0)

            abs = pre_score - post_score
            ratio = abs / pre_score

            if abs > gap_abs or ratio > gap_ratio:
                topk =pre + 1
                break
    final_reranker_docs = reranker_docs[:topk]

    return final_reranker_docs


@step_log("rerank_documents")
def rerank_documents(state: QueryGraphState) -> QueryGraphState:
    """
    重排序服务：
    1. 合并 RRF 和 Web Search 的文档
    2. 使用 BGE Reranker 模型计算相关性得分
    3. 根据得分动态截断，智能截取 TopK
    4. 回写 reranked_docs
    """
    #获取参数并校验
    rewritten_query,rrf_chunks,web_search_docs = get_data_and_validates(state)
    #数据格式化处理
    reranker_docs = deal_rrf_and_web_search(rrf_chunks,web_search_docs)
    #拼接问题和答案  组成一个列表[[],[],[]]
    question_answer_pair_list = create_question_answer_list(rewritten_query,reranker_docs)
    #打分 排序
    logger.info(f"排序和打分之前的数据:{reranker_docs}")
    use_reranker_deal_score(question_answer_pair_list,reranker_docs)
    logger.info(f"排序和打分之后的数据:{reranker_docs}")
    # 5. 动态截取数据
    reranker_docs = dyn_limit_reranker_docs(reranker_docs)

    state['reranked_docs'] = reranker_docs
    return state