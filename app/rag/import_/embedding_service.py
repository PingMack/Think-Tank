from typing import Any

from app.infra.llm.providers import llm_providers
from app.process.import_.agent.state import ImportGraphState
from pathlib import Path

from app.rag.import_.config import EMBEDDING_BATCH_SIZE
from app.shared.runtime.logger import logger, step_log
import json

@step_log('get_data_and_validates')
def get_data_and_validates(state:ImportGraphState)->tuple[list[dict[str,Any]],str]:
    #获取参数
    md_path = state.get('md_path')
    chunks = state.get('chunks')
    item_name = state.get('item_name')

    #校验
    if not item_name:
        if md_path and Path(md_path).exists():
            item_name = Path(md_path).stem

        else:
            item_name = 'default_title'
        logger.warning(f'item_name为空 传入默认参数{item_name}')

    if not chunks:
        if md_path:
            chunks_json_obj = Path(md_path).with_name(f'{Path(md_path).stem}.json')
            chunks = json.loads(chunks_json_obj.read_text(encoding="utf-8"))
        if not chunks:
            logger.error('chunks为空，本地备份依旧为空，业务无法继续进行')
            raise ValueError('chunks为空，本地备份依旧为空，业务无法继续进行')


    return chunks,item_name


@step_log('batch_generate_embeddings')
def batch_generate_embeddings(chunks, item_name, batch_size=EMBEDDING_BATCH_SIZE):
    logger.info(f"未生成向量之前的示例:{chunks[0]}")

    #循环遍历
    for index in range(0,len(chunks),batch_size):
        current_batch_chunks = chunks[index:index+batch_size]
        #拼接 item_name和content
        current_content_list = [f"主体{item_name}，内容{chunk.get('content')}" for chunk in current_batch_chunks ]

        res = llm_providers.generate_embeddings(current_content_list)

        #回填
        for i,current_chunk in enumerate(current_batch_chunks):
            current_chunk['dense_vector'] = res.get('dense')[i]
            current_chunk['sparse_vector'] = res.get('sparse')[i]

    logger.info(f"生成向量之后的示例:{chunks[0]}")

    return chunks





@step_log('generate_chunk_embeddings')
def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """
    向量化服务：
    1. 读取 chunks
    2. 生成 dense_vector / sparse_vector
    3. 将向量结果补充回 chunks
    """
    #获取参数并校验
    chunks,item_name = get_data_and_validates(state)
    #批量生成向量
    embeddings_content = batch_generate_embeddings(chunks,item_name,EMBEDDING_BATCH_SIZE)
    #更新state
    state['embeddings_content'] =embeddings_content


    return state