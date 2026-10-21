from pathlib import Path
from typing import Any
import json

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.infra.llm.providers import llm_providers
from app.infra.vector_storage.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNKS_SPLIT_TOP_NUMBER, MILVUS_VECTOR_DIM, MILVUS_DEFAULT_VARCHAR_MAX_LENGTH
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log

@step_log("validate_chunks_and_title")
def validate_chunks_and_title(state: dict) -> tuple[list[dict], str]:
    """
    校验并提取 state 中的 chunks 和 file_title
    功能：确保后续流程有有效的输入数据，缺失时提供兜底策略
    :param state: LangGraph 流程状态字典
    :return: (chunks列表, file_title字符串)
    """
    # 从 state 中获取核心数据
    chunks = state.get("chunks", [])
    file_title = state.get("file_title")

    # ===================== 校验 chunks =====================
    # 如果 chunks 为空，无法继续业务，直接抛出异常终止流程
    if not chunks:
        logger.error("chunks没有内容,无法继续业务!")
        raise ValueError("chunks没有内容,无法继续业务!")

    # ===================== 处理 file_title 缺失场景 =====================
    # 如果标题为空，使用默认值兜底，避免后续流程出错
    if not file_title:
        logger.warning("file_title为空给与默认值处理!")
        file_title = "default_title"
        state["file_title"] = file_title  # 回填到 state

    # 返回校验后的数据
    return chunks, file_title

def recognize_item_name_by_chunks(chunks,file_title)->str:

    #1.获取语言模型对象
    chain_client = llm_providers.chat()
    #2.加载提示词
    system_prompt_text = load_prompt('product_recognition_system')
    user_context = ''

    used_chunks = chunks[:CHUNKS_SPLIT_TOP_NUMBER]

    for index,chunk in enumerate(used_chunks,start=1):
        user_context +=f'第{index}部分,标题是{chunk.get('title')},内容是{chunk.get('content')}\n'

    user_prompt_text = load_prompt('item_name_recognition',file_title=file_title,context= user_context)

    #3.封装成message
    messages= [
        SystemMessage(content=system_prompt_text),
        HumanMessage(content=user_prompt_text)
    ]

    #4.封装chains
    chains = chain_client | StrOutputParser()

    #5.调用chains
    item_name = chains.invoke(messages)

    #6.非空校验
    if not  item_name:
        item_name = file_title
        logger.warning(f'模型未识别到item_name,使用默认值file_title{file_title}')

    #返回主体item_name
    return item_name

@step_log('delete_and_insert_item_name')
def delete_and_insert_item_name(item_name,file_title):

    #根据item_name生成稠密和稀疏向量
    res = llm_providers.generate_embeddings([item_name])
    item_name_dense = res.get('dense')[0]
    item_name_sparse = res.get('sparse')[0]
    #删除之前file_title对应的数据
    milvus_client = milvus_gateway.milvus_client

    milvus_client.delete(
        collection_name=milvus_gateway.item_name_collection_name,
        filter=f"file_title == '{file_title}'"
    )

    #准备数据
    data=[
        {
            'file_title':file_title,
            'item_name':item_name,
            'dense_vector':item_name_dense,
            'sparse_vector':item_name_sparse
        }
    ]

    milvus_client.insert(
        collection_name=milvus_gateway.item_name_collection_name,
        data=data
    )

    logger.info(f'{item_name}数据更新完成')



@step_log("prepare_item_name_collection")
def prepare_item_name_collection() -> None:
    """
    准备 Milvus 主体名称集合
    功能：检查集合是否存在，不存在则创建 schema 和索引
    :return: 无返回值
    """
    # 获取 Milvus 客户端
    milvus_client = milvus_gateway.client()

    # 获取集合名称（从配置中读取）
    collection_name = milvus_gateway.item_name_collection

    # 如果集合已存在，直接返回，无需重复创建
    if milvus_client.has_collection(collection_name=collection_name):
        return

    # ===================== 创建 Schema =====================
    # 创建 schema，启用自动 ID 和动态字段
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

    # 添加主键字段：pk，INT64 类型，自增
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # 添加文件标题字段：VARCHAR 类型，最大长度 512 nullable=True
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)

    # 添加主体名称字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)

    # 添加稠密向量字段：FLOAT_VECTOR 类型，维度 1024
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)

    # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR 类型
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # ===================== 创建索引 =====================
    # 准备索引参数
    index_params = milvus_client.prepare_index_params()

    # 为稠密向量创建索引：使用 AUTOINDEX，metric_type 为 IP（内积）
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        index_name="dense_vector_index",
        metric_type="IP",
    )

    # 为稀疏向量创建索引：使用 SPARSE_INVERTED_INDEX，算法为 DAAT_MAXSCORE
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    # 创建集合并应用索引
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

@step_log('chunk_update_item_name')
def chunk_update_item_name(chunks,item_name):
    #在chunk里面添加item_name这项参数
    for chunk in chunks:
        chunk['item_name'] = item_name
    logger.info(f'完成chunk的item_name{item_name}参数更新')

@step_log('prepared_milvus_item_name_collection')
def prepared_milvus_item_name_collection():

    item_name_collection_name = milvus_gateway.item_name_collection_name

    milvus_client=milvus_gateway.milvus_client

    #校验是否存在集合
    if milvus_client.has_collection(collection_name=item_name_collection_name):
        logger.info(f'{item_name_collection_name}集合已经存在')
        return
    # if milvus_client.has_collection(collection_name=item_name_collection_name):
    #     logger.info(f"{item_name_collection_name}集合已经存在，开始加载集合至内存")
    #     # 新增关键加载语句
    #     #本地的milvus 版本低需要重新加载
    #     milvus_client.load_collection(collection_name=item_name_collection_name)
    #     logger.info(f"{item_name_collection_name}集合加载完成")
    #     return

    logger.info(f'{item_name_collection_name}集合不存在 创建新的集合')

    #准备schema 集合结构  其中参数 auto_id=True 主键是否自增长
    #enable_dynamic_field=True 是让集合支持拓展
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True
    )
    #bge_m3 1024维度
    schema.add_field(field_name='pk',datatype=DataType.INT64,is_primary=True)
    schema.add_field(field_name='file_title',datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name='item_name',datatype=DataType.VARCHAR,max_length=512)
    schema.add_field(field_name='dense_vector',datatype=DataType.FLOAT_VECTOR,dim=1024)
    schema.add_field(field_name='sparse_vector',datatype=DataType.SPARSE_FLOAT_VECTOR)

    #添加索引
    index_params = milvus_client.prepare_index_params()

    #稠密向量
    index_params.add_index(
        field_name='dense_vector',
        index_type='HNSW',
        metric_type='COSINE',
        params={
            "M": 64,  # 每个点最大的链接数量
            "efConstruction": 100  # 考虑链接的范围 在这个范围内确定 M
        }
    )

    #稀疏向量 索引是  倒排索引  SPARSE_INVERTED_INDEX
    index_params.add_index(
        field_name='sparse_vector',
        index_type='SPARSE_INVERTED_INDEX',
        metric_type = 'IP',
        #对权重值做一些优化，降低一些低权重数据的排名
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
    )
    #创建集合
    milvus_client.create_collection(
        collection_name=item_name_collection_name,
        schema=schema,
        index_params=index_params
    )

def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """
    主体识别服务：
    1. 基于 chunks 构造上下文
    2. 调用 LLM 识别 item_name
    3. 将 item_name 回填到 state 和 chunks
    4. 同步写入主体名称索引
    """
    """
      主体识别服务总入口
      功能：从文档切片中识别主体名称 → 回填到 state 和 chunks → 生成向量 → 写入 Milvus 主体索引
      输出：更新后的 state，包含 item_name 和带有 item_name 的 chunks
      """
    # 1. 校验输入数据，确保 chunks 和 file_title 有效
    chunks, file_title = validate_chunks_and_title(state) #调用模型，识别item_name 主体
    item_name =recognize_item_name_by_chunks(chunks,file_title)
    #给chunk 补全 item_name属性
    chunk_update_item_name(chunks,item_name)
    #准备好milvus集合
    prepared_milvus_item_name_collection()
    # 插入数据
    delete_and_insert_item_name(item_name, file_title)
    # 更新状态state
    state['chunks'] = chunks
    state['item_name'] = item_name

    return state