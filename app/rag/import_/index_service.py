from app.infra.vector_storage.milvus_gateway import milvus_gateway
from app.shared.runtime.logger import logger, step_log

from pymilvus import DataType


@step_log('require_embeddings_content')
def require_embeddings_content(state):
    # 获取数据
    embeddings_content = state.get('embeddings_content', [])

    if not embeddings_content:
        logger.error('embeddings_content为空，业务无法继续进行')
        raise ValueError('embeddings_content为空，业务无法继续进行')

    return embeddings_content


@step_log('prepare_chunks_collection')
def prepare_chunks_collection():
    # 获取客户端
    milvus_client = milvus_gateway.milvus_client

    # 获取集合名称
    collection_name = milvus_gateway.chunk_collection_name

    if milvus_client.has_collection(collection_name=collection_name):
        return

    # 创建schema
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

    schema.add_field(field_name='chunk_id', datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name='file_title', datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='item_name', datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='parent_title', datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name='part', datatype=DataType.INT8)
    schema.add_field(field_name='content', datatype=DataType.VARCHAR, max_length=65535)

    schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name='dense_vector',
        index_type='HNSW',
        index_name='dense_vector_index',
        metric_type='COSINE',
        params={
            'M': 64,  # 每个节点的最大连接数量
            'efConstruction': 100
        }
    )

    index_params.add_index(
        field_name='sparse_vector',
        index_type='SPARSE_INVERTED_INDEX',
        index_name='sparse_vector_index',
        metric_type='IP',
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )


@step_log('remove_old_chunks')
def remove_old_chunks(file_title):
    milvus_gateway.milvus_client.delete(
        collection_name=milvus_gateway.chunk_collection_name,
        filter=f"file_title == '{file_title}'"
    )


@step_log('insert_embeddings_content')
def insert_embeddings_content(embeddings_content):
    res = milvus_gateway.milvus_client.insert(
        collection_name=milvus_gateway.chunk_collection_name,
        data=embeddings_content
    )

    logger.info(f"插入数据成功! 总条数:{res.get('insert_count', 0)}")
    logger.info(f"插入数据主键回显:{res.get('ids', [])}")


@step_log('index_chunks')
def index_chunks(state):
    """
    入库服务：
    1. 准备集合 schema 和索引
    2. 根据 item_name 删除旧数据
    3. 批量插入新的 chunks
    4. 回写 chunk_id 等入库结果
    """
    # 校验切片数据存在
    embeddings_content = require_embeddings_content(state)
    # 集合不存在时先自动创建
    prepare_chunks_collection()
    # 获取主体名称 用于幂等性清理
    file_title = state.get('file_title', '')

    if file_title:
        remove_old_chunks(file_title)
    # 插入数据
    insert_embeddings_content(embeddings_content)

    return state

