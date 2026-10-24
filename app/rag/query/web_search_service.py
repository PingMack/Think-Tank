import asyncio
import json

from agents.mcp import MCPServerStreamableHttp

from app.infra.config.providers import infra_config
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger

def get_data_and_validates(state):
    rewritten_query = state.get('rewritten_query')
    if not rewritten_query:
        logger.error('重写问题为空，业务无法继续，提前终止')
        raise ValueError('重写问题为空，业务无法继续，提前终止')

    return rewritten_query


async  def openai_mcp(rewritten_query):
    #连接mcp服务
    mcp_server = MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params={
            "url": infra_config.mcp_config.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp_config.api_key}"},
            "timeout": 50,
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    )
    try:
        #连接
        await mcp_server.connect()
        #调用
        res = await mcp_server.call_tool(
            tool_name='bailian_web_search',
            arguments={
                'query':rewritten_query,
                'count':5
            }
        )
        return res

    except Exception as e :
        logger.exception(f'mcp服务调用发生问题，问题{str(e)}')
    finally:
        #清洗
        await mcp_server.cleanup()

def search_by_web(state: QueryGraphState) -> QueryGraphState:
    """
    网络搜索服务：
    1. 通过 MCP 协议异步调用百炼联网搜索接口
    2. 将用户的查询转化为实时的、结构化的网络搜索结果
    3. 包含标题、链接和摘要
    4. 回写 web_search_docs
    """
    #获取并校验参数
    rewritten_query = get_data_and_validates(state)
    #异步 async  调用openai提供的mcp方式进行调用  查询结果
    mcp_res = asyncio.run(openai_mcp(rewritten_query))
    #解析结果  获取需要的结果
    text= mcp_res.content[0].text
    # {pages: [{snippet, title, url}, {}, {}]}
    res_dict = json.loads(text)
    web_search_docs = res_dict.get('pages',[])
    #返回列表即可
    return web_search_docs