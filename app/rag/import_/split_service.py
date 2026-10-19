import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN_SIZE, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger, step_log


@step_log('split_chunks_document')
def split_chunks_document(md_content,file_title):#->list[dict{title,content,file_title}]:
    # 按行切割md_content
    md_content_lines:list[str] = md_content.split('\n')
    #准备数据
    #切块结果
    chunks:list[dict] = []
    #当前切块的标题
    current_title = None
    # 当前块的所有行内容
    current_title_lines = []
    #是否属于代码块内部
    is_code_block = False
    #正则 判断是不是标题
    title_reg = re.compile(r'^\s*#{1,6}\s.+')

    for line in md_content_lines:
        line_strip = line.strip()
        if not line_strip:
            #空行
            logger.warning('此行为空，跳过本次处理')
            continue
        #是不是代码块
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            is_code_block = not is_code_block
            current_title_lines.append(line_strip)
            continue

        #是不是有效标题
        if not is_code_block and title_reg.match(line_strip):
            #只看最近的有效标题
            if current_title and len(current_title_lines)>1:

                chunks.append({
                    'content':'\n'.join(current_title_lines),
                    'title':current_title,
                    'file_title':file_title
                })

            if not current_title and len(current_title_lines)>0:
                current_title_lines.append(line_strip)

            else:
                current_title_lines = [line_strip]
            # 新的设置为当前处理标签
            current_title = line_strip

        else:
            # 将标题设置第一行字符串
            current_title_lines.append(line_strip)

    #最后一次没有被结算
    if current_title and len(current_title_lines)>1:
        chunks.append({
            'content':'\n'.join(current_title_lines),
            'title':current_title,
            'file_title':file_title
        })
    #全文没有标题
    if len(chunks) == 0 and len(current_title_lines)>0:
        chunks.append({
            'content': '\n'.join(current_title_lines),
            'title': 'default',
            'file_title': file_title
        })

    logger.info(f'按照语义切块完成 ，切块数量是{len(chunks)}')

    return chunks


def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """
    return state