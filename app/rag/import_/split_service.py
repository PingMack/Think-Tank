import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MIN_SIZE, CHUNK_MAX_SIZE
from app.shared.runtime.logger import logger, step_log


@step_log('split_document')
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分服务：
    1. 按标题层级做一级粗切
    2. 对超长文本做二次细切
    3. 构造 chunks 列表
    4. 回写 chunks
    """
    #获取数据
    md_content, file_title = load_markdown_content(state)

    #按照语义切割

    chunks:list[dict[str,Any]] = split_chunks_document(md_content,file_title)
    #进行精细切割
    refine_chunks_list = refine_chunks(chunks)
    #修改state
    state['chunks'] =refine_chunks_list
    #备份
    backup_chunks_json(refine_chunks_list,state['md_path'])

    return state

@step_log('load_markdown_content')
def load_markdown_content(state:ImportGraphState)->tuple[str,str]:
    #读取数据并校验
    md_content = state.get('md_content')
    file_title = state.get('file_title')
    #md_path兜底
    md_path = state.get('md_path')

    #非空校验
    if not md_content:
        logger.warning('没有从state中读取到md_content，使用md_path重新尝试读取')
        #文件路径存在就读取md_path
        if md_path:
            md_content = Path(md_path).read_text(encoding='utf-8')
            #读取后回填到状态 避免重复读取
            state['md_content'] = md_content

        #双重校验 ，仍然无内容，抛出异常终止程序
        if not md_content:
            logger.error('md_content没有数据，且使用md_path重新尝试读取仍然没有数据，终止执行')
            raise ValueError('md_content没有数据，且使用md_path重新尝试读取仍然没有数据，终止执行')

    if not file_title:
        if md_path and Path(md_path).exists():
            file_title = Path(md_path).stem
        if not file_title:
            file_title = 'default'
        state['file_title'] = file_title
        logger.warning(f'file_title值为空，给予默认赋值{file_title}')

    #数据清洗 统一换行符
    md_content = md_content.replace('\r\n','\n').replace('\r','\n')

    state['md_content'] = md_content
    return md_content,file_title

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

@step_log('_split_long_chunk')
def _split_long_chunk(chunk)-> list[dict[str,Any]]:
    #清洗原有的content 去掉标签前缀
    content = chunk.get('content')
    title = chunk.get('title')
    file_title = chunk.get('file_title')

    # + 1 因为去掉 \n符号 = 1  r"\n" = 2
    #去掉前缀后后  2. 内容1 ||| 3.内容2
    clear_content = content[len(title)+1:]
    #新添加一个公共前缀
    sub_content_prefix =title + '\n'

    #定义递归切割器
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE - len(sub_content_prefix),
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n','\n','。','!','?',';','！','？','.','；']
    )
    sub_chunk_list = []

    #递归切割进行content 切割
    for part_index,split_text in enumerate(spliter.split_text(clear_content),start=1):
        #拼接每一个子chunk
        split_text = split_text.strip()
        content_sub_new = sub_content_prefix + split_text

        sub_chunk_list.append({
            'title':f'{title}的第{part_index}部分',
            'content':content_sub_new,
            'file_title':file_title,
            'parent_title':title,
            'part':part_index
        })

    #返回结果
    logger.info(f"进入标题:{title},完成切割后,切成:{len(sub_chunk_list)}块!")
    return sub_chunk_list

@step_log('_merge_short_chunks_same_parent_title')
def _merge_short_chunks_same_parent_title(refine_list):

    merged_chunk_list = []
    #定义一个参考base的变量
    base_chunk = None
    #循环判断是否需要合并
    for next_chunk in refine_list:
        if not base_chunk:
            #赋值
            base_chunk = next_chunk
            logger.info('第一次进入循环，将第一个设定为base_chunk 的值')
            continue
        #判断合并的逻辑
        is_short_chunk = len(base_chunk.get('content')) < CHUNK_MIN_SIZE
        #确保不同语义的文本块不能合并
        is_same_parent_title = base_chunk.get('parent_title') and base_chunk.get('parent_title') == next_chunk.get('parent_title')

        if is_short_chunk and is_same_parent_title:
            #可以合并
            next_content = next_chunk.get('content')[len(next_chunk.get('parent_title'))+1:]

            is_not_long = (len(base_chunk.get('content')) + len(next_content))  <= CHUNK_MAX_SIZE

            if is_not_long:
                #没有超过最大字符限制 1000
                base_chunk['content'] = base_chunk.get('content') + '\n' + next_content
                continue
            else:

                merged_chunk_list.append(base_chunk)
                #指针变化指向下一个next 作为base
                base_chunk = next_chunk
                continue
        else:
            # 不能合并
            merged_chunk_list.append(base_chunk)
            # 指针变化指向下一个next 作为base
            base_chunk = next_chunk
            continue

    if base_chunk:
        merged_chunk_list.append(base_chunk)
    logger.info(f'进行短合并,合并之前:{len(refine_list)},合并之后:{len(merged_chunk_list)}')
    return merged_chunk_list

@step_log('refine_chunks')
def refine_chunks(chunks)->list[dict[str,Any]]:
    #1.定义接受最终结果的list
    refine_list = []
    #2.循环处理看是否过长
    for chunk in chunks:
        content = chunk.get('content')
        if len(content)>CHUNK_SIZE:
            #过长的短切
            long_chunk_list = _split_long_chunk(chunk)
            refine_list.extend(long_chunk_list)
        else:
            refine_list.append(chunk)
    #3短合并处理
    refine_list = _merge_short_chunks_same_parent_title(refine_list)
    #4补全属性
    for chunk in refine_list:
        if 'parent_title' not in chunk:
            chunk['parent_title'] = chunk.get('title','default_title')
        if 'part' not in chunk:
            chunk['part'] = 1
    #返回结果
    logger.info(f'完成对chunks的精确切割，处理之前的切块数{len(chunks)},处理之后的切块数{len(refine_list)}')
    return  refine_list

@step_log('backup_chunks_json')
def backup_chunks_json(refine_chunks_list,md_path):
    #获取目标文件路径
    json_path_obj:Path = Path(md_path).with_name(f'{Path(md_path).stem}.json')
    #目标写入字符串
    json_path_obj.write_text(json.dumps(refine_chunks_list,indent=4,ensure_ascii=False),encoding='utf-8')
    logger.info(f'已经将切片数据备份到{json_path_obj}')


