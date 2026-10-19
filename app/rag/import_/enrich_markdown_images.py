import os
import re
from mimetypes import guess_type
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject
from pyarrow.lib import month_day_nano_interval

from app.infra.llm.providers import llm_providers
from app.infra.object_storage.minio_gateway import minio_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.rate_limit_utils import apply_api_rate_limit
import base64

def validates_and_get_data(state:ImportGraphState) -> tuple[str,Path,Path]:

    #获取md_path 并判断其存在性
    md_path = state.get('md_path')

    if not md_path:
        logger.error('md_path 为空，业务无法继续，提前终止')
        raise ValueError('md_path 为空，业务无法继续，提前终止')


    md_path_obj:Path=Path(md_path)
    if not md_path_obj.exists():
        logger.error(f'存在md_path的地址为{md_path},但里面没有真实的文件，无法继续，业务终止')
        raise FileNotFoundError(f'存在md_path的地址为{md_path},但里面没有真实的文件，无法继续，业务终止')

    #读取md_content 并判断其存在性
    md_content = md_path_obj.read_text(encoding='utf-8')
    if not md_content:
        logger.error(f'存在md_path的地址为{md_path},有真实的文件，但是没有内容，无法继续，业务终止')
        raise ValueError(f'存在md_path的地址为{md_path},有真实的文件，但是没有内容，无法继续，业务终止')

    state['md_content'] = md_content

    #获取images_path_obj
    images_path_obj:Path = md_path_obj.parent / 'images'
    #返回值
    return md_content,images_path_obj,md_path_obj

@step_log('scan_images')
def scan_images(images_path_obj:Path, md_content:str)->list[tuple[str,str,tuple[str,str]]]:
    #创建一个image_context容器
    image_context = []

    #遍历图片文件夹 获取每一个图片
    for image_obj in images_path_obj.iterdir():
        image_name = image_obj.name
        #判断是否是图片后缀
        if image_obj.suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.warning(f'当前文件{image_name}不是图片，无需处理，跳过本次')
            continue
        #使用图片的名字，  md文件中看是否引用
        images_reg = re.compile(r'\!\[.*?\]\(.*?'+re.escape(image_name) +r'.*?\)')
        #找第一个匹配对象
        match = images_reg.search(md_content)
        if not match:
            logger.warning(f'{image_name}没有被md文件引用，无需处理，跳过本次')
            continue
        start = match.start()
        end = match.end()
        # start_1 = match.span()[0]
        # end_1 = match.span()[1]

        #获取上下文
        pre_content = md_content[max(0,start-100):start]
        post_content = md_content[end:min(len(md_content),end+100)]

        #拼接本次数据
        image_context.append(
            (
                image_name,str(image_obj),
                (pre_content,post_content)
            )
        )

    return image_context


def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:

    # Markdown 图片增强服务：
    # 1. 扫描 Markdown 中的图片
    md_content, images_path_obj, md_path_obj = validates_and_get_data(state)
    print(md_content,'11111')
    # 2. 调用多模态模型生成图片说明
    if not (images_path_obj.exists()) or images_path_obj.is_file() or (len(list(images_path_obj.iterdir()))==0):
        logger.info(f'{md_path_obj}对应的images为空，或者没有图片！不需要图片识别，提前结束当前节点')
        return state
    images_content:list[tuple[str,str,tuple[str,str]]] = scan_images(images_path_obj,md_content)

    # 3. 上传图片到 MinIO
    # 4. 替换 Markdown 图片地址并回写 md_content
    return state

