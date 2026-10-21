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

@step_log('summarize_images')
def summarize_images(image_content:list[tuple[str,str,tuple[str,str]]],stem:str)->dict[str,str]:
    # 1 获取模型对象
    vision_client = llm_providers.vision_chat()
    image_summaries = {}
    #循环处理每一张图片得到 相关的上下文
    for image_name,image_path_str,image_context in image_content:
        #添加访问限制
        apply_api_rate_limit()
        #2.封装提示词
        #导入文本提示词
        image_text = load_prompt('image_summary',root_folder=stem,image_content= image_context)

        #使用base64处理图片
        image_path_obj:Path = Path(image_path_str)
        image_base64_str = base64.b64encode(image_path_obj.read_bytes()).decode(encoding='utf-8')
        human_message = HumanMessage(
            content=[
                {
                    'type':'text',
                    'text':image_text#描述
                },
                {
                   'type':'image_url',
                    'image_url':{'url':f'data:{guess_type(image_name)[0]};base64,{image_base64_str}'}
                }
            ]
        )

        #3.封装chains  模型 | StrOutputParser()
        chains = vision_client | StrOutputParser()

        #4.chains执行时加上一个message列表
        image_summary = chains.invoke([human_message])

        image_summaries[image_name] = image_summary
        logger.info(f"完成:{image_name}图片意图识别,识别内容:{image_summary}")

    return image_summaries

@step_log('upload_images_and_replace')
def upload_images_and_replace(
        md_content:str,
        summaries_images:dict[str,str],
        image_content:list[tuple[str,str,tuple[str,str]]],
        stem:str)->str:
    """
    文件上传和md_content替换
    :param md_content: 处理之前的md_content
    :param summaries_images:图片的描述{图片名.jpg：描述}
    :param image_content:[(图片名，图片地址，(上文，下文))]
    :param stem:文件夹名
    :return:新的md_content
    """
    #1.获取minio客户端对象
    minio_client = minio_gateway.minio_client
    print(minio_client,'-=--=--')
    #2.先删除文件图片
    #先查询  列出当前文档在minio中所有已有的图片
    # prefix方法查询的时候, 前面不能添加 /   -> minio_img_dir 开头自带 /
    select_object_list = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        prefix=minio_gateway.minio_img_dir[1:] + '/' + stem,
        recursive=True#递归查询文件夹下所有文件
    )
    #删除图片
    delete_object_list = [DeleteObject(select_object.object_name) for select_object in select_object_list ]
    errors = minio_client.remove_objects(minio_gateway.bucket_name,delete_object_list=delete_object_list)
    #remove 输出是yield 需要循环输出
    for error in errors:
        logger.warning(f'文件删除状态{error}')


    #3.上传文件
    #image_urls={图片名?: 图片url }
    image_urls = {}
    for image_name,image_path_str,_ in image_content:

        try:
            object_name = minio_gateway.minio_img_dir + '/' + stem + '/' + image_name
            minio_client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=object_name,
                file_path=image_path_str,
                content_type=guess_type(image_name)[0]# [1]是压缩格式
            )
            #图片访问地址
            image_url = minio_gateway.build_image_url(stem,image_name)
            image_urls[image_name]= image_url
            logger.info(f'已经将{image_name}上传到了minio客户端服务器，访问地址{image_url}')
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            logger.warning(f'{image_name}上传失败，跳过本次 继续进行')
            continue

    #4.内容判断
    # image_urls 为空 说明图片全部上传失败
    if not image_urls:
        logger.warning('图片全部上传失败，使用默认的md_content')
        return md_content

    #5.进行md_content的内容置换
    for image_name,image_url in image_urls.items():
        image_summary = summaries_images.get(image_name)
        #正则查找 图片名
        reg = re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+r".*?\)")
        #.sub(替换的内容 类型是str， 被替换的内容 类型也是str )
        # lambda _: f"![{image_summary}]({image_url})"  -> 匿名函数 -> 返回值不会处理, 直接当做完整替换
        md_content =reg.sub(lambda _:f'![{image_summary}]({image_url})',md_content)

    return md_content

@step_log('backup_new_md_content')
def backup_new_md_content(md_content_new:str,md_path_obj:Path)->str:
    #备份 防止后续需要再次读取 也方便调试
    md_new_path_obj:Path = md_path_obj.with_name(f'{md_path_obj.stem}_new{md_path_obj.suffix}')
    md_new_path_obj.write_text(md_content_new,encoding='utf-8')
    logger.info(f'把文件md_content_new备份，备份地址为{md_new_path_obj}')
    return str(md_new_path_obj)

def enrich_markdown_images(state: ImportGraphState) -> ImportGraphState:

    # Markdown 图片增强服务：
    # 1. 扫描 Markdown 中的图片
    md_content, images_path_obj, md_path_obj = validates_and_get_data(state)
    # 2. 调用多模态模型生成图片说明
    if not (images_path_obj.exists()) or images_path_obj.is_file() or (len(list(images_path_obj.iterdir()))==0):
        logger.info(f'{md_path_obj}对应的images为空，或者没有图片！不需要图片识别，提前结束当前节点')
        return state
    images_content:list[tuple[str,str,tuple[str,str]]] = scan_images(images_path_obj,md_content)
    # 进行图片内容识别
    images_summaries: dict[str, str] = summarize_images(images_content, md_path_obj.stem)
    # 3. 上传图片到 MinIO
    # 文件上传和内容替换
    md_content_new: str = upload_images_and_replace(
        md_content, images_summaries, images_content, images_path_obj.stem
    )
    # 修改状态state对应的参数
    state['md_content'] = md_content_new
    # 备份 md_content_new.md
    md_path_new: str = backup_new_md_content(md_content_new, md_path_obj)
    state['md_path'] = md_path_new
    return state

