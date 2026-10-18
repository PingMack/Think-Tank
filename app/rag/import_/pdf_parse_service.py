import shutil
import time

import requests

from app.infra.config.providers import infra_config
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import MINERU_MODEL_VERSION, MINERU_POLL_TIMEOUT_SECONDS, MINERU_POLL_INTERVAL_SECONDS
from app.shared.runtime.logger import logger, PROJECT_ROOT, step_log
from pathlib import Path
# 临时强制覆盖 API Key

#        1. 参数获取和校验 (state -> str)-> (pdf_path:Path,local_dir:Path)  validate_pdf_paths
#           # 获取参数 并且校验 非空 | 真实存在性校验
#           1.1 state获取 pdf_path 和 local_dir : str
#           1.2 进行pdf_path非空校验
#               空 -> 打印日志error -> 异常 ValueError
#           1.3 进行local_path非空校验
#               空 -> 打印日志warning -> 给与默认值  项目根路径 / output
#           1.4 将pdf_path local_dir 转成Path
#           1.5 pdf_path_obj:Path 判断是否存在
#               不存在,打印日志error -> 异常 FileNotFoundError
#           1.6 local_dir_obj:Path 是不是目录(is_dir())
#               不是(不存在,或者不是目录)
#               打印日志warning -> 创建 local_dri_obj对应的文件夹
#           1.7 返回结果 return pdf_path_obj , local_dir_obj


# 校验路径和输出目录
@step_log('validate_pdf_paths')
def validate_pdf_paths(state: dict) -> tuple[Path, Path]:
    # 数据校验
    pdf_path = state.get('pdf_path')
    local_dir = state.get('local_dir')

    if not pdf_path:
        logger.error('pdf_path路径为空，程序停止')
        raise ValueError('pdf_path路径为空，程序停止')

    if not local_dir:
        logger.warning('local_dir的路径为空，系统给与默认值，项目根路径 / output')
        local_dir = PROJECT_ROOT / 'output'
        state['local_dir'] = str(local_dir)

    # 转为Path对象 方便后续路径操作
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        logger.error(f'{pdf_path_obj}文件路径存在，但是里面没有文件')
        raise FileNotFoundError(f'{pdf_path_obj}文件路径存在，但是里面没有文件')
    # 输出目录不存在就自动创建,已经有也不报错
    if not local_dir_obj.exists():
        logger.warning(f'{local_dir_obj}路径存在，没有文件夹，系统将会自动创建')
        # ,exist_ok=True 这个就是说有就不管 没有就创建
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj


@step_log('upload_pdf_and_poll')
def upload_pdf_and_poll(pdf_path_obj: Path) -> str:
    print(infra_config.mineru_config,'0-0' * 50)
    # 函数主要进行mineru的交互和zip的获取
    # 1 向MinerU服务器上传地址
    # url 地址 headers 请求头  params ? 后参数  json 请求体参数  data 请求体 字节数据
    header = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {infra_config.mineru_config.api_key}'
    }

    data = {
        'files': [
            {'name': f'{pdf_path_obj.name}'}
        ],
        'model_version': MINERU_MODEL_VERSION
    }

    url = f'{infra_config.mineru_config.base_url}/file-urls/batch'

    response = requests.post(url=url, json=data, headers=header)
    # 所有的网络请求,必须两步骤判定 1. 状态码 200 http的网络状态  2. 业务状态必须成功
    if response.status_code != 200:
        logger.error(f'http状态为{response.status_code},申请解析网络状态错误，无法解析')
        raise RuntimeError(f'http状态为{response.status_code},申请解析网络状态错误，无法解析')

    response_dict = response.json()
    # 业务状态
    if response_dict.get('code', -1) != 0:
        logger.error(f'http状态正常，但是服务出错，code={response_dict.get('code', -1)}')
        raise RuntimeError(f'http状态正常，但是服务出错，code={response_dict.get('code', -1)}')

    # 两个都没问题
    batch_id = response_dict.get('data', {}).get('batch_id')
    file_upload_urls = response_dict.get('data', {}).get('file_urls', [])
    file_upload_url = None

    if len(file_upload_urls) > 0:
        file_upload_url = file_upload_urls[0]

    if not batch_id:
        logger.error('申请解析文件 返回的batch_id为空，业务进程中断')
        raise ValueError('申请解析文件 返回的batch_id为空，业务进程中断')

    logger.info(f'完成上传文件申请，batch_id:{batch_id},文件的与签名地址是{file_upload_url}')
    # file_upload_url  第一次请求申请地址 -> minerU -> 文件服务器 -> 开辟了一个空间 ->  空间对应的地址 -> 返回
    # 置换 -> 开辟的空间 -> 换成我们本次上传的文件 -> put
    # 预签名地址  第三方文件服务器的地址(想要往服务器上传文件需要认证) ->
    # 方案1: 上传的时候 传入token 认证  方案2: 预先认证(免检)
    # 预先签名地址 -> put(代码) ->  代理(vpn) 添加额外的请求头... -> 电脑的网络(网卡)  ->   文件服务器  很大概率会报错(认为你是免检,但是你中间干了不该干的事)
    # 尽量让请求更加干净 不要携带其他不相关的代理头
    # 预先签名 -> 服务器对你检查 -> 越严格

    # session会话 (1.复用请求请求对象 2. 属性设置了以后,可以不信任当前系统的环境,保证请求的整洁性) 和 requests 都可以发起请求
    with requests.Session() as session:
        session.trust_env = False

        upload_response = session.put(url=file_upload_url, data=pdf_path_obj.read_bytes())

        if upload_response.status_code != 200:
            logger.error(
                f"向:{file_upload_url}上传文件,服务器返回的网络状态码为:{upload_response.status_code},业务失败,提前终止!")
            raise RuntimeError(
                f"向:{file_upload_url}上传文件,服务器返回的网络状态码为:{upload_response.status_code},业务失败,提前终止!")

    # 轮询向minerU获取batch_id解析状态 zip_url
    result_url = f'{infra_config.mineru_config.base_url}/extract-results/batch/{batch_id}'
    start_time = time.time()

    # 一个循环
    while True:
        # 第一步先判断时间是否超时
        if time.time() - start_time >= MINERU_POLL_TIMEOUT_SECONDS:
            logger.error(f'轮询获取{batch_id}的解析结果超时，耗时{time.time() - start_time}')
            raise TimeoutError(f'轮询获取{batch_id}的解析结果超时，耗时{time.time() - start_time}')

        # 没有超时就向接口发起请求获取解析结果
        try:
            poll_result = requests.get(result_url, headers=header)
        except Exception as e:
            logger.warning(f'向接口发起请求出现网络波动{str(e)},稍后再试')
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue

        # 网络状态判定
        # 1 2 3 不通过 4 本地重试 5 本地正常
        if poll_result.status_code != 200:
            # 5 给机会
            # 4 不给机会
            if 500 <= poll_result.status_code <= 600:
                logger.warning(f'与服务器连接出现网络状态波动{poll_result.status_code},请稍后再试')
                time.sleep(MINERU_POLL_INTERVAL_SECONDS)
                continue

            else:
                logger.error(f'解析{batch_id}时，与服务器连接出错,http的状态码是{poll_result.status_code}，业务终止')
                raise RuntimeError(
                    f'解析{batch_id}时，与服务器连接出错,http的状态码是{poll_result.status_code}，业务终止')

        # 业务状态判定
        poll_result_dict = poll_result.json()
        if poll_result_dict.get('code', -1) != 0:
            # 业务失败 业务终止
            logger.error(f'服务器网络连接正常，解析{batch_id}失败，业务状态码是{poll_result_dict.get('code', -1)}')
            raise RuntimeError(f'服务器网络连接正常，解析{batch_id}失败，业务状态码是{poll_result_dict.get('code', -1)}')

        # 获得解析结果和状态判定
        extract_result_list = poll_result_dict.get('data', {}).get('extract_result', [])
        if len(extract_result_list) == 0:
            logger.warning(f"解析结果extract_result_list为空,跳过本次!稍后再试")
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue
        extract_result = extract_result_list[0]
        state = extract_result.get('state')

        # 解析完成
        if state == 'done':
            full_zip_url = extract_result.get('full_zip_url')
            if not full_zip_url:
                # 不给机会
                logger.error(f'{batch_id}的解析已经完成，但是full_zip_url为空，业务失败 提前终止')
                raise ValueError(f'{batch_id}的解析已经完成，但是full_zip_url为空，业务失败 提前终止')

            return full_zip_url
        # 解析失败
        elif state == 'failed':
            logger.error(f'{batch_id}的解析任务失败，业务失败 提前终止')
            raise ValueError(f'{batch_id}的解析任务失败，业务失败 提前终止')
        # 其他状态 比如 running 正在解析
        else:
            logger.warning('本次解析正在进行中，暂未获得结果，耐心等待')
            time.sleep(MINERU_POLL_INTERVAL_SECONDS)
            continue


@step_log("download_and_extract_markdown")
def download_and_extract_markdown(zip_url: str, local_dir_obj: Path, file_name: str) -> Path:
    # 1下载数据
    response = requests.get(zip_url, timeout=MINERU_POLL_TIMEOUT_SECONDS)

    # 1.1与文件服务器连接 判定
    # response 的几个方法
    #     .status_code
    #     .json()  -> 服务器返回的json字符串 -> dict
    #     .text    -> 服务器返回的json字符串 -> str -> json.loads...
    #     .content -> 服务器返回的字节数据
    if response.status_code != 200:
        logger.error(f'解压文件地址{zip_url}下载出错，状态码是{response.status_code},业务终止')
        raise RuntimeError(f'解压文件地址{zip_url}下载出错，状态码是{response.status_code},业务终止')

    # 提前准备解压文件对象Path
    zip_file_obj: Path = local_dir_obj / f'{file_name}.zip'
    # 这步骤进行玩已经有，已经有压缩包了
    zip_file_obj.write_bytes(response.content)

    # 2 解压数据
    # 创建解压目录
    zip_extract_dir: Path = local_dir_obj / file_name

    if zip_extract_dir.is_dir():
        # 说明已经解压过 需要清空数据,避免脏数据
        # shutil-> shell utils
        # shutil.rmtree -> 递归清空 清空文件夹内容和文件夹本身
        shutil.rmtree(zip_extract_dir)

    # 无论是否存在 都需要解压
    zip_extract_dir.mkdir(parents=True, exist_ok=True)
    # 压缩 shutil.make_archive(压缩的文件名 stem ,压缩的格式 zip tar tar.gz , 那个文件夹进行压缩 地址)
    # 解压
    shutil.unpack_archive(zip_file_obj, zip_extract_dir)

    # 3重命名解压后的文件
    # 找到文件夹的指定类型文件 .md

    # .rglob()递归搜索
    md_obj_list: list[Path] = list(zip_extract_dir.rglob('*.md'))

    if len(md_obj_list) == 0:
        logger.error(f'在地址{zip_url}下载文件解压后没有md文件，业务无法继续进行')
        raise ValueError(f'在地址{zip_url}下载文件解压后没有md文件，业务无法继续进行')
    # 有就重命名
    # 同名
    for current_md_obj in md_obj_list:
        if current_md_obj.stem == file_name:
            logger.info(f"向指定地址:{zip_url}下载zip文件,解压后的文件名,等于原文件名{file_name},直接返回!!")
            return current_md_obj

    md_obj_path: Path = None
    # 如果文件名默认full.md
    for current_md_obj in md_obj_list:
        if current_md_obj.stem == 'full':
            md_obj_path = current_md_obj
            break
    # 随机名字
    if not md_obj_list:
        md_obj_path = md_obj_list[0]

    logger.info(f"触发了md文件的重命名机制,原名称:{md_obj_path.stem},目标名称:{file_name}")

    md_obj_path = md_obj_path.rename(md_obj_path.with_name(f"{file_name}.md"))

    return md_obj_path


@step_log('parse_pdf_to_markdown')
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务：
    1. 调用 MinerU
    2. 下载并解压解析结果
    3. 获取 Markdown 路径和正文内容
    4. 回写 md_path / md_content / local_dir
    """
    # 第一步先校验传入的路径和输出目录
    pdf_path_obj, local_dir_obj = validate_pdf_paths(state)
    # 调用MinerU 上传pdf 轮询得到最终亚索包地址
    zip_url: str = upload_pdf_and_poll(pdf_path_obj)
    # 根据zip_url下载并解压 返回到输出地址 重命名文件
    md_path_obj: Path = download_and_extract_markdown(zip_url, local_dir_obj, pdf_path_obj.stem)
    # 更新state状态
    state['md_path'] = str(md_path_obj)
    return state






