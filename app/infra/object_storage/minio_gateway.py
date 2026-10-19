from app.infra.config.providers import infra_config
from minio import Minio
from app.shared.clients.minio_utils import get_minio_client


class MinioGateway:
    # 获取桶名称
    @property
    def bucket_name(self):
        return infra_config.minio_config.bucket_name

    # 获取图片前缀
    @property
    def minio_img_dir(self):
        return infra_config.minio_config.minio_img_dir

    # 获取 minio_client
    @property
    def minio_client(self):
        return get_minio_client()

    # 封装拼接地址的函数
    def build_image_url(self, stem: str, object_name: str) -> str:
        # 1. 先决定协议 (修复了原来的优先级错误)
        protocol = "https://" if infra_config.minio_config.minio_secure else "http://"

        # 2. 正确拼接所有变量
        image_url = (
            f"{protocol}{infra_config.minio_config.endpoint}"
            f"/{self.bucket_name}{self.minio_img_dir}/{stem}/{object_name}"
        )

        print(f"生成的图片URL: {image_url}")
        return image_url


minio_gateway = MinioGateway()